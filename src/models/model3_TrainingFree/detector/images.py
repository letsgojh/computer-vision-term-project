"""이미지 목록과 AeroBlade/RDD용 중앙 크롭."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".jfif"}


def list_images(path: str | Path) -> list[Path]:
    root = Path(path)
    if not root.exists():
        return []
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTENSIONS else []
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=lambda p: str(p).lower())


def open_rgb(path: str | Path) -> Image.Image:
    """휴대폰 EXIF 회전을 적용하고, 투명 배경은 흰 배경 위에 얹는다."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.getchannel("A"))
            return background
        return image.convert("RGB")


def native_center_square(image: Image.Image, size: int) -> Image.Image:
    """짧은 변이 size보다 작을 때만 확대하고, 큰 사진은 줄이지 않고 중앙을 자른다.

    AeroBlade는 고주파가 망가지지 않도록 512 중앙 크롭을 쓴다.
    중고 사진이 512보다 작으면 부득이하게 확대한 뒤 자른다.
    """
    width, height = image.size
    if min(width, height) < size:
        scale = size / float(min(width, height))
        resized = (
            max(int(round(width * scale)), size),
            max(int(round(height * scale)), size),
        )
        image = image.resize(resized, Image.BICUBIC)
        width, height = image.size
    left = max((width - size) // 2, 0)
    top = max((height - size) // 2, 0)
    cropped = image.crop((left, top, left + size, top + size))
    if cropped.size != (size, size):
        cropped = cropped.resize((size, size), Image.BICUBIC)
    return cropped


def to_vae_tensor(image: Image.Image) -> torch.Tensor:
    """VAE 입력. 범위는 [-1, 1], 모양은 (1, 3, H, W)."""
    tensor = TF.to_tensor(image)
    return (tensor * 2.0 - 1.0).unsqueeze(0)


def prepare_vae_input(path: str | Path, size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    image = open_rgb(path)
    original_size = image.size
    cropped = native_center_square(image, size)
    return to_vae_tensor(cropped), original_size
