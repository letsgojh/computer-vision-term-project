"""WaRPAD: DINOv2가 고주파 흔들림에 얼마나 버티는지.

이미지를 정사각으로 맞춘 뒤 224 패치로 나누고, 2단계 Haar 고주파를 빼서
CLS 토큰 코사인 유사도를 잰다. 유사도가 높을수록 실제 사진 쪽이다.
공식 실험의 평균 점수는 패치 유사도의 평균이다.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

from detector.haar import high_frequency
from detector.images import open_rgb

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DINO_REPOS = {
    "base": "facebook/dinov2-base",
    "large": "facebook/dinov2-large",
    "small": "facebook/dinov2-small",
}


def resolve_dino(name: str) -> str:
    return DINO_REPOS.get(name, name)


class WarpadScorer:
    def __init__(
        self,
        model_id: str,
        device: torch.device,
        prep_size: int = 448,
        patch_size: int = 224,
        alpha: float = 0.1,
        chunk_size: int = 4,
    ) -> None:
        from transformers import AutoModel

        if prep_size % patch_size != 0:
            raise ValueError(f"prep-size({prep_size})는 patch-size({patch_size})의 배수여야 합니다.")
        if patch_size % 4 != 0:
            raise ValueError("2단계 Haar를 위해 패치 크기는 4의 배수여야 합니다.")
        repo = resolve_dino(model_id)
        try:
            self.model = AutoModel.from_pretrained(repo)
        except Exception as error:
            raise RuntimeError(
                "DINOv2를 불러오지 못했습니다. 처음 실행이면 인터넷으로 가중치를 받습니다.\n"
                f"요청한 모델: {model_id} ({repo})\n{error}"
            ) from error
        self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.device = device
        self.prep_size = prep_size
        self.patch_size = patch_size
        self.alpha = alpha
        self.chunk_size = chunk_size
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.mean = mean.to(device)
        self.std = std.to(device)

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prepare(self, path: str | Path) -> torch.Tensor:
        image = open_rgb(path).resize((self.prep_size, self.prep_size), Image.BICUBIC)
        # 이미지 전체에 ImageNet 정규화를 적용한 뒤 패치로 나눈다. 패치마다 다시 정규화하지 않는다.
        tensor = to_tensor(image).unsqueeze(0).to(self.device)
        return (tensor - self.mean) / self.std

    def _patches(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, channels, _height, _width = tensor.shape
        windows = tensor.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        windows = windows.reshape(batch, channels, -1, self.patch_size, self.patch_size)
        windows = windows.transpose(1, 2).reshape(-1, channels, self.patch_size, self.patch_size)
        return windows

    @torch.inference_mode()
    def _cls(self, patches: torch.Tensor) -> torch.Tensor:
        tokens = []
        for start in range(0, patches.shape[0], self.chunk_size):
            batch = patches[start : start + self.chunk_size]
            hidden = self.model(pixel_values=batch).last_hidden_state[:, 0]
            tokens.append(hidden)
        return torch.cat(tokens, dim=0)

    @torch.inference_mode()
    def score_path(self, path: str | Path) -> float:
        patches = self._patches(self._prepare(path))
        perturbed = patches - self.alpha * high_frequency(patches, levels=2)
        original = self._cls(patches)
        changed = self._cls(perturbed)
        similarity = F.cosine_similarity(original, changed, dim=-1)
        return float(similarity.mean().item())
