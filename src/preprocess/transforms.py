"""③ 후처리 변환. 진짜·가짜 모두에 똑같이 적용한다.

진짜는 카메라 JPEG, 가짜는 생성기 PNG라서 그대로 두면 모델이 "파일 형식·EXIF·해상도"만 보고
맞힐 수 있다. 그래서 모든 출력은 EXIF를 지운 JPEG로 같은 방식으로 다시 저장한다.
"""

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

# IJG 표준 휘도 양자화 테이블 (quality 50 기준). 순서와 무관하게 합으로만 비교한다.
_IJG_LUMA_SUM = sum([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
])


def load_image(path: Path) -> Image.Image:
    """EXIF 회전을 반영해 RGB로 연다. 반환 이미지에는 EXIF가 없다."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def resize_long_side(img: Image.Image, long_side: int) -> Image.Image:
    """긴 변을 long_side로 줄인다. 이미 작으면 그대로 둔다."""
    w, h = img.size
    scale = long_side / max(w, h)
    if scale >= 1:
        return img
    return img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)


def resize_short_side(img: Image.Image, short_side: int) -> Image.Image:
    """짧은 변을 short_side로 줄인다. 이미 작으면 그대로 둔다."""
    w, h = img.size
    scale = short_side / min(w, h)
    if scale >= 1:
        return img
    return img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)


def save_jpeg(img: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=quality)


def jpeg_bytes(img: Image.Image, quality: int) -> bytes:
    buf = BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def estimate_jpeg_quality(path: Path) -> int | None:
    """양자화 테이블로 JPEG 저장 품질을 추정한다 (IJG 기준). JPEG가 아니면 None.

    카톡으로 받은 사진에 이 함수를 돌려 configs/preprocess.yaml의 kakao.quality를 정한다.
    """
    with Image.open(path) as im:
        tables = getattr(im, "quantization", None)
    if not tables:
        return None
    scale = sum(tables[0]) / _IJG_LUMA_SUM * 100
    quality = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return max(1, min(100, round(quality)))
