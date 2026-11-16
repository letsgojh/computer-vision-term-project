"""메타데이터 검증과 메모 크롭 데이터셋."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T

from .boxes import Box, BoxRecord, expand_box, jitter_box, to_pixels

REQUIRED_COLUMNS = ["image_path", "label", "fake_type", "source_id", "split"]
POSTPROCESSES = ("original", "kakao", "resize", "recapture")
SPLITS = ("train", "val", "test")
FAKE_TYPES = ("real", "A", "B", "C", "unknown")
_PP_RE = re.compile(r"(?:^|/)processed/(original|kakao|resize|recapture)/")


def infer_postprocess(image_path: str) -> str:
    """경로의 processed/<종류>/ 에서 후처리 종류를 추론한다. 없으면 original."""
    m = _PP_RE.search(str(image_path).replace("\\", "/"))
    return m.group(1) if m else "original"


def read_metadata(csv_path, root, leakage_columns: Sequence[str] = ("item_id",),
                  verify_images: bool = True) -> pd.DataFrame:
    """메타데이터 CSV를 읽고 검증한다. _resolved_path 컬럼을 추가한다."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"메타데이터 CSV가 없습니다: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")
    if df.empty:
        raise ValueError("메타데이터에 행이 없습니다")
    for c in REQUIRED_COLUMNS:
        if (df[c].str.strip() == "").any():
            raise ValueError(f"'{c}' 컬럼에 빈 값이 있습니다")
    if not df.label.isin(["0", "1"]).all():
        raise ValueError("label은 0 또는 1이어야 합니다")
    df["label"] = df.label.astype(int)
    if not df.fake_type.isin(FAKE_TYPES).all():
        raise ValueError(f"fake_type은 {FAKE_TYPES} 중 하나여야 합니다")
    if ((df.fake_type == "real") != (df.label == 0)).any():
        raise ValueError("label과 fake_type이 맞지 않습니다 (real <-> 0)")
    if not df.split.isin(SPLITS).all():
        raise ValueError(f"split은 {SPLITS} 중 하나여야 합니다")

    inferred = df.image_path.map(infer_postprocess)
    if "postprocess" in df.columns:
        given = df.postprocess.str.strip()
        df["postprocess"] = given.where(given != "", inferred)
    else:
        df["postprocess"] = inferred
    if not df.postprocess.isin(POSTPROCESSES).all():
        raise ValueError(f"postprocess는 {POSTPROCESSES} 중 하나여야 합니다")
    if "domain" not in df.columns:
        df["domain"] = ""

    root = Path(root)
    df["_resolved_path"] = [str((root / p).resolve()) for p in df.image_path]
    if df._resolved_path.duplicated().any():
        raise ValueError("이미지 경로가 중복됩니다 (duplicate)")

    for col in ["source_id", *leakage_columns]:
        if col not in df.columns:
            continue
        sub = df[df[col].str.strip() != ""]
        multi = sub.groupby(col)["split"].nunique()
        bad = multi[multi > 1].index.tolist()
        if bad:
            raise ValueError(f"split 간 leak: '{col}' 값이 여러 split에 있습니다 {bad[:3]}")

    for p in df._resolved_path:
        if not Path(p).is_file():
            raise FileNotFoundError(f"이미지가 없습니다: {p}")
        if verify_images:
            with Image.open(p) as im:
                im.verify()
    return df


def select_rows(df: pd.DataFrame, split: str, exclude_types: Sequence[str] = (),
                postprocesses: Sequence[str] | None = None) -> pd.DataFrame:
    """split, fake_type 제외, 후처리 종류로 거른다. 비면 ValueError."""
    mask = df.split == split
    if exclude_types:
        mask &= ~df.fake_type.isin(list(exclude_types))
    if postprocesses is not None:
        mask &= df.postprocess.isin(list(postprocesses))
    out = df[mask].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"조건에 맞는 행이 없습니다: split={split}, "
                         f"exclude={list(exclude_types)}, postprocesses={postprocesses}")
    return out


def crop_region(image: Image.Image, box: Box, margin: float) -> Image.Image:
    """EXIF 회전, RGB 변환 후 박스를 margin만큼 넓혀 크롭한다 (평가용 크롭 영역)."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image.crop(to_pixels(expand_box(box, margin), *image.size))


class MemoCropTransform:
    """이미지와 박스를 받아 메모 영역 크롭 텐서를 만든다."""

    def __init__(self, input_size: int, mean, std, margin: float, train: bool = False,
                 jitter_shift: float = 0.0, jitter_scale: tuple[float, float] = (1.0, 1.0)):
        self.size = int(input_size)
        self.margin = margin
        self.train = train
        self.jitter_shift = jitter_shift
        self.jitter_scale = tuple(jitter_scale)
        self.fill = tuple(round(255 * m) for m in mean)
        self.post = T.Compose([T.ToTensor(), T.Normalize(list(mean), list(std))])

    def __call__(self, image: Image.Image, box: Box) -> torch.Tensor:
        if self.train:
            box = jitter_box(box, self.jitter_shift, self.jitter_scale)
        crop = crop_region(image, box, self.margin)
        crop = ImageOps.pad(crop, (self.size, self.size), Image.BICUBIC, color=self.fill)
        return self.post(crop)


def build_transform(config: dict, train: bool) -> MemoCropTransform:
    """설정에서 변환을 만든다. jitter는 train일 때만 켠다."""
    m, c = config["model"], config["crop"]
    return MemoCropTransform(
        m["input_size"], m["mean"], m["std"], c["margin"], train=train,
        jitter_shift=c["jitter_shift"] if train else 0.0,
        jitter_scale=tuple(c["jitter_scale"]) if train else (1.0, 1.0))


class MemoCropDataset(Dataset):
    """프레임의 각 행을 박스로 크롭해 돌려주는 데이터셋."""

    def __init__(self, frame: pd.DataFrame, boxes: dict[str, BoxRecord],
                 transform: MemoCropTransform):
        absent = [p for p in frame.image_path if p not in boxes]
        if absent:
            raise ValueError(
                f"박스가 없는 이미지 {len(absent)}개 (앞 3개: {absent[:3]}). "
                "python -m src.models.model2_memo.locate --metadata <CSV> 를 먼저 실행하세요")
        self.frame = frame.reset_index(drop=True)
        self.boxes = boxes
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int) -> dict:
        row = self.frame.iloc[i]
        rec = self.boxes[row.image_path]
        with Image.open(row._resolved_path) as im:
            image = self.transform(im, rec["box"])
        return {
            "image": image,
            "label": torch.tensor(float(row.label), dtype=torch.float32),
            "image_path": row.image_path,
            "fake_type": row.fake_type,
            "source_id": row.source_id,
            "postprocess": row.postprocess,
            "box_source": rec["source"],
            "domain": str(row.get("domain", "") or ""),
        }
