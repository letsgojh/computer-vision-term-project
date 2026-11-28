"""메모 박스 유틸. 박스는 EXIF 회전 후 이미지 기준 0~1 정규화 (x0, y0, x1, y1)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

Box = tuple[float, float, float, float]
FULL_BOX: Box = (0.0, 0.0, 1.0, 1.0)
BOX_COLUMNS = ["image_path", "x0", "y0", "x1", "y1", "score", "source", "detector"]
BOX_SOURCES = ("auto", "manual", "fallback")

# {"image_path": str, "box": Box, "score": float, "source": str, "detector": str}
BoxRecord = dict


def validate_box(box) -> Box:
    """float로 바꾸고 0<=x0<x1<=1, 0<=y0<y1<=1이 아니면 ValueError."""
    x0, y0, x1, y1 = (float(v) for v in box)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"잘못된 박스: {tuple(box)}")
    return (x0, y0, x1, y1)


def _clamp(v: float) -> float:
    return min(1.0, max(0.0, v))


def expand_box(box: Box, margin: float) -> Box:
    """폭·높이의 margin 배만큼 사방으로 넓히고 [0, 1]로 자른다."""
    x0, y0, x1, y1 = validate_box(box)
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    return (_clamp(x0 - dx), _clamp(y0 - dy), _clamp(x1 + dx), _clamp(y1 + dy))


def jitter_box(box: Box, shift: float, scale_range: tuple[float, float],
               generator: torch.Generator | None = None) -> Box:
    """중심 이동과 크기 변경을 무작위로 준다. 결과는 항상 유효하다."""
    x0, y0, x1, y1 = validate_box(box)
    w, h = x1 - x0, y1 - y0
    u = torch.rand(4, generator=generator).tolist()
    lo, hi = scale_range
    cx = (x0 + x1) / 2 + w * (-shift + 2 * shift * u[0])
    cy = (y0 + y1) / 2 + h * (-shift + 2 * shift * u[1])
    nw = w * (lo + (hi - lo) * u[2])
    nh = h * (lo + (hi - lo) * u[3])
    out = (_clamp(cx - nw / 2), _clamp(cy - nh / 2), _clamp(cx + nw / 2), _clamp(cy + nh / 2))
    try:
        return validate_box(out)
    except ValueError:
        return (x0, y0, x1, y1)  # 면적 0이면 원본 박스


def _span(lo: float, hi: float, size: int) -> tuple[int, int]:
    a = min(max(math.floor(lo * size), 0), size)
    b = min(max(math.ceil(hi * size), 0), size)
    if b - a < 1:  # 최소 1px, 이미지 안에서 넓힌다
        if a >= size:
            a = size - 1
        b = a + 1
    return a, b


def to_pixels(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    """정규화 박스를 (left, top, right, bottom) 픽셀로 바꾼다. 최소 1px."""
    x0, y0, x1, y1 = box
    left, right = _span(x0, x1, width)
    top, bottom = _span(y0, y1, height)
    return left, top, right, bottom


def read_boxes(path: str | Path) -> dict[str, BoxRecord]:
    """박스 CSV를 image_path 키 dict로 읽는다. 파일이 없으면 {}."""
    p = Path(path)
    if not p.is_file():
        return {}
    df = pd.read_csv(p, keep_default_na=False, dtype=str)
    missing = [c for c in BOX_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"박스 CSV에 컬럼이 없습니다: {missing}")
    out: dict[str, BoxRecord] = {}
    for row in df.to_dict("records"):
        if row["source"] not in BOX_SOURCES:
            raise ValueError(f"잘못된 source 값: {row['source']!r} ({row['image_path']})")
        box = validate_box([row[c] for c in ("x0", "y0", "x1", "y1")])
        out[row["image_path"]] = {
            "image_path": row["image_path"], "box": box, "score": float(row["score"]),
            "source": str(row["source"]), "detector": str(row["detector"]),
        }
    return out


def write_boxes(records: Iterable[BoxRecord], path: str | Path) -> None:
    """BOX_COLUMNS 순서로 image_path 순 정렬해 쓴다. 상위 폴더를 만든다."""
    rows = []
    for r in records:
        x0, y0, x1, y1 = r["box"]
        rows.append({"image_path": r["image_path"], "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                     "score": r["score"], "source": r["source"], "detector": r["detector"]})
    rows.sort(key=lambda r: r["image_path"])
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=BOX_COLUMNS).to_csv(p, index=False, encoding="utf-8")
