"""결과표 형식을 보여 주는 합성 입력. 모델 성능이 아니다."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.schema import FAKE_MAPPING_COLUMNS, REAL_METADATA_COLUMNS, SPLIT_COLUMNS


def _score(kind: str, postprocess: str, meta: dict, model: str, rng: np.random.Generator) -> float:
    """간격을 크게 두어, 형식 예시에서 약한 칸과 오탐 집중이 안정적으로 보이게 한다."""
    decay = {"original": 0.0, "kakao": 0.12, "resize": 0.04, "recapture": 0.28}[postprocess]
    if model == "model1":
        if kind == "real":
            base = 0.82 if meta["pen"] == "highlighter" and meta["lighting"] == "dark" else 0.12
        elif kind == "A":
            base = 0.22 if meta["pen"] == "highlighter" and meta["lighting"] == "dark" else 0.82
        elif kind == "B":
            base = 0.90
        else:
            base = 0.78
        if kind != "real":
            base -= decay
    else:
        if kind == "real":
            base = 0.72 if meta["phone_model"] == "iPhone15" else 0.12
        elif kind == "A":
            base = 0.90
        elif kind == "B":
            base = 0.40
        else:
            base = 0.74
        if kind != "real":
            base -= decay
    return float(np.clip(base + rng.normal(0, 0.01), 0.02, 0.98))


def build_demo(root: Path, seed: int = 7) -> dict[str, Path]:
    """예측·split·메타데이터를 root 아래에 쓰고 경로를 돌려준다."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    real_rows = []

    def add_reals(split, n, photographer, obj, pen, lighting, phone, start):
        made = []
        for offset in range(n):
            filename = f"{photographer}_{obj}_{start + offset:03d}.jpg"
            made.append(
                {
                    "filename": filename,
                    "photographer": photographer,
                    "object": obj,
                    "memo_text": "2026.11.04 / demo",
                    "pen": pen,
                    "paper": "postit",
                    "lighting": lighting,
                    "background": "desk",
                    "phone_model": phone,
                    "width": 3024,
                    "height": 4032,
                    "split": split,
                    "group_id": filename.rsplit(".", 1)[0],
                    "label": 0,
                    "fake_type": "real",
                }
            )
        real_rows.extend(made)
        return made

    add_reals("test", 8, "jaehwan", "mouse", "highlighter", "dark", "GalaxyS23", 1)
    add_reals("test", 8, "taei", "cup", "ballpoint", "natural", "iPhone15", 1)
    add_reals("test", 8, "taei", "book", "ballpoint", "natural", "GalaxyS23", 1)
    add_reals("val", 4, "jaehwan", "mouse", "highlighter", "dark", "GalaxyS23", 21)
    add_reals("val", 4, "taei", "cup", "ballpoint", "natural", "iPhone15", 21)
    add_reals("train", 2, "taei", "book", "ballpoint", "natural", "GalaxyS23", 21)

    images = []
    mapping = []
    for real in real_rows:
        images.append({**real, "kind": "real", "meta": real})
    fake_sources = [row for row in real_rows if row["split"] in {"test", "val"} and row["object"] in {"mouse", "cup"}]
    for index, real in enumerate(fake_sources, start=1):
        for fake_type in ("A", "B"):
            filename = f"fake{fake_type}_{real['group_id']}_s{index}.png"
            images.append(
                {
                    "filename": filename,
                    "photographer": real["photographer"],
                    "object": real["object"],
                    "split": real["split"],
                    "group_id": real["group_id"],
                    "label": 1,
                    "fake_type": fake_type,
                    "kind": fake_type,
                    "meta": real,
                }
            )
            mapping.append(
                {
                    "fake_filename": filename,
                    "fake_type": fake_type,
                    "source_filename": real["filename"],
                    "mask_filename": filename.replace(".png", "_mask.png"),
                    "object": real["object"],
                    "prompt": "demo",
                    "negative_prompt": "",
                    "seed": index,
                    "model": "demo",
                    "created_by": "demo",
                    "is_reinforcement": 0,
                }
            )
    for index in range(1, 17):
        obj = "mouse" if index <= 8 else "cup"
        filename = f"fakeC_{obj}_{index:03d}_s{index}.png"
        images.append(
            {
                "filename": filename,
                "photographer": "sdxl",
                "object": obj,
                "split": "test",
                "group_id": filename.rsplit(".", 1)[0],
                "label": 1,
                "fake_type": "C",
                "kind": "C",
                "meta": {"pen": "", "lighting": "", "phone_model": ""},
            }
        )
        mapping.append(
            {
                "fake_filename": filename,
                "fake_type": "C",
                "source_filename": "",
                "mask_filename": "",
                "object": obj,
                "prompt": "demo certification photo",
                "negative_prompt": "",
                "seed": index,
                "model": "demo",
                "created_by": "demo",
                "is_reinforcement": 0,
            }
        )
    for index in range(1, 5):
        filename = f"fakeC_book_{index:03d}_s{index}.png"
        images.append(
            {
                "filename": filename,
                "photographer": "sdxl",
                "object": "book",
                "split": "val",
                "group_id": filename.rsplit(".", 1)[0],
                "label": 1,
                "fake_type": "C",
                "kind": "C",
                "meta": {"pen": "", "lighting": "", "phone_model": ""},
            }
        )
        mapping.append(
            {
                "fake_filename": filename,
                "fake_type": "C",
                "source_filename": "",
                "mask_filename": "",
                "object": "book",
                "prompt": "demo certification photo",
                "negative_prompt": "",
                "seed": index,
                "model": "demo",
                "created_by": "demo",
                "is_reinforcement": 0,
            }
        )

    split_path = root / "split.csv"
    pd.DataFrame(
        [
            {
                "filename": row["filename"],
                "label": row["label"],
                "fake_type": row["fake_type"],
                "object": row["object"],
                "group_id": row["group_id"],
                "photographer": row["photographer"],
                "split": row["split"],
            }
            for row in images
        ],
        columns=SPLIT_COLUMNS,
    ).to_csv(split_path, index=False)

    meta_path = root / "real_metadata.csv"
    pd.DataFrame(real_rows, columns=REAL_METADATA_COLUMNS).to_csv(meta_path, index=False)
    map_path = root / "fake_mapping.csv"
    pd.DataFrame(mapping, columns=FAKE_MAPPING_COLUMNS).to_csv(map_path, index=False)

    pred_dir = root / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    postprocesses = ["original", "kakao", "resize", "recapture"]
    for model in ("model1", "model2"):
        records = []
        for row in images:
            if row["split"] == "train":
                continue
            for postprocess in postprocesses:
                records.append(
                    {
                        "filename": row["filename"],
                        "postprocess": postprocess,
                        "prob_fake": _score(row["kind"], postprocess, row["meta"], model, rng),
                    }
                )
        path = pred_dir / f"{model}_baseline.csv"
        pd.DataFrame(records).to_csv(path, index=False)

    return {
        "predictions": pred_dir,
        "split": split_path,
        "real_metadata": meta_path,
        "fake_mapping": map_path,
    }
