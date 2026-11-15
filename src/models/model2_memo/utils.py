"""모델 2 공용 유틸: 설정 읽기, 장치 선택, 시드 고정, JSON 저장."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    """YAML 설정을 읽는다. 파일이 없으면 FileNotFoundError."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"설정 파일이 없습니다: {p}")
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def select_device(name: str = "auto") -> torch.device:
    """auto는 CUDA -> MPS -> CPU 순. 지정한 장치를 못 쓰면 RuntimeError."""
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if mps_ok else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다.")
    if name == "mps" and not mps_ok:
        raise RuntimeError("MPS를 사용할 수 없습니다.")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    """random, numpy, torch 시드를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: str | Path, obj) -> None:
    """UTF-8 JSON(indent 2)으로 저장한다. 상위 폴더를 만든다."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
