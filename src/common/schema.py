"""역할 간 데이터 규약. docs/data_contract.md와 내용을 같게 유지할 것."""

import re

FAKE_TYPES = ("A", "B", "C")
POSTPROCESSES = ("original", "kakao", "resize", "recapture")
SPLITS = ("train", "val", "test")

# 파일명 규칙
REAL_NAME = re.compile(r"^(?P<photographer>[a-z0-9]+)_(?P<object>[a-z0-9]+)_(?P<idx>\d{3})\.jpe?g$")
FAKE_AB_NAME = re.compile(r"^fake(?P<type>[AB])_(?P<source>[a-z0-9]+_[a-z0-9]+_\d{3})_s(?P<seed>\d+)\.png$")
FAKE_C_NAME = re.compile(r"^fakeC_(?P<object>[a-z0-9]+)_(?P<idx>\d{3})_s(?P<seed>\d+)\.png$")

# CSV 컬럼
REAL_METADATA_COLUMNS = [
    "filename", "photographer", "object", "memo_text", "pen", "paper",
    "lighting", "background", "phone_model", "width", "height",
]
FAKE_MAPPING_COLUMNS = [
    "fake_filename", "fake_type", "source_filename", "mask_filename", "object",
    "prompt", "negative_prompt", "seed", "model", "created_by", "is_reinforcement",
]
SPLIT_COLUMNS = ["filename", "label", "fake_type", "object", "group_id", "photographer", "split"]
PREDICTION_COLUMNS = ["filename", "postprocess", "prob_fake"]


def fake_ab_name(fake_type: str, source_filename: str, seed: int) -> str:
    stem = source_filename.rsplit(".", 1)[0]
    return f"fake{fake_type}_{stem}_s{seed}.png"


def mask_name(fake_filename: str) -> str:
    return fake_filename.rsplit(".", 1)[0] + "_mask.png"


def fake_c_name(obj: str, idx: int, seed: int) -> str:
    return f"fakeC_{obj}_{idx:03d}_s{seed}.png"


def group_id(filename: str) -> str:
    """같은 원본에서 나온 진짜·가짜를 같은 split에 묶기 위한 키."""
    m = FAKE_AB_NAME.match(filename)
    if m:
        return m["source"]
    return filename.rsplit(".", 1)[0]
