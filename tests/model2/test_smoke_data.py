import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.models.model2_memo.boxes import read_boxes
from src.models.model2_memo.scripts.make_smoke_data import make_smoke_data

COLS = ["image_path", "label", "fake_type", "source_id", "split", "domain", "item_id", "photographer", "postprocess"]


def _load(root, rel):
    return Image.open(root / rel).convert("RGB")


def _memo_diff(root, df, boxes, a, b):
    ia, ib = _load(root, a), _load(root, b)
    x0, y0, x1, y1 = boxes[a]["box"]
    w, h = ia.size
    sl = (slice(int(y0 * h), int(y1 * h)), slice(int(x0 * w), int(x1 * w)))
    da = np.asarray(ia, dtype=float)[sl]
    db = np.asarray(ib, dtype=float)[sl]
    return np.abs(da - db).mean()


def test_smoke_csv_schema(smoke_data):
    df = pd.read_csv(smoke_data, keep_default_na=False)
    assert list(df.columns) == COLS
    assert set(df.fake_type) == {"real", "A", "B", "C"}
    assert set(df.split) == {"train", "val", "test"}
    assert set(df.postprocess) == {"original", "kakao", "resize"}
    assert ((df.label == 0) == (df.fake_type == "real")).all()
    assert len(df) == (6 * 2 * 3 + 6 * 1) * 3   # (진짜+A+B + C) × 후처리 3종 = 126


def test_smoke_files_and_boxes_exist(smoke_data):
    df = pd.read_csv(smoke_data, keep_default_na=False)
    assert all((smoke_data.parent / p).is_file() for p in df.image_path)
    boxes = read_boxes(smoke_data.parent / "memo_boxes.csv")
    assert set(boxes) == set(df.image_path)
    assert {r["source"] for r in boxes.values()} == {"manual"}


def test_smoke_split_has_no_leakage(smoke_data):
    df = pd.read_csv(smoke_data, keep_default_na=False)
    assert (df.groupby("source_id").split.nunique() == 1).all()
    assert (df.groupby("item_id").split.nunique() == 1).all()


def test_fake_b_keeps_memo_and_fake_a_changes_it(smoke_data):
    root = smoke_data.parent
    df = pd.read_csv(smoke_data, keep_default_na=False)
    boxes = read_boxes(root / "memo_boxes.csv")
    orig = df[df.postprocess == "original"]
    real = orig[orig.fake_type == "real"].iloc[0]
    for ft, check in (("B", lambda d: d < 3), ("A", lambda d: d > 10)):
        row = orig[(orig.fake_type == ft) & (orig.source_id == real.source_id)].iloc[0]
        d = _memo_diff(root, df, boxes, real.image_path, row.image_path)
        assert check(d), (ft, d)


def test_resize_variant_is_half_size(smoke_data):
    root = smoke_data.parent
    df = pd.read_csv(smoke_data, keep_default_na=False)
    for stem in df.image_path.map(lambda p: p.rsplit("/", 1)[1]).unique()[:5]:
        w, h = Image.open(root / "data/processed/original" / stem).size
        rw, rh = Image.open(root / "data/processed/resize" / stem).size
        assert abs(rw - w // 2) <= 1 and abs(rh - h // 2) <= 1


def test_deterministic_and_validation(tmp_path):
    kw = dict(items=3, per_item=1, c_per_item=1, size=(80, 60), seed=3)
    a = make_smoke_data(tmp_path / "a", **kw)
    b = make_smoke_data(tmp_path / "b", **kw)
    assert a.read_bytes() == b.read_bytes()
    assert (tmp_path / "a/memo_boxes.csv").read_bytes() == (tmp_path / "b/memo_boxes.csv").read_bytes()
    with pytest.raises(ValueError):
        make_smoke_data(tmp_path / "c", items=2)
