"""메타데이터 검증과 메모 크롭 데이터셋 테스트."""
import pandas as pd
import pytest
import torch
from PIL import Image

from src.models.model2_memo.boxes import read_boxes
from src.models.model2_memo.dataset import (
    MemoCropDataset, MemoCropTransform, build_transform, crop_region, infer_postprocess,
    read_metadata, select_rows,
)

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _raw(smoke_data):
    return pd.read_csv(smoke_data, dtype=str, keep_default_na=False)


def _write(df, tmp_path, smoke_data):
    """고친 CSV를 원본과 같은 폴더 기준으로 쓰기 위해 tmp에 두고 root는 원본 폴더를 쓴다."""
    p = tmp_path / "meta.csv"
    df.to_csv(p, index=False)
    return p


def test_infer_postprocess():
    assert infer_postprocess("data/processed/kakao/a.jpg") == "kakao"
    assert infer_postprocess("data\\processed\\resize\\a.jpg") == "resize"
    assert infer_postprocess("x/processed/recapture/b.jpg") == "recapture"
    assert infer_postprocess("data/real/a.jpg") == "original"


def test_read_metadata_smoke_ok(smoke_data):
    df = read_metadata(smoke_data, smoke_data.parent)
    assert len(df) == 126 and "_resolved_path" in df
    assert set(df.postprocess) == {"original", "kakao", "resize"}


def _mut_label(df):
    df.loc[0, "label"] = "2"


def _mut_fake_type(df):
    i = df.index[df.label == "0"][0]
    df.loc[i, "fake_type"] = "A"


def _mut_split(df):
    df.loc[0, "split"] = "dev"


def _mut_src_leak(df):
    t = df.index[df.split == "test"][0]
    df.loc[t, "source_id"] = df.loc[df.split == "train", "source_id"].iloc[0]


def _mut_item_leak(df):
    t = df.index[df.split == "test"][0]
    df.loc[t, "item_id"] = df.loc[df.split == "train", "item_id"].iloc[0]


@pytest.mark.parametrize("mut,match", [
    (_mut_label, None), (_mut_fake_type, None), (_mut_split, None),
    (_mut_src_leak, "leak"), (_mut_item_leak, "leak"),
    (lambda df: df.pop("split"), None),
    (lambda df: df.__setitem__("image_path", df.image_path.iloc[0]), "dup|중복"),
])
def test_read_metadata_rejects(smoke_data, tmp_path, mut, match):
    df = _raw(smoke_data)
    mut(df)
    p = _write(df, tmp_path, smoke_data)
    with pytest.raises(ValueError, match=match):
        read_metadata(p, smoke_data.parent)


def test_read_metadata_missing_files(smoke_data, tmp_path):
    with pytest.raises(FileNotFoundError):
        read_metadata(tmp_path / "none.csv", smoke_data.parent)
    df = _raw(smoke_data)
    df.loc[0, "image_path"] = "data/nope.jpg"
    with pytest.raises(FileNotFoundError):
        read_metadata(_write(df, tmp_path, smoke_data), smoke_data.parent)


def test_read_metadata_infers_postprocess_when_column_missing(smoke_data, tmp_path):
    df = _raw(smoke_data).drop(columns=["postprocess"])
    out = read_metadata(_write(df, tmp_path, smoke_data), smoke_data.parent)
    assert (out.postprocess == out.image_path.map(infer_postprocess)).all()


def test_select_rows_excludes_b_and_filters_postprocess(smoke_data):
    df = read_metadata(smoke_data, smoke_data.parent)
    tr = select_rows(df, "train", ["B"], ["original"])
    assert "B" not in set(tr.fake_type) and set(tr.postprocess) == {"original"}
    te = select_rows(df, "test")
    assert "B" in set(te.fake_type) and set(te.postprocess) == {"original", "kakao", "resize"}
    with pytest.raises(ValueError):
        select_rows(df, "train", ["B"], ["recapture"])


def test_transform_eval_shape_and_deterministic():
    img = Image.new("RGB", (160, 120), (120, 80, 40))
    t = MemoCropTransform(64, MEAN, STD, 0.15)
    a, b = t(img, (0.2, 0.2, 0.6, 0.7)), t(img, (0.2, 0.2, 0.6, 0.7))
    assert a.shape == (3, 64, 64) and torch.equal(a, b)


def test_transform_train_jitter_varies():
    img = Image.new("RGB", (200, 100))
    img.putdata([(x * 255 // 199, 0, 0) for _ in range(100) for x in range(200)])
    t = MemoCropTransform(64, MEAN, STD, 0.15, train=True, jitter_shift=0.05, jitter_scale=(0.9, 1.1))
    assert not torch.equal(t(img, (0.2, 0.2, 0.6, 0.7)), t(img, (0.2, 0.2, 0.6, 0.7)))


def test_transform_tiny_edge_box():
    out = MemoCropTransform(64, MEAN, STD, 0.15)(Image.new("RGB", (160, 120)), (0.999, 0.999, 1.0, 1.0))
    assert out.shape == (3, 64, 64) and torch.isfinite(out).all()


def test_transform_handles_non_rgb():
    t = MemoCropTransform(64, MEAN, STD, 0.15)
    for mode in ["L", "RGBA"]:
        assert t(Image.new(mode, (50, 40)), (0.1, 0.1, 0.9, 0.9)).shape == (3, 64, 64)


def test_crop_uses_box_region():
    img = Image.new("RGB", (100, 100), (0, 0, 0))
    img.paste((255, 255, 255), (50, 0, 100, 100))
    t = MemoCropTransform(32, MEAN, STD, 0.0)
    m, s = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)
    assert (t(img, (0.6, 0.2, 0.9, 0.5)) * s + m).mean() > 0.9
    assert (t(img, (0.1, 0.2, 0.4, 0.5)) * s + m).mean() < 0.1


def test_crop_region_matches_box():
    img = Image.new("RGB", (100, 50), (0, 0, 0))
    img.paste((255, 255, 255), (50, 0, 100, 50))
    crop = crop_region(img, (0.5, 0.0, 1.0, 1.0), 0.0)
    assert crop.size == (50, 50)
    assert crop.getpixel((10, 10)) == (255, 255, 255)
    assert crop_region(Image.new("L", (40, 40)), (0.0, 0.0, 0.5, 0.5), 0.0).mode == "RGB"


def test_build_transform_jitter_only_in_train():
    cfg = {"model": {"input_size": 32, "mean": MEAN, "std": STD},
           "crop": {"margin": 0.15, "jitter_shift": 0.05, "jitter_scale": [0.9, 1.1]}}
    assert build_transform(cfg, True).jitter_scale == (0.9, 1.1)
    img = Image.new("RGB", (100, 100))
    img.putdata([(x * 255 // 99, 0, 0) for _ in range(100) for x in range(100)])
    ev = build_transform(cfg, False)
    assert torch.equal(ev(img, (0.2, 0.2, 0.6, 0.7)), ev(img, (0.2, 0.2, 0.6, 0.7)))


def test_dataset_missing_boxes_raises(smoke_data):
    df = read_metadata(smoke_data, smoke_data.parent)
    boxes = read_boxes(smoke_data.parent / "memo_boxes.csv")
    boxes.pop(df.image_path[0])
    with pytest.raises(ValueError, match="locate"):
        MemoCropDataset(df, boxes, MemoCropTransform(64, MEAN, STD, 0.15))


def test_dataset_item_fields(smoke_data):
    df = read_metadata(smoke_data, smoke_data.parent)
    ds = MemoCropDataset(df, read_boxes(smoke_data.parent / "memo_boxes.csv"),
                         MemoCropTransform(64, MEAN, STD, 0.15))
    item = ds[0]
    assert {"image", "label", "image_path", "fake_type", "source_id", "postprocess",
            "box_source", "domain"} <= set(item)
    assert item["label"].dtype == torch.float32 and item["box_source"] == "manual"
    assert item["image"].shape == (3, 64, 64) and len(ds) == len(df)
