import pandas as pd
import pytest
from PIL import Image

from src.preprocess import postprocess, split
from src.preprocess.make_dummy import make
from src.preprocess.transforms import estimate_jpeg_quality

CFG = {"jpeg_quality": 95, "kakao": {"long_side": 320, "quality": 70}, "resize": {"short_side": 128}}


@pytest.fixture(scope="module")
def dummy(tmp_path_factory):
    root = tmp_path_factory.mktemp("dummy")
    make(root, per_object=4)
    meta = root / "metadata"
    rows = split.build_rows(
        pd.read_csv(meta / "real_metadata.csv", dtype=str, keep_default_na=False),
        pd.read_csv(meta / "fake_mapping.csv", dtype=str, keep_default_na=False),
    )
    return root, rows


# ---- postprocess ----

def test_postprocess_outputs_same_format_for_real_and_fake(dummy):
    root, _ = dummy
    data = root / "data"
    written = postprocess.run(data, CFG, postprocess.AUTO_VARIANTS)
    n_sources = len(postprocess.find_sources(data))
    assert all(n == n_sources for n in written.values())

    real = data / "processed" / "original" / "jaehwan_mouse_001.jpg"
    fake = data / "processed" / "original" / "fakeA_jaehwan_mouse_001_s1.jpg"
    for p in (real, fake):
        with Image.open(p) as im:
            assert im.format == "JPEG"
            assert not im.getexif()  # 진짜의 EXIF가 남으면 모델이 그걸로 맞힌다


def test_postprocess_sizes_and_quality(dummy):
    root, _ = dummy
    out = root / "data" / "processed"
    with Image.open(out / "kakao" / "jaehwan_mouse_001.jpg") as im:
        assert max(im.size) == 320
    with Image.open(out / "resize" / "jaehwan_mouse_001.jpg") as im:
        assert min(im.size) == 128
    assert abs(estimate_jpeg_quality(out / "kakao" / "jaehwan_mouse_001.jpg") - 70) <= 2
    assert abs(estimate_jpeg_quality(out / "original" / "jaehwan_mouse_001.jpg") - 95) <= 2


def test_postprocess_skips_existing(dummy):
    root, _ = dummy
    written = postprocess.run(root / "data", CFG, postprocess.AUTO_VARIANTS)
    assert sum(written.values()) == 0


def test_recapture_matches_by_stem(dummy, tmp_path):
    root, _ = dummy
    Image.new("RGB", (64, 64)).save(tmp_path / "jaehwan_mouse_001.jpg")
    Image.new("RGB", (64, 64)).save(tmp_path / "IMG_1234.jpg")
    n, unknown = postprocess.ingest_recapture(root / "data", tmp_path, CFG)
    assert n == 1 and unknown == ["IMG_1234.jpg"]


# ---- split ----

def test_build_has_no_leakage(dummy):
    _, rows = dummy
    result, skipped = split.build(rows, [0.7, 0.15, 0.15], seed=42)
    assert split.check(result) == []
    assert set(result["split"]) == {"train", "val", "test"}
    assert len(result) + len(skipped) == len(rows)


def test_reinforcement_only_goes_to_train(dummy):
    _, rows = dummy
    result, skipped = split.build(rows, [0.7, 0.15, 0.15], seed=42)
    reinforced = rows.loc[rows["is_reinforcement"] == "1", "filename"]
    in_result = result[result["filename"].isin(reinforced)]
    assert (in_result["split"] == "train").all()
    assert len(skipped) > 0  # val/test 원본에서 나온 보강분은 빠져야 함
    assert set(skipped["filename"]) == set(reinforced) - set(in_result["filename"])


def test_test_photographers_are_held_out(dummy):
    _, rows = dummy
    result, _ = split.build(rows, [0.7, 0.15, 0.15], seed=42, test_photographers=["taei"])
    taei_objects = set(rows.loc[rows["photographer"] == "taei", "object"])
    assert (result.loc[result["object"].isin(taei_objects), "split"] == "test").all()
    assert split.check(result) == []


def test_append_keeps_existing_rows_fixed(dummy):
    _, rows = dummy
    base, _ = split.build(rows.iloc[: len(rows) // 2], [0.7, 0.15, 0.15], seed=0)
    result, skipped = split.append(base, rows)
    merged = result.merge(base, on="filename", suffixes=("", "_before"))
    assert (merged["split"] == merged["split_before"]).all()
    added = result[~result["filename"].isin(base["filename"])]
    assert (added["split"] == "train").all()
    assert len(added) + len(skipped) == len(rows) - len(base)
    assert split.check(result) == []


def test_check_detects_group_leakage(dummy):
    _, rows = dummy
    result, _ = split.build(rows, [0.7, 0.15, 0.15], seed=42)
    fake_row = result[result["fake_type"] == "A"].index[0]
    result.loc[fake_row, "split"] = "test" if result.loc[fake_row, "split"] != "test" else "train"
    errors = split.check(result)
    assert any("원본 그룹" in e for e in errors)


def test_fake_with_unknown_source_is_rejected(dummy):
    root, _ = dummy
    meta = root / "metadata"
    real = pd.read_csv(meta / "real_metadata.csv", dtype=str, keep_default_na=False)
    fake = pd.read_csv(meta / "fake_mapping.csv", dtype=str, keep_default_na=False)
    fake.loc[0, "source_filename"] = "nobody_nothing_999.jpg"
    with pytest.raises(ValueError, match="real_metadata"):
        split.build_rows(real, fake)
