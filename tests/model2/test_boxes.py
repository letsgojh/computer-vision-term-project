import pytest
import torch

from src.models.model2_memo.boxes import (
    BOX_COLUMNS, FULL_BOX, expand_box, jitter_box, read_boxes, to_pixels, validate_box, write_boxes,
)


def test_expand_box_adds_margin_and_clamps():
    assert expand_box((0.4, 0.4, 0.6, 0.6), 0.15) == pytest.approx((0.37, 0.37, 0.63, 0.63))
    assert expand_box((0.0, 0.9, 0.2, 1.0), 0.5) == pytest.approx((0.0, 0.85, 0.3, 1.0))


@pytest.mark.parametrize("bad", [(0.5, 0.1, 0.4, 0.2), (-0.1, 0, 0.5, 0.5), (0, 0, 1.2, 1), (0.2, 0.2, 0.2, 0.4)])
def test_validate_box_rejects_bad(bad):
    with pytest.raises(ValueError):
        validate_box(bad)


def test_jitter_box_stays_valid_and_varies():
    g = torch.Generator().manual_seed(0)
    boxes = [jitter_box((0.4, 0.4, 0.6, 0.6), 0.05, (0.9, 1.1), g) for _ in range(50)]
    for b in boxes:
        validate_box(b)
        assert b[0] >= 0.38 - 1e-9 and b[2] <= 0.62 + 1e-9
    assert len(set(boxes)) > 1
    validate_box(jitter_box(FULL_BOX, 0.05, (0.9, 1.1), g))
    assert jitter_box((0.4, 0.4, 0.6, 0.6), 0.0, (1.0, 1.0), g) == pytest.approx((0.4, 0.4, 0.6, 0.6))


def test_to_pixels_min_one_pixel():
    l, t, r, b = to_pixels((0.5, 0.5, 0.5001, 0.5001), 160, 120)
    assert r - l >= 1 and b - t >= 1
    assert to_pixels((0.99, 0.99, 1.0, 1.0), 100, 100) == (99, 99, 100, 100)
    assert to_pixels(FULL_BOX, 64, 48) == (0, 0, 64, 48)
    assert to_pixels((0.25, 0.5, 0.75, 1.0), 100, 10) == (25, 5, 75, 10)


def test_boxes_csv_roundtrip(tmp_path):
    recs = [{"image_path": "b.jpg", "box": (0.1, 0.2, 0.3, 0.4), "score": 0.9, "source": "auto", "detector": "d"},
            {"image_path": "a.jpg", "box": FULL_BOX, "score": 0.0, "source": "fallback", "detector": ""}]
    write_boxes(recs, tmp_path / "x" / "boxes.csv")
    got = read_boxes(tmp_path / "x" / "boxes.csv")
    assert set(got) == {"a.jpg", "b.jpg"} and got["b.jpg"]["box"] == pytest.approx((0.1, 0.2, 0.3, 0.4))
    assert got["a.jpg"]["detector"] == "" and isinstance(got["b.jpg"]["score"], float)
    assert read_boxes(tmp_path / "missing.csv") == {}
    bad = tmp_path / "bad.csv"
    bad.write_text(",".join(BOX_COLUMNS) + "\na.jpg,0.1,0.1,0.5,0.5,0.9,bogus,d\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_boxes(bad)
    wrong_cols = tmp_path / "cols.csv"
    wrong_cols.write_text("image_path,x0\na.jpg,0.1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_boxes(wrong_cols)
