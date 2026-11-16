"""locate.py 테스트 (네트워크 없이)."""
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.models.model2_memo.boxes import FULL_BOX, read_boxes, validate_box, write_boxes
from src.models.model2_memo.dataset import read_metadata
from src.models.model2_memo.locate import (
    HFZeroShotDetector, choose_box, detector_fingerprint, locate_frame, locate_report,
    manual_box_for, read_manual_boxes, run,
)
from src.models.model2_memo.utils import load_config

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "model2_memo.yaml"


class FakeDetector:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return []


class StemDetector:
    """fakeA는 탐지 실패, 나머지는 박스 하나. 호출 횟수를 센다."""
    name = "fake"

    def __init__(self):
        self.calls = 0
        self.current = ""

    def detect(self, image):
        self.calls += 1
        if Path(image.filename).stem.startswith("fakeA"):
            return []
        return [((0.2, 0.2, 0.5, 0.5), 0.8)]


def _frame(smoke_data):
    return read_metadata(smoke_data, smoke_data.parent)


def _rows(df):
    return df.assign(stem=df.image_path.map(lambda p: Path(p).stem))


def test_choose_box_picks_highest_valid_area():
    dets = [(FULL_BOX, 0.9), ((0.1, 0.1, 0.2, 0.2), 0.5), ((0.5, 0.5, 0.7, 0.7), 0.7)]
    assert choose_box(dets, 0.002, 0.8) == ((0.5, 0.5, 0.7, 0.7), 0.7)
    assert choose_box([((0.1, 0.1, 0.11, 0.11), 0.9)], 0.05, 0.8) is None
    assert choose_box([], 0.002, 0.8) is None


def test_read_manual_boxes_and_lookup(tmp_path):
    p = tmp_path / "m.csv"
    head = "stem,postprocess,x0,y0,x1,y1,note\n"
    p.write_text(head + "s1,,0.1,0.1,0.3,0.3,\ns1,recapture,0.2,0.2,0.4,0.4,\n", encoding="utf-8")
    m = read_manual_boxes(p)
    assert manual_box_for(m, "data/processed/kakao/s1.jpg", "kakao") == (0.1, 0.1, 0.3, 0.3)
    assert manual_box_for(m, "data/processed/recapture/s1.jpg", "recapture") == (0.2, 0.2, 0.4, 0.4)
    assert manual_box_for(m, "data/processed/original/s2.jpg", "original") is None
    # 빈 postprocess 박스는 recapture에 적용하지 않는다
    only = tmp_path / "o.csv"
    only.write_text(head + "s1,,0.1,0.1,0.3,0.3,\n", encoding="utf-8")
    assert manual_box_for(read_manual_boxes(only), "x/s1.jpg", "recapture") is None
    assert read_manual_boxes(tmp_path / "none.csv") == {}
    dup = tmp_path / "d.csv"
    dup.write_text(head + "s1,,0.1,0.1,0.3,0.3,\ns1,,0.2,0.2,0.4,0.4,\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_manual_boxes(dup)
    bad = tmp_path / "b.csv"
    bad.write_text(head + "s1,foo,0.1,0.1,0.3,0.3,\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_manual_boxes(bad)
    badbox = tmp_path / "bb.csv"
    badbox.write_text(head + "s1,,0.5,0.1,0.3,0.3,\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_manual_boxes(badbox)


def test_locate_frame_manual_only_in_train_and_fallback(smoke_data):
    df = _rows(_frame(smoke_data))
    real = df[df.fake_type == "real"]
    tr = real[real.split == "train"].stem.iloc[0]
    te = real[real.split == "test"].stem.iloc[0]
    manual = {(tr, ""): (0.1, 0.1, 0.3, 0.3), (te, ""): (0.1, 0.1, 0.3, 0.3)}
    det = StemDetector()
    recs = locate_frame(df, lambda: det, manual, {}, ["train"], 0.002, 0.8, "fake")
    assert len(recs) == len(df)
    assert len({r["image_path"] for r in recs}) == len(df)
    by_path = {r["image_path"]: r for r in recs}
    for _, row in df.iterrows():
        r = by_path[row.image_path]
        if row.stem == tr and row.split == "train":
            assert r["source"] == "manual" and r["score"] == 1.0 and r["detector"] == "manual"
        elif row.stem == te:
            assert r["source"] == "auto"
        if row.stem.startswith("fakeA"):
            assert r["source"] == "fallback" and r["box"] == FULL_BOX and r["score"] == 0.0
    assert any(r["source"] == "manual" for r in recs)
    assert all(r["detector"] == "fake" for r in recs if r["source"] != "manual")


def test_locate_frame_uses_cache_and_lazy_detector(smoke_data):
    df = _frame(smoke_data)
    cached = {p: {"image_path": p, "box": (0.1, 0.1, 0.4, 0.4), "score": 0.9,
                  "source": "auto", "detector": "fp1"} for p in df.image_path}

    def boom():
        raise AssertionError("탐지기를 부르면 안 됩니다")

    recs = locate_frame(df, boom, {}, cached, ["train"], 0.002, 0.8, "fp1")
    assert all(r["detector"] == "fp1" for r in recs)
    calls = []
    det = FakeDetector()

    def factory():
        calls.append(1)
        return det

    recs = locate_frame(df, factory, {}, cached, ["train"], 0.002, 0.8, "fp1", overwrite=True)
    assert len(calls) == 1 and det.calls == len(df)
    # manual 캐시는 재사용하지 않고 다시 탐지한다
    one = df.image_path.iloc[0]
    mc = {one: {"image_path": one, "box": (0.1, 0.1, 0.4, 0.4), "score": 1.0,
                "source": "manual", "detector": "manual"}}
    det2 = FakeDetector()
    recs = locate_frame(df.iloc[:1], lambda: det2, {}, mc, ["train"], 0.002, 0.8, "fp1")
    assert det2.calls == 1 and recs[0]["source"] == "fallback"


def test_locate_frame_redetects_on_fingerprint_change(smoke_data):
    df = _frame(smoke_data)
    cached = {p: {"image_path": p, "box": (0.1, 0.1, 0.4, 0.4), "score": 0.9,
                  "source": "auto", "detector": "fp-old"} for p in df.image_path}
    det, stats = StemDetector(), {}
    recs = locate_frame(df, lambda: det, {}, cached, ["train"], 0.002, 0.8, "fp-new", stats=stats)
    assert det.calls == len(df)
    assert all(r["detector"] == "fp-new" for r in recs)
    assert stats == {"reused": 0, "redetected": len(df)}


def test_detector_fingerprint_changes_with_settings():
    cfg = load_config(CONFIG)["locate"]
    base = detector_fingerprint(cfg)
    assert base.startswith(cfg["model_id"] + "#") and len(base.split("#")[1]) == 8
    assert detector_fingerprint(dict(cfg)) == base
    assert detector_fingerprint({**cfg, "queries": ["a memo"]}) != base
    assert detector_fingerprint({**cfg, "box_threshold": cfg["box_threshold"] + 0.1}) != base


def test_locate_frame_checkpoints_progressively(smoke_data):
    df = _frame(smoke_data)
    sizes = []
    det = FakeDetector()
    locate_frame(df, lambda: det, {}, {}, ["train"], 0.002, 0.8, "fp", save_every=2,
                 on_checkpoint=lambda recs: sizes.append(len(recs)))
    assert len(df) >= 5 and len(sizes) >= 2
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]
    assert sizes[0] == 2


def test_run_saves_partial_cache_on_failure(tmp_path, smoke_data):
    cfg_path = _run_cfg(tmp_path, smoke_data)
    write_boxes([{"image_path": "other.jpg", "box": (0.1, 0.1, 0.5, 0.5), "score": 0.5,
                  "source": "auto", "detector": "old"}], tmp_path / "boxes.csv")

    class Boom:
        name = "boom"

        def __init__(self):
            self.calls = 0

        def detect(self, image):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("OOM")
            return []

    with pytest.raises(RuntimeError, match="OOM"):
        run(cfg_path, detector=Boom())
    boxes = read_boxes(tmp_path / "boxes.csv")
    assert "other.jpg" in boxes
    assert len(boxes) - 1 >= 2


def _report_frame(types):
    rows = [{"image_path": f"{t}{i}.jpg", "split": "test", "fake_type": t, "postprocess": "original"}
            for t, n in types for i in range(n)]
    return pd.DataFrame(rows)


def _boxes(df, source_of):
    return {r.image_path: {"image_path": r.image_path, "box": FULL_BOX, "score": 0.0,
                           "source": source_of(r), "detector": "x"} for r in df.itertuples()}


def test_locate_report_warns_on_rate_gaps():
    df = _report_frame([("real", 4), ("A", 4)])
    boxes = _boxes(df, lambda r: "fallback" if r.fake_type == "A" and r.image_path != "A0.jpg" else "auto")
    table, warns = locate_report(df, boxes)
    assert len(warns) == 1 and "A" in warns[0] and "fallback" in warns[0]
    assert {"split", "fake_type", "postprocess", "n",
                                  "auto_rate", "manual_rate", "fallback_rate"} <= set(table.columns)
    a = table[table.fake_type == "A"].iloc[0]
    assert a.n == 4 and a.fallback_rate == 0.75 and a.auto_rate == 0.25
    _, warns = locate_report(df, _boxes(df, lambda r: "auto"))
    assert warns == []
    _, warns = locate_report(df, _boxes(df, lambda r: "manual" if r.fake_type == "A" and r.image_path in ("A0.jpg", "A1.jpg") else "auto"))
    assert len(warns) == 1 and "manual" in warns[0]


def _run_cfg(tmp_path, smoke_data, manual=None):
    cfg = load_config(CONFIG)
    cfg["data"].update(metadata=str(smoke_data), root=str(smoke_data.parent))
    cfg["locate"].update(boxes_csv=str(tmp_path / "boxes.csv"),
                         manual_csv=str(manual or tmp_path / "없음.csv"))
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return cfg_path


def test_run_writes_cache_and_report(tmp_path, smoke_data):
    cfg_path = _run_cfg(tmp_path, smoke_data)
    write_boxes([{"image_path": "other.jpg", "box": (0.1, 0.1, 0.5, 0.5), "score": 0.5,
                  "source": "auto", "detector": "old"}], tmp_path / "boxes.csv")
    df = _frame(smoke_data)
    table = run(cfg_path, detector=StemDetector())
    boxes = read_boxes(tmp_path / "boxes.csv")
    assert len(boxes) == len(df) + 1 and "other.jpg" in boxes
    assert (tmp_path / f"locate_report_{smoke_data.stem}.csv").is_file()
    assert isinstance(table, pd.DataFrame) and len(table) > 0


def test_run_warns_on_unmatched_manual_stems(tmp_path, smoke_data, capsys):
    df = _rows(_frame(smoke_data))
    stem = df.stem.iloc[0]
    man = tmp_path / "manual.csv"
    man.write_text("stem,postprocess,x0,y0,x1,y1\n"
                   f"{stem},,0.1,0.1,0.3,0.3\nnope1,,0.1,0.1,0.3,0.3\n"
                   f"{stem},recapture,0.1,0.1,0.3,0.3\nnope2,kakao,0.1,0.1,0.3,0.3\n",
                   encoding="utf-8")
    run(_run_cfg(tmp_path, smoke_data, manual=man), detector=StemDetector())
    out = capsys.readouterr().out
    assert "수동 박스 3개가 CSV의 어떤 이미지와도 맞지 않습니다" in out
    assert "nope1" in out and "nope2" in out and f"({stem}" not in out.split("예:")[1]


def test_locate_report_quality_columns_and_warnings():
    df = _report_frame([("real", 4), ("A", 4)])
    boxes = {}
    for r in df.itertuples():
        box = (0.4, 0.4, 0.5, 0.5) if r.fake_type == "real" else (0.1, 0.1, 0.9, 0.9)
        boxes[r.image_path] = {"image_path": r.image_path, "box": box, "score": 0.8,
                               "source": "auto", "detector": "x"}
    sizes = {p: (1000, 500) for p in df.image_path}
    table, warns = locate_report(df, boxes, image_sizes=sizes)
    real = table[table.fake_type == "real"].iloc[0]
    a = table[table.fake_type == "A"].iloc[0]
    assert real.median_area_frac == pytest.approx(0.01)
    assert a.median_area_frac == pytest.approx(0.64)
    assert real.median_crop_short_px == pytest.approx(50.0)   # min(100, 50)
    assert a.median_crop_short_px == pytest.approx(400.0)
    assert real.auto_mean_score == pytest.approx(0.8)
    assert any("median_area_frac" in w and "A" in w for w in warns)
    assert any("median_crop_short_px" in w for w in warns)
    assert not any("auto_mean_score" in w for w in warns)
    # 크기를 안 주면 crop 짧은 변은 NaN이고 그 경고는 없다
    table, warns = locate_report(df, boxes)
    assert table.median_crop_short_px.isna().all()
    assert not any("median_crop_short_px" in w for w in warns)
    # 점수 차이 경고
    boxes2 = {p: {**b, "score": 0.3 if p.startswith("A") else 0.9} for p, b in boxes.items()}
    _, warns = locate_report(df, boxes2, threshold=0.10)
    assert any("auto_mean_score" in w for w in warns)


def test_run_dump_crops(tmp_path, smoke_data):
    from PIL import Image
    cfg_path = _run_cfg(tmp_path, smoke_data)
    run(cfg_path, detector=StemDetector(), dump_crops=1)
    root = tmp_path / "crop_samples"
    df = _frame(smoke_data)
    groups = df.groupby(["split", "fake_type", "postprocess"]).size().index
    for split, ft, pp in groups:
        folder = root / f"{split}_{ft}_{pp}"
        jpgs = list(folder.glob("*.jpg"))
        assert len(jpgs) == 1, folder
        with Image.open(jpgs[0]) as im:
            assert max(im.size) <= 512


@pytest.mark.skipif(os.environ.get("MODEL2_RUN_DETECTOR") != "1", reason="모델 다운로드 필요")
def test_hf_detector_real(smoke_data):
    from PIL import Image
    cfg = load_config(CONFIG)["locate"]
    df = _frame(smoke_data)
    path = df[df.postprocess == "original"]._resolved_path.iloc[0]
    det = HFZeroShotDetector(cfg["model_id"], cfg["queries"], cfg["box_threshold"], cfg["text_threshold"])
    with Image.open(path) as im:
        out = det.detect(im)
    assert isinstance(out, list)
    for box, _ in out:
        validate_box(box)
