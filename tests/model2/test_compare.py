import math

import pandas as pd
import pytest

from src.models.model2_memo.compare import compare, write_comparison

COLS = ["image_path", "label", "fake_type", "fake_probability", "prediction"]
O = "data/processed/original/"


def _write(path, rows, extra=None):
    df = pd.DataFrame(rows, columns=COLS)
    for k, v in (extra or {}).items():
        df[k] = v
    df.to_csv(path, index=False)


def _basic(tmp_path):
    _write(tmp_path / "m1.csv", [(O + "r1.jpg", 0, "real", 0.1, 0), (O + "r2.jpg", 0, "real", 0.2, 0),
                                 (O + "a1.jpg", 1, "A", 0.3, 0), (O + "a2.jpg", 1, "A", 0.9, 1)])
    _write(tmp_path / "m2.csv", [(O + "r1.jpg", 0, "real", 0.1, 0), (O + "r2.jpg", 0, "real", 0.6, 1),
                                 (O + "a1.jpg", 1, "A", 0.8, 1), (O + "a2.jpg", 1, "A", 0.9, 1)])
    return compare(tmp_path / "m1.csv", tmp_path / "m2.csv")


def test_compare_rates_and_auc(tmp_path):
    table, info = _basic(tmp_path)
    assert list(table.columns) == ["postprocess", "fake_type", "n", "metric", "model1_rate",
                                   "model2_rate", "model1_auc", "model2_auc"]
    idx = table.set_index(["postprocess", "fake_type"])
    a = idx.loc[("original", "A")]
    assert a.n == 2 and a.metric == "recall"
    assert a.model1_rate == pytest.approx(0.5) and a.model2_rate == pytest.approx(1.0)
    assert a.model1_auc == pytest.approx(1.0) and a.model2_auc == pytest.approx(1.0)
    r = idx.loc[("original", "real")]
    assert r.metric == "fpr" and r.model1_rate == 0.0 and r.model2_rate == pytest.approx(0.5)
    assert math.isnan(r.model1_auc)
    assert info == {"n_joined": 4, "missing_in_model1": 0, "missing_in_model2": 0}


def test_compare_infers_postprocess_and_counts_missing(tmp_path):
    k = "data/processed/kakao/"
    _write(tmp_path / "m1.csv", [(k + "r1.jpg", 0, "real", 0.1, 0), (k + "a1.jpg", 1, "A", 0.9, 1)])
    _write(tmp_path / "m2.csv", [(k + "r1.jpg", 0, "real", 0.1, 0), (k + "a1.jpg", 1, "A", 0.9, 1),
                                 (k + "a2.jpg", 1, "A", 0.9, 1)], extra={"postprocess": "original"})
    table, info = compare(tmp_path / "m1.csv", tmp_path / "m2.csv")
    assert set(table.postprocess) == {"kakao"}
    assert info == {"n_joined": 2, "missing_in_model1": 1, "missing_in_model2": 0}


def test_compare_label_mismatch_raises(tmp_path):
    _write(tmp_path / "m1.csv", [(O + "a.jpg", 1, "A", 0.9, 1)])
    _write(tmp_path / "m2.csv", [(O + "a.jpg", 0, "real", 0.9, 1)])
    with pytest.raises(ValueError, match="a.jpg"):
        compare(tmp_path / "m1.csv", tmp_path / "m2.csv")


def test_compare_missing_column_and_duplicates_and_empty(tmp_path):
    pd.DataFrame({"image_path": ["x"], "label": [0]}).to_csv(tmp_path / "bad.csv", index=False)
    _write(tmp_path / "ok.csv", [(O + "a.jpg", 1, "A", 0.9, 1)])
    with pytest.raises(ValueError, match="fake_type"):
        compare(tmp_path / "bad.csv", tmp_path / "ok.csv")
    _write(tmp_path / "dup.csv", [(O + "a.jpg", 1, "A", 0.9, 1)] * 2)
    with pytest.raises(ValueError, match="dup.csv"):
        compare(tmp_path / "dup.csv", tmp_path / "ok.csv")
    _write(tmp_path / "other.csv", [(O + "b.jpg", 1, "A", 0.9, 1)])
    with pytest.raises(ValueError):
        compare(tmp_path / "other.csv", tmp_path / "ok.csv")


def test_compare_fake_without_real_gives_nan_auc(tmp_path):
    _write(tmp_path / "m1.csv", [(O + "a.jpg", 1, "A", 0.9, 1)])
    table, _ = compare(tmp_path / "m1.csv", tmp_path / "m1.csv")
    assert math.isnan(table.iloc[0].model1_auc)


def test_write_comparison(tmp_path):
    table, info = _basic(tmp_path)
    out = tmp_path / "out" / "cmp"
    write_comparison(table, info, out)
    assert (out / "comparison.csv").exists()
    lines = (out / "comparison.md").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "A 유형 탐지율 (original): 모델1 50.0% → 모델2 100.0%"
    assert "| original | A | 2 | recall | 50.0% | 100.0% | 1.000 | 1.000 |" in lines
    assert any("n_joined" in ln or "4" in ln for ln in lines[-3:])


def test_write_comparison_without_a_omits_summary(tmp_path):
    _write(tmp_path / "m.csv", [(O + "r.jpg", 0, "real", 0.1, 0)])
    table, info = compare(tmp_path / "m.csv", tmp_path / "m.csv")
    write_comparison(table, info, tmp_path)
    assert (tmp_path / "comparison.md").read_text(encoding="utf-8").startswith("| postprocess")
