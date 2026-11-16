import pandas as pd
import pytest

from src.models.model2_memo.metrics import binary_metrics, choose_threshold, rate_table


def test_binary_metrics_values():
    m = binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1)
    assert m["accuracy"] == m["precision"] == m["recall"] == m["f1"] == pytest.approx(0.5)
    assert m["roc_auc"] == pytest.approx(0.75)


def test_binary_metrics_single_class_auc_none():
    assert binary_metrics([1, 1], [0.2, 0.9], 0.5)["roc_auc"] is None


def test_choose_threshold_maximizes_f1():
    # 모델 1과 같은 규칙: 최대 F1 동점이면 0.5에 가까운 값, 후보에 0.5 포함
    assert choose_threshold([0, 0, 1, 1], [0.1, 0.3, 0.6, 0.8]) == pytest.approx(0.5)
    assert choose_threshold([0, 0, 1, 1], [0.2, 0.7, 0.7, 0.9]) == pytest.approx(0.5)
    assert choose_threshold([0, 0, 1, 1], [0.6, 0.7, 0.8, 0.9]) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        choose_threshold([1, 1], [0.2, 0.9])


def test_rate_table():
    pred = pd.DataFrame({"fake_type": ["A", "A", "real"], "postprocess": ["original"] * 3,
                         "prediction": [1, 0, 1], "fake_probability": [0.9, 0.3, 0.6]})
    t = rate_table(pred, ["fake_type", "postprocess"]).set_index("fake_type")
    assert t.loc["A", "n"] == 2 and t.loc["A", "pred_fake_rate"] == pytest.approx(0.5)
    assert t.loc["real", "pred_fake_rate"] == pytest.approx(1.0)
