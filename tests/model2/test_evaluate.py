import json

import pandas as pd
import pytest

from src.models.model2_memo.evaluate import PREDICTION_COLUMNS, evaluate


def test_evaluate_writes_predictions(tmp_path, smoke_data, trained_model):
    cfg, out = trained_model
    report = evaluate(out / "checkpoints/best_model.pth", cfg, output_dir=tmp_path / "pred")
    df = pd.read_csv(tmp_path / "pred/test_predictions.csv", keep_default_na=False)
    src = pd.read_csv(smoke_data, keep_default_na=False)
    assert list(df.columns) == PREDICTION_COLUMNS
    assert len(df) == (src.split == "test").sum()
    assert "B" in set(df.fake_type) and set(df.postprocess) == {"original", "kakao", "resize"}
    assert df.fake_probability.between(0, 1).all()
    assert (df.prediction == (df.fake_probability >= report["threshold"]).astype(int)).all()
    assert (tmp_path / "pred/test_metrics.json").is_file()
    assert (tmp_path / "pred/test_rates.csv").is_file()
    saved = json.loads((tmp_path / "pred/test_metrics.json").read_text(encoding="utf-8"))
    assert set(saved["by_postprocess"]) == {"original", "kakao", "resize"}
    assert saved["overall"]["n"] == len(df)


def test_evaluate_output_dir_does_not_touch_default(tmp_path, trained_model):
    cfg, out = trained_model
    evaluate(out / "checkpoints/best_model.pth", cfg, output_dir=tmp_path / "p2")
    assert not (out / "predictions/test_predictions.csv").exists()


def test_evaluate_metadata_override(tmp_path, smoke_data, trained_model):
    cfg, out = trained_model
    src = pd.read_csv(smoke_data, keep_default_na=False)
    sub = src[(src.split == "test") & (~src.image_path.str.contains("kakao|resize"))]
    csv = tmp_path / "sub.csv"
    sub.to_csv(csv, index=False)
    evaluate(out / "checkpoints/best_model.pth", cfg, metadata=str(csv), output_dir=tmp_path / "p3")
    df = pd.read_csv(tmp_path / "p3/test_predictions.csv", keep_default_na=False)
    assert len(df) == len(sub) and set(df.postprocess) == {"original"}


def test_evaluate_refuses_overlap_with_training(tmp_path, smoke_data, trained_model):
    cfg, out = trained_model
    src = pd.read_csv(smoke_data, keep_default_na=False)
    tr = src[(src.split == "train") & (src.fake_type == "real") & (src.postprocess == "original")]
    sid = tr.source_id.iloc[0]
    src = src[~((src.split == "train") & (src.source_id == sid))]
    t = src.index[src.split == "test"][0]
    src.loc[t, "source_id"] = sid
    csv = tmp_path / "overlap.csv"
    src.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="leak"):
        evaluate(out / "checkpoints/best_model.pth", cfg, metadata=str(csv),
                 output_dir=tmp_path / "p4")
