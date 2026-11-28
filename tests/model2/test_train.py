import pandas as pd
import pytest
import torch

from src.models.model2_memo.model import build_model
from src.models.model2_memo.train import train
from tests.model2.helpers import make_cfg


def test_build_model_output_shape():
    m = build_model({"name": "efficientnet_b0", "pretrained": False, "drop_rate": 0.2})
    assert m(torch.zeros(2, 3, 64, 64)).shape == (2,)


def test_train_end_to_end(tmp_path, smoke_data):
    out = train(make_cfg(tmp_path, smoke_data))
    assert (out / "checkpoints/best_model.pth").is_file()
    assert (out / "checkpoints/last_model.pth").is_file()
    assert (out / "config.yaml").is_file()
    assert 1 <= len(pd.read_csv(out / "logs/training.csv")) <= 2
    ckpt = torch.load(out / "checkpoints/best_model.pth", weights_only=False)
    assert 0.0 <= ckpt["threshold"] <= 1.0 and ckpt["config"]["model"]["input_size"] == 64
    assert {"model_state", "epoch", "val_metrics", "best_f1", "protected_ids"} <= set(ckpt)
    assert ckpt["protected_ids"]["source_id"] == sorted(set(ckpt["protected_ids"]["source_id"]))
    summary = pd.read_csv(out / "data_summary.csv", keep_default_na=False)
    assert "B" not in set(summary.fake_type) and set(summary.postprocess) == {"original"}
    assert set(summary.split) == {"train", "val"}


def test_train_refuses_existing_output(tmp_path, smoke_data):
    cfg = make_cfg(tmp_path, smoke_data, epochs=1)
    train(cfg)
    with pytest.raises(FileExistsError):
        train(cfg)


def test_train_requires_both_classes_in_val(tmp_path, smoke_data):
    df = pd.read_csv(smoke_data, keep_default_na=False)
    df = df[~((df.split == "val") & (df.label == 0))]
    bad = tmp_path / "val_fake_only.csv"
    df.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="both"):
        train(make_cfg(tmp_path, smoke_data, metadata=bad))


def test_train_missing_boxes_guided_error(tmp_path, smoke_data):
    import yaml
    cfg_path = make_cfg(tmp_path, smoke_data)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["locate"]["boxes_csv"] = str(tmp_path / "없음.csv")
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="locate"):
        train(cfg_path)
    assert not (tmp_path / "out" / "checkpoints").exists()
