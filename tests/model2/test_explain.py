"""Grad-CAM 설명 스크립트 테스트."""
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.models.model2_memo.boxes import read_boxes, write_boxes
from src.models.model2_memo.explain import crop_cam, explain, gradcam
from src.models.model2_memo.model import build_model
from tests.model2.helpers import make_cfg


def test_gradcam_shape_range_and_model_state():
    model = build_model({"name": "efficientnet_b0", "pretrained": False, "drop_rate": 0.2})
    model.train()
    cam, prob = gradcam(model, torch.randn(1, 3, 64, 64))
    assert cam.shape == (64, 64) and cam.dtype == np.float32
    assert np.isfinite(cam).all() and cam.min() >= 0 and cam.max() <= 1
    assert 0.0 <= prob <= 1.0
    assert model.training is True
    assert all(m.training for m in model.modules())
    assert all(p.grad is None for p in model.parameters())


def test_crop_cam_removes_padding():
    cam = np.zeros((64, 64), dtype=np.float32)
    cam[16:48, :] = 1.0  # 200x100 크롭의 내용 영역
    out = crop_cam(cam, (200, 100), 64)
    assert out.shape == (100, 200)
    assert out.mean() > 0.95


def test_explain_writes_panels_and_index(tmp_path, smoke_data, trained_model):
    cfg, out_dir = trained_model
    ckpt = out_dir / "checkpoints" / "best_model.pth"
    out = tmp_path / "cam"
    result = explain(ckpt, cfg, per_group=1, output_dir=out)
    assert result == out
    index = pd.read_csv(out / "gradcam_index.csv")
    df = pd.read_csv(smoke_data, dtype=str, keep_default_na=False)
    test = df[df.split == "test"]
    from src.models.model2_memo.dataset import infer_postprocess
    pp = test["postprocess"] if "postprocess" in test else test.image_path.map(infer_postprocess)
    expected = set(zip(test.fake_type, pp))
    assert len(index) == len(expected)
    assert set(zip(index.fake_type, index.postprocess)) == expected
    assert list(index.columns) == ["image_path", "fake_type", "postprocess", "box_source",
                                   "fake_probability", "prediction", "png"]
    for png in index.png:
        with Image.open(out / png) as im:
            im.load()
    assert index.fake_probability.between(0, 1).all()
    assert (out / "README.txt").is_file()


def test_explain_missing_box_guided_error(tmp_path, smoke_data, trained_model):
    cfg, out_dir = trained_model
    ckpt = out_dir / "checkpoints" / "best_model.pth"
    boxes_csv = smoke_data.parent / "memo_boxes.csv"
    boxes = read_boxes(boxes_csv)
    df = pd.read_csv(smoke_data, dtype=str, keep_default_na=False)
    drop = df[df.split == "test"].image_path.iloc[0]
    new_csv = tmp_path / "boxes.csv"
    write_boxes([r for k, r in boxes.items() if k != drop], new_csv)
    import yaml
    c = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    c["locate"]["boxes_csv"] = str(new_csv)
    cfg2 = tmp_path / "cfg2.yaml"
    cfg2.write_text(yaml.safe_dump(c, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="locate"):
        explain(ckpt, cfg2, per_group=100, output_dir=tmp_path / "cam")
