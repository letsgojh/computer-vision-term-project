import pytest
import torch

from src.models.model2_memo.utils import load_config, save_json, seed_everything, select_device


def test_select_device_cpu_and_auto():
    assert select_device("cpu").type == "cpu"
    assert select_device("auto").type in {"cuda", "mps", "cpu"}


def test_select_device_unavailable_cuda_raises():
    if torch.cuda.is_available():
        pytest.skip("CUDA 있음")
    with pytest.raises(RuntimeError):
        select_device("cuda")


def test_load_config_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_load_config_and_save_json(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\n", encoding="utf-8")
    assert load_config(p) == {"a": 1}
    save_json(tmp_path / "d" / "o.json", {"k": "한글"})
    assert "한글" in (tmp_path / "d" / "o.json").read_text(encoding="utf-8")


def test_seed_everything_reproducible():
    seed_everything(3)
    a = torch.rand(2)
    seed_everything(3)
    assert torch.equal(a, torch.rand(2))
