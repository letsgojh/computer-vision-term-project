"""model 2 테스트 공용 도우미."""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def make_cfg(tmp_path, smoke_data, **override) -> Path:
    """smoke 데이터용 설정 yaml을 쓰고 경로를 돌려준다. metadata=는 data.metadata 덮어쓰기."""
    cfg = yaml.safe_load((REPO / "configs/model2_memo.yaml").read_text(encoding="utf-8"))
    cfg["data"]["metadata"] = str(override.pop("metadata", smoke_data))
    cfg["data"]["root"] = str(smoke_data.parent)
    cfg["locate"]["boxes_csv"] = str(smoke_data.parent / "memo_boxes.csv")
    cfg["model"].update(pretrained=False, input_size=64)
    cfg["training"].update(epochs=2, batch_size=8, num_workers=0, device="cpu",
                           output_dir=str(Path(tmp_path) / "out"))
    cfg["training"].update(override)
    path = Path(tmp_path) / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return path
