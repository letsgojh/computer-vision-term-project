"""모델 2 평가: 체크포인트에 고정된 임계값으로 test 전체를 예측하고 지표를 낸다."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .boxes import read_boxes
from .dataset import MemoCropDataset, build_transform, read_metadata, select_rows
from .metrics import binary_metrics, rate_table
from .model import build_model
from .utils import load_config, save_json, select_device

PREDICTION_COLUMNS = ["image_path", "label", "fake_type", "fake_probability", "prediction",
                      "source_id", "domain", "postprocess", "box_source"]


def _records(table: pd.DataFrame) -> list[dict]:
    """DataFrame을 JSON 저장 가능한 레코드 목록으로 바꾼다."""
    return [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
            for r in table.to_dict("records")]


def evaluate(checkpoint_path, config_path: str | None = None, metadata: str | None = None,
             output_dir: str | None = None, device_name: str | None = None) -> dict:
    """test split 전체를 평가하고 예측 CSV와 지표 파일을 쓴다. 지표 dict를 돌려준다."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"체크포인트가 없습니다: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    src = load_config(config_path) if config_path else cfg
    dc, tc = src["data"], dict(src["training"])
    if device_name:
        tc["device"] = device_name
    if metadata:
        dc = {**dc, "metadata": str(metadata)}
    device = select_device(tc["device"])
    threshold = float(ckpt["threshold"])

    df = read_metadata(dc["metadata"], dc["root"], dc.get("leakage_columns", ("item_id",)),
                       dc.get("verify_images", True))
    boxes = read_boxes(src["locate"]["boxes_csv"])
    test_df = select_rows(df, "test")
    n_manual = sum(boxes[p]["source"] == "manual" for p in test_df["image_path"] if p in boxes)
    if n_manual:
        print(f"경고: test 행 {n_manual}개가 수동 박스를 씁니다 (평가 박스와 달라질 수 있음).")

    model = build_model(cfg["model"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    loader = DataLoader(MemoCropDataset(test_df, boxes, build_transform(cfg, train=False)),
                        batch_size=tc["batch_size"], shuffle=False,
                        num_workers=tc["num_workers"])
    rows = []
    with torch.inference_mode():
        for b in loader:
            probs = torch.sigmoid(model(b["image"].to(device))).cpu().numpy()
            for i, p in enumerate(probs):
                rows.append({
                    "image_path": b["image_path"][i], "label": int(b["label"][i]),
                    "fake_type": b["fake_type"][i], "fake_probability": float(p),
                    "prediction": int(p >= threshold), "source_id": b["source_id"][i],
                    "domain": b["domain"][i], "postprocess": b["postprocess"][i],
                    "box_source": b["box_source"][i]})
    pred = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)

    out = Path(output_dir) if output_dir else Path(tc["output_dir"]) / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "test_predictions.csv", index=False)
    rates = rate_table(pred, ["fake_type", "postprocess"])
    rates.to_csv(out / "test_rates.csv", index=False)

    y, p = pred["label"].to_numpy(), pred["fake_probability"].to_numpy()
    report = {
        "checkpoint": str(checkpoint_path), "epoch": int(ckpt["epoch"]), "threshold": threshold,
        "overall": binary_metrics(y, p, threshold),
        "by_postprocess": {
            pp: binary_metrics(g["label"], g["fake_probability"], threshold)
            for pp, g in pred.groupby("postprocess")},
        "by_fake_type": _records(rate_table(pred, ["fake_type"])),
        "by_fake_type_postprocess": _records(rates),
    }
    save_json(out / "test_metrics.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="모델 2 평가 (test 전체, 고정 임계값)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()
    try:
        report = evaluate(args.checkpoint, args.config, args.metadata, args.output_dir,
                          args.device)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    o = report["overall"]
    print(f"완료: n={o['n']} f1={o['f1']:.4f} acc={o['accuracy']:.4f} "
          f"threshold={report['threshold']:.4f}")


if __name__ == "__main__":
    main()
