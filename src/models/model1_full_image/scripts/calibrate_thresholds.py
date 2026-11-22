"""Score the validation split and write thresholds.json for the three-level verdict.

Validation contains only training generators; record that the AI recall on hold-out
generators can be lower (see analyze_generalization.py).
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.model1_full_image.dataset import ImageDataset, build_transform, read_metadata
from src.models.model1_full_image.model import build_model
from src.models.model1_full_image.thresholds import calibrate, save_thresholds
from src.models.model1_full_image.utils import load_checkpoint, load_config, select_device


@torch.inference_mode()
def score_split(checkpoint, config, split, device):
    data = config["data"]
    df = read_metadata(data["metadata"], data["root"], data.get("leakage_columns", []), verify_images=False)
    dataset = ImageDataset(df, split, build_transform(checkpoint["config"]["model"]))
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], num_workers=config["training"]["num_workers"])
    model = build_model(checkpoint["config"]["model"], pretrained=False).to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    labels, scores = [], []
    for batch in loader:
        scores += model(batch["image"].to(device)).float().sigmoid().cpu().tolist()
        labels += batch["label"].int().tolist()
    return labels, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="Default: <checkpoint folder>/../thresholds.json")
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--max-real-fpr", type=float, default=0.01, help="Real FPR allowed at the AI threshold")
    parser.add_argument("--review-fpr", type=float, default=0.05, help="Real FPR allowed at the suspicious threshold")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    config = load_config(args.config)
    labels, scores = score_split(checkpoint, config, "val", select_device(args.device))
    values = calibrate(labels, scores, args.target_recall, args.max_real_fpr, args.review_fpr)
    values["checkpoint"] = str(Path(args.checkpoint).resolve())
    output = Path(args.output) if args.output else Path(args.checkpoint).resolve().parent.parent / "thresholds.json"
    save_thresholds(output, values)
    print(values)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
