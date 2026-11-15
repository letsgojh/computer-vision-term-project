"""모델 2 학습: val F1 기준 체크포인트, val에서 고른 임계값을 체크포인트에 고정."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .boxes import read_boxes
from .dataset import MemoCropDataset, build_transform, read_metadata, select_rows
from .metrics import binary_metrics, choose_threshold
from .model import build_model
from .utils import load_config, seed_everything, select_device

LOG_FIELDS = ["epoch", "lr", "train_loss", "val_loss", "val_f1", "val_auc", "threshold"]


def _summary(train_df, val_df, boxes):
    """split, fake_type, postprocess, box_source별 개수 표."""
    import pandas as pd
    parts = []
    for name, df in (("train", train_df), ("val", val_df)):
        d = df[["fake_type", "postprocess"]].copy()
        d["box_source"] = [boxes[p]["source"] for p in df["image_path"]]
        d.insert(0, "split", name)
        parts.append(d)
    return (pd.concat(parts).groupby(["split", "fake_type", "postprocess", "box_source"])
            .size().reset_index(name="n"))


@torch.no_grad()
def _evaluate(model, loader, device, criterion):
    """val를 float32로 돌려 (손실, 라벨, 확률)을 돌려준다."""
    model.eval()
    losses, labels, probs = [], [], []
    for b in loader:
        x, y = b["image"].to(device), b["label"].to(device)
        logit = model(x)
        losses.append(criterion(logit, y).item() * len(y))
        labels.append(y.cpu().numpy())
        probs.append(torch.sigmoid(logit).cpu().numpy())
    labels, probs = np.concatenate(labels), np.concatenate(probs)
    return sum(losses) / len(labels), labels, probs


def train(config_path: str | Path, device_name: str | None = None) -> Path:
    """학습을 돌리고 출력 폴더를 돌려준다."""
    cfg = load_config(config_path)
    if device_name:
        cfg["training"]["device"] = device_name
    tc, dc = cfg["training"], cfg["data"]
    seed_everything(cfg["seed"])
    device = select_device(tc["device"])

    df = read_metadata(dc["metadata"], dc["root"], dc.get("leakage_columns", ("item_id",)),
                       dc.get("verify_images", True))
    boxes = read_boxes(cfg["locate"]["boxes_csv"])
    train_df = select_rows(df, "train", dc["exclude_from_training"], dc["train_postprocess"])
    val_df = select_rows(df, "val", dc["exclude_from_training"], dc["train_postprocess"])
    if val_df["label"].nunique() < 2:
        raise ValueError("val needs both real and fake rows after filtering "
                         "(exclude_from_training / train_postprocess).")

    out = Path(tc["output_dir"])
    ckpt_dir = out / "checkpoints"
    for n in ("best", "last"):
        if (ckpt_dir / f"{n}_model.pth").exists():
            raise FileExistsError(f"이미 체크포인트가 있습니다: {ckpt_dir / (n + '_model.pth')}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    _summary(train_df, val_df, boxes).to_csv(out / "data_summary.csv", index=False)
    n_manual = sum(boxes[p]["source"] == "manual" for p in val_df["image_path"])
    if n_manual:
        print(f"경고: val 행 {n_manual}개가 수동 박스를 씁니다 (평가 박스와 달라질 수 있음).")

    bs = tc["batch_size"]
    gen = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(
        MemoCropDataset(train_df, boxes, build_transform(cfg, train=True)),
        batch_size=bs, shuffle=True, generator=gen, num_workers=tc["num_workers"],
        drop_last=len(train_df) % bs == 1)
    val_loader = DataLoader(
        MemoCropDataset(val_df, boxes, build_transform(cfg, train=False)),
        batch_size=bs, shuffle=False, num_workers=tc["num_workers"])

    model = build_model(cfg["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"],
                            weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
    criterion = nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda" and bool(tc.get("amp", False))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, bad = -1.0, 0
    with (out / "logs/training.csv").open("w", newline="", encoding="utf-8") as f:
        log = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        log.writeheader()
        for epoch in range(1, tc["epochs"] + 1):
            model.train()
            total, count = 0.0, 0
            for b in train_loader:
                x, y = b["image"].to(device), b["label"].to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total += loss.item() * len(y)
                count += len(y)
            lr = opt.param_groups[0]["lr"]
            sched.step()

            val_loss, labels, probs = _evaluate(model, val_loader, device, criterion)
            thr = choose_threshold(labels, probs)
            vm = binary_metrics(labels, probs, thr)
            log.writerow({"epoch": epoch, "lr": lr, "train_loss": total / count,
                          "val_loss": val_loss, "val_f1": vm["f1"],
                          "val_auc": vm["roc_auc"], "threshold": thr})
            f.flush()
            print(f"epoch {epoch}: train_loss={total / count:.4f} val_f1={vm['f1']:.4f}")

            improved = vm["f1"] > best_f1
            best_f1 = max(best_f1, vm["f1"])
            ckpt = {"model_state": model.state_dict(), "config": cfg, "epoch": epoch,
                    "threshold": thr, "val_metrics": vm, "best_f1": best_f1}
            torch.save(ckpt, ckpt_dir / "last_model.pth")
            if improved:
                torch.save(ckpt, ckpt_dir / "best_model.pth")
                bad = 0
            else:
                bad += 1
                if bad >= tc["patience"]:
                    break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="모델 2 학습")
    parser.add_argument("--config", default="configs/model2_memo.yaml")
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()
    try:
        out = train(args.config, args.device)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"완료: {out}")


if __name__ == "__main__":
    main()
