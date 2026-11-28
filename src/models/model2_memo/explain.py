"""모델 2 Grad-CAM: 모델이 메모 크롭의 어디에 반응하는지 PNG 패널로 저장한다."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.nn import functional as F

from .boxes import read_boxes
from .dataset import build_transform, crop_region, read_metadata, select_rows
from .model import build_model
from .utils import load_config, select_device

INDEX_COLUMNS = ["image_path", "fake_type", "postprocess", "box_source",
                 "fake_probability", "prediction", "png"]
DISCLAIMER = (
    "Grad-CAM 설명 안내\n"
    "- 강조된 곳은 이 모델의 '가짜 점수'가 반응한 위치이며, 실제로 조작된 영역이라는 증거가 아닙니다.\n"
    "- 색의 세기는 이미지마다 최댓값 기준으로 정규화한 상대값입니다. 이미지 사이에 비교할 수 없고 "
    "조작 확률도 아닙니다.\n"
    "- 예측이 진짜여도 지도는 가짜 점수 기준으로 계산됩니다.\n"
    "- 종이 가장자리, 여백, 배경에 열이 몰리면 글씨 흔적이 아닌 지름길(shortcut)을 의심하세요.\n")


def gradcam(model: torch.nn.Module, tensor: torch.Tensor) -> tuple[np.ndarray, float]:
    """가짜 로짓 하나에 대한 Grad-CAM. (S×S 지도 [0,1], 가짜 확률)을 돌려준다."""
    layer = getattr(getattr(model, "backbone", None), "conv_head", None)
    if not isinstance(layer, torch.nn.Conv2d):
        raise ValueError("model.backbone.conv_head(Conv2d)가 없어 Grad-CAM을 계산할 수 없습니다")
    size = tensor.shape[-1]
    states = [(m, m.training) for m in model.modules()]
    captured: list[torch.Tensor] = []
    handle = layer.register_forward_hook(lambda _m, _i, out: captured.append(out))
    try:
        model.eval()
        with torch.inference_mode(False), torch.enable_grad():
            x = tensor.detach().clone().float().requires_grad_(True)
            with torch.autocast(device_type=x.device.type, enabled=False):
                logit = model(x)
                act = captured[0]
                grad, = torch.autograd.grad(logit[0], act)
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1, keepdim=True).relu()
            cam = F.interpolate(cam.detach(), size=(size, size), mode="bilinear",
                                align_corners=False)[0, 0].float().cpu().numpy()
            prob = float(torch.sigmoid(logit[0].detach()))
    finally:
        handle.remove()
        for m, was_training in states:
            m.training = was_training
    peak = float(cam.max())
    cam = cam / peak if peak > 0 else np.zeros_like(cam)
    return cam.astype(np.float32), prob


def crop_cam(cam: np.ndarray, crop_size: tuple[int, int], input_size: int) -> np.ndarray:
    """ImageOps.pad가 넣은 여백을 잘라내고 크롭 크기 (W,H)로 되돌린다. [H,W], [0,1]."""
    w, h = crop_size
    if w > h:
        rw, rh = input_size, round(h / w * input_size)
    elif h > w:
        rw, rh = round(w / h * input_size), input_size
    else:
        rw = rh = input_size
    left = round((input_size - rw) * 0.5)
    top = round((input_size - rh) * 0.5)
    inner = np.asarray(cam, dtype=np.float32)[top:top + max(rh, 1), left:left + max(rw, 1)]
    img = Image.fromarray(inner, mode="F").resize((w, h), Image.BILINEAR)
    return np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)


_STOPS = np.array([[0, 0, 255], [0, 255, 255], [255, 255, 0], [255, 0, 0]], dtype=np.float32)


def overlay(image: Image.Image, heat: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """파랑→하늘→노랑→빨강 색지도를 RGB 이미지 위에 섞는다."""
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    t = np.clip(np.asarray(heat, dtype=np.float32), 0, 1) * (len(_STOPS) - 1)
    lo = np.minimum(t.astype(int), len(_STOPS) - 2)
    frac = (t - lo)[..., None]
    color = _STOPS[lo] * (1 - frac) + _STOPS[lo + 1] * frac
    out = base * (1 - alpha) + color * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _resize_to_height(img: Image.Image, h: int) -> Image.Image:
    return img.resize((max(1, round(img.width * h / img.height)), h), Image.BICUBIC)


def _panel(image: Image.Image, box, crop: Image.Image, heat: np.ndarray, header: str) -> Image.Image:
    """[전체+박스 | 크롭 | 크롭+CAM] 패널 위에 흰 머리글을 붙인다."""
    full = ImageOps.exif_transpose(image).convert("RGB")
    scale = min(1.0, 512 / max(full.size))
    full = full.resize((max(1, round(full.width * scale)), max(1, round(full.height * scale))))
    x0, y0, x1, y1 = box
    ImageDraw.Draw(full).rectangle([x0 * full.width, y0 * full.height,
                                    x1 * full.width - 1, y1 * full.height - 1],
                                   outline=(255, 0, 0), width=3)
    big = _resize_to_height(crop, 256)
    heat_big = np.asarray(Image.fromarray(heat.astype(np.float32), mode="F")
                          .resize(big.size, Image.BILINEAR), dtype=np.float32)
    parts = [full, big, overlay(big, heat_big)]
    head_h = 28
    height = head_h + max(p.height for p in parts)
    canvas = Image.new("RGB", (sum(p.width for p in parts) + 8 * (len(parts) - 1), height),
                       "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((6, 6), header, fill="black", font=font)
    x = 0
    for p in parts:
        canvas.paste(p, (x, head_h))
        x += p.width + 8
    return canvas


def explain(checkpoint_path, config_path: str | None = None, metadata: str | None = None,
            split: str = "test", per_group: int = 4, output_dir: str | None = None,
            device_name: str | None = None) -> Path:
    """그룹(fake_type, postprocess)마다 앞 per_group장의 Grad-CAM 패널을 저장한다."""
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
    rows = select_rows(df, split)
    rows = rows.groupby(["fake_type", "postprocess"], sort=False).head(max(per_group, 0))
    absent = [p for p in rows.image_path if p not in boxes]
    if absent:
        raise ValueError(
            f"박스가 없는 이미지 {len(absent)}개 (앞 3개: {absent[:3]}). "
            "python -m src.models.model2_memo.locate --metadata <CSV> 를 먼저 실행하세요")

    model = build_model(cfg["model"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    transform = build_transform(cfg, train=False)
    size = int(cfg["model"]["input_size"])
    out = Path(output_dir) if output_dir else Path(tc["output_dir"]) / "gradcam"
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for row in rows.to_dict("records"):
        rec = boxes[row["image_path"]]
        with Image.open(row["_resolved_path"]) as im:
            im.load()
            tensor = transform(im, rec["box"]).unsqueeze(0).to(device)
            crop = crop_region(im, rec["box"], cfg["crop"]["margin"])
            cam, prob = gradcam(model, tensor)
            pred = int(prob >= threshold)
            header = (f"{row['fake_type']} | {row['postprocess']} | P(fake)={prob:.2f} -> "
                      f"{'fake' if pred else 'real'} (box: {rec['source']})")
            panel = _panel(im, rec["box"], crop, crop_cam(cam, crop.size, size), header)
        rel = Path(f"{row['fake_type']}_{row['postprocess']}") / f"{Path(row['image_path']).stem}.png"
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        panel.save(out / rel)
        records.append({"image_path": row["image_path"], "fake_type": row["fake_type"],
                        "postprocess": row["postprocess"], "box_source": rec["source"],
                        "fake_probability": prob, "prediction": pred, "png": rel.as_posix()})
    pd.DataFrame(records, columns=INDEX_COLUMNS).to_csv(out / "gradcam_index.csv", index=False)
    (out / "README.txt").write_text(DISCLAIMER, encoding="utf-8")
    print(f"완료: Grad-CAM 패널 {len(records)}장 -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="모델 2 Grad-CAM 설명 패널 저장")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--per-group", type=int, default=4)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()
    try:
        explain(args.checkpoint, args.config, args.metadata, args.split, args.per_group,
                args.output_dir, args.device)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
