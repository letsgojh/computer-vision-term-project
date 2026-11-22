"""Whole-photo scoring and Grad-CAM for native_crop checkpoints.

A native_crop model sees one fixed-size patch. For a user photo we resize the short side
to work_short (inside the training size range), tile it with overlapping crops that cover
the whole image, average the crop probabilities, and stitch per-crop Grad-CAM maps into one
heatmap over the full photo. Letterbox checkpoints fall back to a single full-image pass.
"""

import math

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageOps
from torch.nn import functional as F

from src.models.model1_full_image.dataset import NativeCropTransform, build_transform
from src.models.model1_full_image.explain import DISCLAIMER, align_cam, generate_gradcam
from src.models.model1_full_image.model import build_model
from src.models.model1_full_image.utils import load_checkpoint, select_device

WORK_SHORT = 512


def load_detector(checkpoint_path, device_name="auto"):
    """Return (model, checkpoint, device) ready for inference."""
    checkpoint = load_checkpoint(checkpoint_path)
    device = select_device(device_name)
    model = build_model(checkpoint["config"]["model"], pretrained=False).to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, checkpoint, device


def crop_grid(width, height, size):
    """Top-left corners of overlapping crops covering the whole image."""
    def positions(length):
        if length <= size:
            return [0]
        count = math.ceil(length / size) + 1  # +1 so crops overlap; abutting crops leave visible heatmap seams
        return [round(i * (length - size) / (count - 1)) for i in range(count)]
    return [(x, y) for y in positions(height) for x in positions(width)]


def overlay(image, heatmap, alpha=0.45):
    """Blend a [H,W] 0..1 heatmap over an RGB PIL image."""
    colours = (colormaps["jet"](heatmap)[..., :3] * 255).astype(np.uint8)
    return Image.blend(image, Image.fromarray(colours).resize(image.size), alpha)


def analyze(model, checkpoint, device, image, with_heatmap=True, work_short=WORK_SHORT):
    """Score one PIL image; returns probability, per-crop scores and an optional heatmap overlay."""
    model_config = checkpoint["config"]["model"]
    image = ImageOps.exif_transpose(image).convert("RGB")
    if model_config.get("preprocess", "letterbox") != "native_crop":
        tensor = build_transform(model_config)(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            probability = float(model(tensor).float().sigmoid().item())
        result = {"fake_probability": probability, "crop_probabilities": [probability], "work_size": image.size}
        if with_heatmap:
            cam = generate_gradcam(model, tensor)["cam"]
            heatmap, _, _ = align_cam(cam, image.size, model_config["input_size"])
            result["heatmap"] = overlay(image, np.asarray(heatmap))
        return result
    size = int(model_config["input_size"])
    work = NativeCropTransform.resize_short(image, max(size, work_short))
    corners = crop_grid(*work.size, size)
    to_tensor = NativeCropTransform(size, model_config["mean"], model_config["std"]).tensor
    batch = torch.stack([to_tensor(work.crop((x, y, x + size, y + size))) for x, y in corners]).to(device)
    with torch.inference_mode():
        crop_probabilities = model(batch).float().sigmoid().cpu().tolist()
    result = {"fake_probability": float(np.mean(crop_probabilities)), "crop_probabilities": crop_probabilities,
              "work_size": work.size, "crops": len(corners)}
    if with_heatmap:
        canvas = torch.zeros(work.size[1], work.size[0])
        counts = torch.zeros_like(canvas)
        # Taper each crop toward its edges so overlapping crops blend without grid seams.
        window = torch.hann_window(size + 2, periodic=False)[1:-1]
        window = torch.outer(window, window)
        for (x, y), tensor in zip(corners, batch):
            cam = generate_gradcam(model, tensor.unsqueeze(0))["cam"]
            cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)[0, 0]
            canvas[y:y + size, x:x + size] += cam * window
            counts[y:y + size, x:x + size] += window
        heatmap = (canvas / counts.clamp(min=1e-6)).numpy()
        peak = heatmap.max()
        heatmap = heatmap / peak if peak > 0 else heatmap
        result["heatmap"] = overlay(work, heatmap)
    return result


__all__ = ["DISCLAIMER", "analyze", "crop_grid", "load_detector", "overlay"]
