"""v6 local-edit detector: scan a photo at native resolution and map suspicious tiles.

Partial AI edits (e.g. only a memo's date regenerated and pasted back) leave a region without
camera noise, which survives only at native resolution (DEVELOPMENT_LOG, v5/v6). The photo is
covered with 256 tiles (stride 256, last row/column aligned to the edge) and scored per tile.
Scores are logits: probabilities saturate at 1.0 for some real tiles (photographed screens), so
the image score is the max tile logit, thresholded with real-photo FPR targets (thresholds.json).
"""

import numpy as np
import torch
from PIL import Image, ImageOps

from src.models.model1_full_image.inference import overlay

TILE = 256


def positions(length):
    if length <= TILE:
        return [0]
    return sorted(set(list(range(0, length - TILE + 1, TILE)) + [length - TILE]))


def load_image(image_or_path):
    if isinstance(image_or_path, Image.Image):
        return ImageOps.exif_transpose(image_or_path).convert("RGB")
    with Image.open(image_or_path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


@torch.no_grad()
def scan(model, checkpoint, device, image_or_path, batch=64):
    """Return (tile logits, tile top-left corners, the RGB image scanned)."""
    cfg = checkpoint["config"]["model"]
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    image = load_image(image_or_path)
    if min(image.size) < TILE:
        image = image.resize((max(TILE, image.width), max(TILE, image.height)), Image.Resampling.BICUBIC)
    pixels = np.asarray(image)
    corners = [(x, y) for y in positions(image.height) for x in positions(image.width)]
    logits = []
    for i in range(0, len(corners), batch):
        tiles = np.stack([pixels[y:y + TILE, x:x + TILE] for x, y in corners[i:i + batch]])
        x = torch.from_numpy(tiles).to(device).permute(0, 3, 1, 2).float().div(255)
        logits += model((x - mean) / std).float().reshape(-1).cpu().tolist()
    return np.array(logits), corners, image


def tile_heatmap(image, logits, corners, low, high, max_side=1600):
    """Overlay tiles whose logit is above `low`, full colour at `high` (e.g. the suspicious/ai thresholds)."""
    scale = min(1.0, max_side / max(image.size))
    small = image.resize((round(image.width * scale), round(image.height * scale)))
    heat = np.zeros((small.height, small.width), np.float32)
    for (x, y), z in zip(corners, logits):
        value = float(np.clip((z - low) / max(high - low, 1e-6), 0, 1))
        x0, y0, x1, y1 = (round(v * scale) for v in (x, y, x + TILE, y + TILE))
        heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], value)
    return overlay(small, heat)
