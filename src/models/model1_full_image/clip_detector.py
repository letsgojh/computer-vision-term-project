"""Frozen CLIP ViT-L/14 + linear head (from scripts/clip_probe.py) for the demo.

Scoring matches clip_probe.extract exactly (CLIP preprocessing: short side 224, center crop;
fp16 on GPU; L2-normalised image embedding). The heatmap is Grad-CAM on the patch tokens
entering the last transformer block (the final block's patch outputs never reach the CLS
token, so their gradient is zero). It covers only the center square CLIP actually sees.
"""

import numpy as np
import torch
from PIL import ImageOps

from src.models.model1_full_image.inference import overlay
from src.models.model1_full_image.utils import select_device

MODEL = "openai/clip-vit-large-patch14"


def load_clip_detector(head_path, device_name="auto"):
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    device = select_device(device_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = CLIPVisionModelWithProjection.from_pretrained(MODEL, torch_dtype=dtype).to(device).eval()
    head = torch.nn.Linear(model.config.projection_dim, 1)
    head.load_state_dict(torch.load(head_path, map_location="cpu"))
    return {"model": model, "processor": CLIPImageProcessor.from_pretrained(MODEL), "head": head.to(device), "device": device, "dtype": dtype}


def center_square(image, size):
    short = min(image.size)
    left, top = (image.width - short) // 2, (image.height - short) // 2
    return image.crop((left, top, left + short, top + short)).resize((size, size))


def analyze_clip(detector, image, with_heatmap=True):
    """Same result keys as inference.analyze: fake_probability, crop_probabilities, heatmap."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    model, head = detector["model"], detector["head"]
    pixels = detector["processor"](images=[image], return_tensors="pt").pixel_values.to(detector["device"], detector["dtype"])
    with torch.set_grad_enabled(with_heatmap):
        out = model(pixel_values=pixels, output_hidden_states=with_heatmap)
        logit = head(torch.nn.functional.normalize(out.image_embeds.float(), dim=-1)).squeeze()
    probability = float(torch.sigmoid(logit.detach()))
    result = {"fake_probability": probability, "crop_probabilities": [probability], "work_size": image.size}
    if with_heatmap:
        tokens = out.hidden_states[-2]  # [1, 1 + grid*grid, width]
        grad, = torch.autograd.grad(logit, tokens)
        weights = grad[0, 1:].float().mean(0)  # Grad-CAM channel weights over patch tokens
        cam = torch.relu(tokens[0, 1:].detach().float() @ weights)
        grid = int(cam.numel() ** 0.5)
        cam = cam.view(grid, grid).cpu().numpy()
        cam = np.nan_to_num(cam / cam.max()) if cam.max() > 0 else cam
        square = center_square(image, 448)
        result["heatmap"] = overlay(square, cam.astype(np.float32))
        result["heatmap_values"] = cam
    return result
