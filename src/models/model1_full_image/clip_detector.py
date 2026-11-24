"""Frozen CLIP ViT-L/14 + linear head (from scripts/clip_probe.py) for the demo.

views=1: CLIP's own preprocessing only (short side 224, center crop) - matches clip_probe.
views=3: that center view plus the two end crops along the long side of the short-side-224
         image, so the whole photo is seen; the score is the mean of the three logits. This
         matches scripts/clip_layers.load_views / clip_crop_coverage.py exactly.
fp16 on GPU, L2-normalised image embedding. The heatmap is Grad-CAM on the patch tokens entering
the last transformer block (the final block's patch outputs never reach the CLS token, so their
gradient is zero), drawn where each view sits on the photo.
"""

import numpy as np
import torch
from PIL import Image, ImageOps

from src.models.model1_full_image.inference import overlay
from src.models.model1_full_image.utils import select_device

MODEL = "openai/clip-vit-large-patch14"
SIZE = 224


def load_clip_detector(head_path, device_name="auto"):
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    device = select_device(device_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = CLIPVisionModelWithProjection.from_pretrained(MODEL, torch_dtype=dtype).to(device).eval()
    head = torch.nn.Linear(model.config.projection_dim, 1)
    head.load_state_dict(torch.load(head_path, map_location="cpu"))
    return {"model": model, "processor": CLIPImageProcessor.from_pretrained(MODEL), "head": head.to(device), "device": device, "dtype": dtype}


def view_layout(image):
    """Short-side-224 copy of the photo and the (left, top) of the center, start and end 224 crops."""
    w, h = image.size
    scale = SIZE / min(w, h)
    small = image.resize((max(SIZE, round(w * scale)), max(SIZE, round(h * scale))), Image.Resampling.BICUBIC)
    span, along_x = (small.width - SIZE, True) if small.width >= small.height else (small.height - SIZE, False)
    offsets = [(o, 0) if along_x else (0, o) for o in (span // 2, 0, span)]
    return small, offsets


def analyze_clip(detector, image, with_heatmap=True, views=1):
    """Same result keys as inference.analyze: fake_probability, crop_probabilities, heatmap."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    model, head, processor = detector["model"], detector["head"], detector["processor"]
    small, offsets = view_layout(image)
    crops = [small.crop((x, y, x + SIZE, y + SIZE)) for x, y in offsets[1:views]]
    pixels = processor(images=[image] + crops, return_tensors="pt").pixel_values.to(detector["device"], detector["dtype"])
    with torch.set_grad_enabled(with_heatmap):
        out = model(pixel_values=pixels, output_hidden_states=with_heatmap)
        logits = head(torch.nn.functional.normalize(out.image_embeds.float(), dim=-1)).squeeze(-1)  # one per view
    probability = float(torch.sigmoid(logits.detach().mean()))
    result = {"fake_probability": probability, "crop_probabilities": torch.sigmoid(logits.detach()).cpu().tolist(), "work_size": image.size}
    if with_heatmap:
        tokens = out.hidden_states[-2]  # [views, 1 + grid*grid, width]
        grad, = torch.autograd.grad(logits.sum(), tokens)  # each view's logit only depends on its own tokens
        weights = grad[:, 1:].float().mean(1, keepdim=True)  # Grad-CAM channel weights per view
        cams = torch.relu((tokens[:, 1:].detach().float() * weights).sum(-1))
        grid = int(cams.shape[1] ** 0.5)
        cams = torch.nn.functional.interpolate(cams.view(-1, 1, grid, grid), size=(SIZE, SIZE), mode="bilinear", align_corners=False)[:, 0]
        canvas = np.zeros((small.height, small.width), np.float32)
        for (x, y), cam in zip(offsets[:views], cams.cpu().numpy()):
            canvas[y:y + SIZE, x:x + SIZE] = np.maximum(canvas[y:y + SIZE, x:x + SIZE], cam)
        canvas = np.nan_to_num(canvas / canvas.max()) if canvas.max() > 0 else canvas
        result["heatmap"] = overlay(small.resize((small.width * 2, small.height * 2), Image.Resampling.BICUBIC), canvas)
        result["heatmap_values"] = canvas
    return result
