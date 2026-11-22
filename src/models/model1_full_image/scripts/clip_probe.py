"""Frozen CLIP ViT-L/14 image features + a logistic-regression head (UnivFD-style detector).

Features are extracted once (CLIP's own preprocessing: short side 224, center crop) and
cached. The head is trained on the train split with each source's sample_weight, so the
source balance matches the CNN runs. Val/test sampling and the acceptance criteria are the
same as scripts/evaluate_criteria.py, so the numbers are directly comparable.
(sklearn's linear models are blocked by Windows application control on this machine,
so the head is a small PyTorch model.)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from src.models.model1_full_image.scripts.evaluate_criteria import criteria_report, sample
from src.models.model1_full_image.utils import load_config

MODEL = "openai/clip-vit-large-patch14"


@torch.no_grad()
def extract(paths, root, cache, device, batch=32):
    if cache.exists():
        stored = np.load(cache, allow_pickle=False)
        if list(stored["paths"]) == list(paths):
            return stored["features"]
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    processor = CLIPImageProcessor.from_pretrained(MODEL)
    model = CLIPVisionModelWithProjection.from_pretrained(MODEL, torch_dtype=torch.float16).to(device).eval()
    features = []
    for i in tqdm(range(0, len(paths), batch), desc=f"clip {cache.stem}"):
        images = []
        for p in paths[i:i + batch]:
            with Image.open(Path(root) / p) as image:
                images.append(ImageOps.exif_transpose(image).convert("RGB"))
        inputs = processor(images=images, return_tensors="pt").to(device)
        embeds = model(pixel_values=inputs.pixel_values.half()).image_embeds.float()
        features.append(torch.nn.functional.normalize(embeds, dim=-1).cpu().numpy())
    features = np.concatenate(features)
    np.savez(cache, paths=np.array(paths), features=features)
    return features


def train_head(x, y, w, epochs=300, lr=0.01, weight_decay=1e-4):
    x, y = torch.as_tensor(x), torch.as_tensor(y, dtype=torch.float32)
    w = torch.as_tensor(w / w.sum() * len(w), dtype=torch.float32)
    head = torch.nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(head(x).squeeze(1), y, reduction="none") * w).mean()
        loss.backward()
        optimizer.step()
    return head


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Config whose data section points at the metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-source-train", type=int, default=3000)
    parser.add_argument("--per-domain", type=int, default=400)
    args = parser.parse_args()
    config = load_config(args.config)
    root = config["data"]["root"]
    meta = pd.read_csv(config["data"]["metadata"], keep_default_na=False)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = pd.concat([g.sample(min(len(g), args.per_source_train), random_state=0) for _, g in meta[meta.split == "train"].groupby("source")])
    # Per-source weight so each source keeps its configured class share after subsampling.
    share = meta[meta.split == "train"].groupby("source").sample_weight.sum()
    train = train.assign(w=train.source.map(share) / train.source.map(train.source.value_counts()))
    val = sample(meta[meta.split == "val"], args.per_domain)
    test = sample(meta[meta.split == "test"], args.per_domain)
    feats = {name: extract(list(frame.image_path), root, output / f"{name}_clip.npz", device)
             for name, frame in [("train", train), ("val", val), ("test", test)]}
    head = train_head(feats["train"], train.label.to_numpy(), train.w.to_numpy())
    torch.save(head.state_dict(), output / "clip_head.pt")
    with torch.no_grad():
        val = val.assign(score=torch.sigmoid(head(torch.as_tensor(feats["val"]))).squeeze(1).numpy())
        test = test.assign(score=torch.sigmoid(head(torch.as_tensor(feats["test"]))).squeeze(1).numpy())
    report = criteria_report(val, test, output, {"model": f"{MODEL} frozen + logistic head", "condition": "CLIP preprocessing (224 center crop)"})
    print({k: v for k, v in report.items() if k.startswith("criterion")})


if __name__ == "__main__":
    main()
