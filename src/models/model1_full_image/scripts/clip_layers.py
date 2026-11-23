"""Two follow-ups to the final CLIP detector, on the same v3 rows and criteria as clip_probe.py.

1. Intermediate layers: the CLS token of every transformer block is cached. Earlier blocks keep
   low-level generation traces that the final (semantic) embedding discards (RINE-style idea).
   Linear heads on several layer sets are trained on the same train rows as clip_probe and
   *chosen on val* (domain-balanced log loss). Test numbers are reported for every head.
2. Whole-image coverage: val/test are also scored on the two end crops along the long side of the
   short-side-224 image (CLIP's standard view is the center crop); logits are combined by mean or max.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from src.models.model1_full_image.scripts.clip_probe import train_head
from src.models.model1_full_image.scripts.evaluate_criteria import criteria_report, sample
from src.models.model1_full_image.utils import load_config

MODEL = "openai/clip-vit-large-patch14"
SIZE = 224
LAYER_SETS = {"L12": [12], "L16": [16], "L20": [20], "L8-24": [8, 12, 16, 20, 24], "Lall": list(range(1, 25))}


def load_views(path, n_views):
    """[full image, start crop, end crop]. The full image goes through CLIP's own preprocessing (center crop),
    so view 0 scores exactly like clip_probe / the demo; the end crops come from a short-side-224 resize."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
    if n_views == 1:
        return [image]
    w, h = image.size
    scale = SIZE / min(w, h)
    small = image.resize((max(SIZE, round(w * scale)), max(SIZE, round(h * scale))), Image.Resampling.BICUBIC)
    w, h = small.size
    span, along_x = (w - SIZE, True) if w >= h else (h - SIZE, False)
    box = lambda o: (o, 0, o + SIZE, SIZE) if along_x else (0, o, SIZE, o + SIZE)
    return [image, small.crop(box(0)), small.crop(box(span))]


@torch.no_grad()
def extract(paths, cache, model, processor, device, n_views, batch=8):
    if cache.exists():
        stored = np.load(cache)
        if list(stored["paths"]) == list(paths):
            return stored["layers"], stored["embeds"]
    layers, embeds = [], []
    with ThreadPoolExecutor(4) as pool:  # decode in threads, but only one batch of full-size photos at a time
        for i in tqdm(range(0, len(paths), batch), desc=cache.stem, mininterval=30):
            views = [v for vs in pool.map(lambda p: load_views(p, n_views), paths[i:i + batch]) for v in vs]
            pixels = processor(images=views, return_tensors="pt").pixel_values.to(device, torch.float16)
            out = model(pixel_values=pixels, output_hidden_states=True)
            cls = torch.stack([h[:, 0] for h in out.hidden_states[1:]], 1)  # [B*V, 24, width]
            layers.append(cls.view(-1, n_views, *cls.shape[1:]).cpu().numpy())
            e = torch.nn.functional.normalize(out.image_embeds.float(), dim=-1)
            embeds.append(e.view(-1, n_views, e.shape[-1]).cpu().numpy())
    layers, embeds = np.concatenate(layers), np.concatenate(embeds)
    np.savez(cache, paths=np.array(paths), layers=layers, embeds=embeds)
    return layers, embeds


def features(cached, name, view):
    layers, embeds = cached
    if name == "final":
        return embeds[:, view].astype(np.float32)
    return layers[:, view, [i - 1 for i in LAYER_SETS[name]]].reshape(len(layers), -1).astype(np.float32)


def balanced_log_loss(frame, logits):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(torch.as_tensor(logits), torch.as_tensor(frame.label.to_numpy(), dtype=torch.float64), reduction="none").numpy()
    return float(pd.Series(loss, index=frame.index).groupby(frame.domain.values).mean().mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/model1_mixed_v3.yaml")
    parser.add_argument("--output", default="outputs/model1/clip_layers")
    parser.add_argument("--per-source-train", type=int, default=3000)
    args = parser.parse_args()
    config = load_config(args.config)
    root, meta = config["data"]["root"], pd.read_csv(config["data"]["metadata"], keep_default_na=False)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train = pd.concat([g.sample(min(len(g), args.per_source_train), random_state=0) for _, g in meta[meta.split == "train"].groupby("source")])
    share = meta[meta.split == "train"].groupby("source").sample_weight.sum()
    train = train.assign(w=train.source.map(share) / train.source.map(train.source.value_counts()))
    frames = {"train": train, "val": sample(meta[meta.split == "val"], 400), "test": sample(meta[meta.split == "test"], 400)}

    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    device = torch.device("cuda")
    processor = CLIPImageProcessor.from_pretrained(MODEL)
    model = CLIPVisionModelWithProjection.from_pretrained(MODEL, torch_dtype=torch.float16).to(device).eval()
    cached = {s: extract([str(Path(root) / p) for p in f.image_path], output / f"{s}_layers.npz", model, processor, device, 1 if s == "train" else 3)
              for s, f in frames.items()}
    del model
    torch.cuda.empty_cache()

    summary = {}
    for name in ["final", *LAYER_SETS]:
        x = features(cached["train"], name, 0)
        mean, std = (0, 1) if name == "final" else (x.mean(0), x.std(0) + 1e-6)  # "final" = exactly clip_probe's head
        torch.manual_seed(0)
        head = train_head((x - mean) / std, train.label.to_numpy(), train.w.to_numpy())
        logits = {}
        with torch.no_grad():
            for s in ["val", "test"]:
                logits[s] = np.stack([head(torch.as_tensor((features(cached[s], name, v) - mean) / std)).squeeze(1).numpy() for v in range(3)], 1)
        for agg, combine in [("center", lambda z: z[:, 0]), ("mean3", lambda z: z.mean(1)), ("max3", lambda z: z.max(1))]:
            key = f"{name}_{agg}"
            scored = {s: frames[s].assign(score=1 / (1 + np.exp(-combine(logits[s])))) for s in ["val", "test"]}
            report = criteria_report(scored["val"], scored["test"], output / key, {"head": name, "aggregation": agg})
            test = scored["test"]
            summary[key] = {"val_balanced_log_loss": round(balanced_log_loss(frames["val"], combine(logits["val"])), 4),
                            "A": round(report["criterion_A_phone_real_under_0.30"], 4), "B": round(report["criterion_B_unseen_fake_at_or_above_suspicious"], 4),
                            "real_under_0.30": {d: round(v["under_0.30"], 3) for d, v in report["real_by_domain"].items()},
                            "holdout_at_or_above_suspicious": {d: round(v["at_or_above_suspicious"], 3) for d, v in report["fake_by_domain"].items() if v["group"] == "holdout"},
                            "thresholds": {k: round(v, 3) for k, v in report["thresholds"].items()}}
            print(key, {k: summary[key][k] for k in ["val_balanced_log_loss", "A", "B"]}, flush=True)
    best = min(summary, key=lambda k: summary[k]["val_balanced_log_loss"])
    summary["chosen_on_val"] = best
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("chosen on val:", best, summary[best])


if __name__ == "__main__":
    main()
