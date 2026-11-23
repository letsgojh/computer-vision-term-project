"""Choose a clip_layers head without the test split.

Val only holds generators that are also in train, so val loss rewards heads that memorise those
generators (clip_layers.py: L12 best on val loss, weakest on unseen test generators). Two
generalisation probes, both built from train/val data only:

1. Leave-generators-out: each fold drops one Defactify and one FraudBench generator from training;
   score = AUC of the dropped generators' val images vs all val reals (same-family novelty).
2. Different family: 400 DALL-E 3 images from Defactify's *validation* split (never in v3; captions
   of v3 train/test rows removed) vs all val reals. DALL-E 3 test numbers are then
   no longer fully blind (used for selection, not training); Midjourney and the FraudBench
   holdout generators stay blind.

Rule: pick the head with the highest min(probe 1, probe 2). Candidates are each layer set, plus
equal-weight logit averages of the final embedding head with a layer head (fixed 50/50, not tuned).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import roc_auc_score

from src.models.model1_full_image.scripts.clip_layers import LAYER_SETS, MODEL, extract, features
from src.models.model1_full_image.scripts.clip_probe import train_head
from src.models.model1_full_image.scripts.evaluate_criteria import criteria_report, sample
from src.models.model1_full_image.utils import load_config

FOLDS = [("defactify:sd21", "fraudbench:grok-imagine-image"), ("defactify:sdxl", "fraudbench:qwen-image-2.0-pro"),
         ("defactify:sd3", "fraudbench:wan2.7-image-pro"), ("defactify:sd21", "fraudbench:qwen-image-edit-max")]
FUSIONS = ["L16", "L20", "L8-24"]


def dalle3_selection_set(defactify_root, out_dir, meta, n=400):
    out_dir = Path(out_dir)
    paths = sorted(out_dir.glob("*.jpg"))
    if len(paths) >= n:
        return [str(p) for p in paths[:n]]
    frame = pd.concat([pq.read_table(f).to_pandas() for f in sorted(Path(defactify_root, "data").glob("validation-*.parquet"))])
    frame = frame[frame.Label_B == 4]  # 4 = DALL-E 3 (prepare_defactify.GENERATORS)
    caption = frame.Caption.map(lambda c: hashlib.sha1(c.strip().lower().encode("utf-8")).hexdigest()[:16])
    # Defactify val captions are shared by its COCO and generated images, so most already sit in v3 val;
    # only captions of v3 train/test rows would leak.
    used = set(meta[meta.split != "val"].group_id.str.split(":").str[-1])
    frame = frame[~caption.isin(used)].sample(n, random_state=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(frame.itertuples()):
        (out_dir / f"val_dalle3_{i:04d}.jpg").write_bytes(row.Image["bytes"])
    return [str(p) for p in sorted(out_dir.glob("*.jpg"))]


def main(config_path="configs/model1_mixed_v3.yaml", folder="outputs/model1/clip_layers"):
    config = load_config(config_path)
    data_root = Path(config["data"]["root"]).parent
    meta = pd.read_csv(config["data"]["metadata"], keep_default_na=False)
    train = pd.concat([g.sample(min(len(g), 3000), random_state=0) for _, g in meta[meta.split == "train"].groupby("source")])
    share = meta[meta.split == "train"].groupby("source").sample_weight.sum()
    train = train.assign(w=train.source.map(share) / train.source.map(train.source.value_counts()))
    frames = {"val": sample(meta[meta.split == "val"], 400), "test": sample(meta[meta.split == "test"], 400)}
    folder = Path(folder)
    cached = {s: tuple(np.load(folder / f"{s}_layers.npz")[k] for k in ["layers", "embeds"]) for s in ["train", "val", "test"]}

    sel_paths = dalle3_selection_set(data_root / "defactify", data_root / "defactify_dalle3_selection", meta)
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    model = CLIPVisionModelWithProjection.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
    cached["sel"] = extract(sel_paths, folder / "dalle3_selection_layers.npz", model, CLIPImageProcessor.from_pretrained(MODEL), torch.device("cuda"), 3)
    del model
    torch.cuda.empty_cache()

    val = frames["val"]
    real_val = (val.label == 0).to_numpy()

    def fit_and_score(name, keep):
        x = features(cached["train"], name, 0)[keep]
        mean, std = (0, 1) if name == "final" else (x.mean(0), x.std(0) + 1e-6)
        torch.manual_seed(0)
        head = train_head((x - mean) / std, train.label.to_numpy()[keep], train.w.to_numpy()[keep])
        with torch.no_grad():  # logits [N, 3 views] per split
            return {s: head(torch.as_tensor((np.stack([features(cached[s], name, v) for v in range(3)], 1) - mean) / std)).squeeze(-1).numpy()
                    for s in ["val", "test", "sel"]}

    everything = np.ones(len(train), bool)
    full = {name: fit_and_score(name, everything) for name in ["final", *LAYER_SETS]}
    folds = {name: [fit_and_score(name, ~train.domain.isin(f).to_numpy())["val"] for f in FOLDS] for name in ["final", *LAYER_SETS]}
    for layer in FUSIONS:  # fixed 50/50 average of logits
        full[f"final+{layer}"] = {s: (full["final"][s] + full[layer][s]) / 2 for s in ["val", "test", "sel"]}
        folds[f"final+{layer}"] = [(a + b) / 2 for a, b in zip(folds["final"], folds[layer])]

    result = {}
    for name in full:
        for agg, combine in [("center", lambda z: z[:, 0]), ("mean3", lambda z: z.mean(1))]:
            rows = [real_val | val.domain.isin(f).to_numpy() for f in FOLDS]
            lgo = np.mean([roc_auc_score(val.label.to_numpy()[m], combine(z)[m]) for m, z in zip(rows, folds[name])])
            sel_scores = np.concatenate([combine(full[name]["val"])[real_val], combine(full[name]["sel"])])
            dalle = roc_auc_score(np.r_[np.zeros(real_val.sum()), np.ones(len(sel_paths))], sel_scores)
            key = f"{name}_{agg}"
            result[key] = {"leave_generators_out_auc": round(float(lgo), 4), "dalle3_selection_auc": round(float(dalle), 4),
                           "selection_score": round(float(min(lgo, dalle)), 4)}
            print(key, result[key], flush=True)
    chosen = max(result, key=lambda k: result[k]["selection_score"])
    name, agg = chosen.rsplit("_", 1)
    combine = (lambda z: z[:, 0]) if agg == "center" else (lambda z: z.mean(1))
    scored = {s: frames[s].assign(score=1 / (1 + np.exp(-combine(full[name][s])))) for s in ["val", "test"]}
    report = criteria_report(scored["val"], scored["test"], folder / f"chosen_{chosen}", {"head": name, "aggregation": agg, "selection": "clip_layers_select.py"})
    result["chosen"] = chosen
    result["chosen_test"] = {"A": report["criterion_A_phone_real_under_0.30"], "B": report["criterion_B_unseen_fake_at_or_above_suspicious"]}
    (folder / "selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("chosen:", chosen, result["chosen_test"])


if __name__ == "__main__":
    main()
