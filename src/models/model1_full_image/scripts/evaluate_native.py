"""Evaluate the v6 native-resolution local-edit detector on whole photos.

Each photo is scanned at native resolution with 256 tiles (stride 256, last row/column aligned to
the edge); the image score is the max, top-3 or top-10 mean tile *logit* (probabilities saturate
at 1.0 for many tiles, e.g. photographed screens, so they cannot rank photos). Thresholds use *real val
photos only* (v3 val split): "suspicious" = 5% real FPR, "ai" = 1% real FPR, because the risk of
a whole-photo scan is false alarms on real photos. Reported on test:
  - real photos (held-out phones, FraudBench, COCO): share at/above each threshold
  - simulated local edits per VAE (MoVQ never trained): detection rate, AUC vs their originals,
    and whether the top tile overlaps the edited box (is it finding the edit?)
  - --extra-pair ORIGINAL EDITED: real tool edits (e.g. the teammate's Qwen edit)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.models.model1_full_image.inference import load_detector
from src.models.model1_full_image.local_detector import TILE, scan
from src.models.model1_full_image.thresholds import save_thresholds


def summarize(logits):
    ranked = np.sort(logits)[::-1]
    return {"max": float(ranked[0]), "top3": float(ranked[:3].mean()), "top10": float(ranked[:10].mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--v3-metadata", required=True)
    parser.add_argument("--v6-metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extra-pair", nargs=2, action="append", default=[], metavar=("ORIGINAL", "EDITED"))
    args = parser.parse_args()
    v3 = pd.read_csv(args.v3_metadata, keep_default_na=False)
    v6 = pd.read_csv(args.v6_metadata, keep_default_na=False)
    reals = v3[(v3.label == 0)]
    pick = lambda g, n: g.sample(min(n, len(g)), random_state=0)
    val = pd.concat([pick(g, 200) for _, g in reals[reals.split == "val"].groupby("source")])
    test_real = pd.concat([pick(g, 200) if s != "team_real" else g for s, g in reals[reals.split == "test"].groupby("source")])
    test_edit = v6[v6.split == "test"]
    extra = pd.DataFrame([{"image_path": p, "label": lbl, "domain": "real_tool_edit" if lbl else "real_tool_orig", "source_id": f"extra:{i}",
                           "edit_box": "", "edit_vae": "real_tool" if lbl else ""} for i, pair in enumerate(args.extra_pair) for lbl, p in enumerate(pair)])
    frames = {"val_real": val, "test_real": test_real, "test_edit": pd.concat([test_edit, extra], ignore_index=True)}
    model, checkpoint, device = load_detector(args.checkpoint)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scored = {}
    for name, frame in frames.items():
        cache = output / f"{name}_scores.csv"
        if cache.exists():
            scored[name] = pd.read_csv(cache, keep_default_na=False)
            continue
        rows = []
        for item in tqdm(list(frame.itertuples()), desc=name, mininterval=30):
            probs, corners, _ = scan(model, checkpoint, device, item.image_path)
            top = corners[int(probs.argmax())]
            hit = ""
            if getattr(item, "edit_box", ""):
                x0, y0, x1, y1 = map(int, item.edit_box.split(","))
                hit = int(top[0] < x1 and top[0] + TILE > x0 and top[1] < y1 and top[1] + TILE > y0)
            rows.append({**summarize(probs), "tiles": len(probs), "top_tile": f"{top[0]},{top[1]}", "top_tile_in_box": hit})
        scored[name] = pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
        scored[name].to_csv(cache, index=False)
    report = {}
    for agg in ["max", "top3", "top10"]:
        sus, ai = (float(np.quantile(scored["val_real"][agg], q)) for q in (0.95, 0.99))
        tr, te = scored["test_real"], scored["test_edit"]
        r = {"thresholds": {"suspicious_5pct_val_fpr": sus, "ai_1pct_val_fpr": ai}, "real": {}, "edits": {}}
        for d, g in tr.groupby("domain"):
            r["real"][d] = {"n": len(g), "at_or_above_suspicious": round(float((g[agg] >= sus).mean()), 4), "at_or_above_ai": round(float((g[agg] >= ai).mean()), 4)}
        for vae, g in te[te.label == 1].groupby("edit_vae"):
            orig = te[(te.label == 0) & te.source_id.isin(g.source_id)]
            hits = pd.to_numeric(g.top_tile_in_box, errors="coerce")
            r["edits"][vae] = {"n": len(g), "at_or_above_suspicious": round(float((g[agg] >= sus).mean()), 4),
                               "at_or_above_ai": round(float((g[agg] >= ai).mean()), 4),
                               "originals_at_or_above_suspicious": round(float((orig[agg] >= sus).mean()), 4),
                               "auc_vs_own_originals": round(float(roc_auc_score(np.r_[np.zeros(len(orig)), np.ones(len(g))], np.r_[orig[agg], g[agg]])), 4) if len(orig) else None,
                               "top_tile_in_edit_box": round(float(hits.mean()), 4) if hits.notna().any() else None}
        if len(extra):
            r["real_tool_scores"] = te[te.domain.astype(str).str.startswith("real_tool")][["domain", agg]].round(4).values.tolist()
        report[agg] = r
        print(agg, json.dumps(r, ensure_ascii=False), flush=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # The demo reads these next to the checkpoints folder (image score = max tile logit).
    th = report["max"]["thresholds"]
    save_thresholds(Path(args.checkpoint).resolve().parent.parent / "thresholds.json",
                    {"suspicious": th["suspicious_5pct_val_fpr"], "ai": th["ai_1pct_val_fpr"], "score": "max tile logit",
                     "rule": "suspicious = 5% / ai = 1% false-alarm rate on real val photos", "source": str((output / "report.json").resolve())})


if __name__ == "__main__":
    main()
