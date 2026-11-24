"""Score a CNN checkpoint on v3 criteria *and* on local-edit test pairs, for mean / max / top-2 pooling.

Thresholds come from val rows without the local-edit crops (those 320 px crops are training
material, not demo-condition photos), so A and B stay comparable with v3/v4. Local-edit test
pairs are whole photos (short side 1024) scored like the demo. Reported per VAE:
  - paired AUC: edited photo vs its own original
  - share of edited photos at/above "suspicious", and share of their originals at/above it
--extra-pair ORIGINAL EDITED adds real tool edits (e.g. the teammate's Qwen edit) to the report.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.model1_full_image.inference import load_detector
from src.models.model1_full_image.scripts.evaluate_criteria import criteria_report, sample
from src.models.model1_full_image.scripts.evaluate_pooling import POOLS, crop_scores
from src.models.model1_full_image.utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/model1_local_edit_v5.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--extra-pair", nargs=2, action="append", default=[], metavar=("ORIGINAL", "EDITED"))
    args = parser.parse_args()
    config = load_config(args.config)
    meta = pd.read_csv(config["data"]["metadata"], keep_default_na=False)
    local = meta.source.isin(["local_orig", "local_edit"])
    extra = pd.DataFrame([{"image_path": p, "label": lbl, "domain": "real_tool_edit" if lbl else "real_tool_orig", "source_id": f"extra:{i}"}
                          for i, pair in enumerate(args.extra_pair) for lbl, p in enumerate(pair)])
    frames = {"val": sample(meta[(meta.split == "val") & ~local], 400), "test": sample(meta[(meta.split == "test") & ~local], 400),
              "local": pd.concat([meta[(meta.split == "test") & local], extra], ignore_index=True)}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    detector = load_detector(args.checkpoint)
    crops = {s: crop_scores(f, config["data"]["root"], detector, output / f"{s}_crops.json") for s, f in frames.items()}
    summary = {}
    for name, pool in POOLS.items():
        scored = {s: frames[s].assign(score=[float(pool(np.array(c))) for c in crops[s]]) for s in frames}
        r = criteria_report(scored["val"], scored["test"], output / name, {"checkpoint": args.checkpoint, "pooling": name})
        sus = r["thresholds"]["suspicious"]
        loc = scored["local"]
        edits = {}
        tool = loc.edit_vae.fillna("").replace("", "real_tool")  # extra pairs have no edit_vae
        for vae, g in loc[loc.label == 1].groupby(tool[loc.label == 1]):
            originals = loc[(loc.label == 0) & loc.source_id.isin(g.source_id)]
            edits[vae] = {"n": len(g), "paired_auc": round(float(roc_auc_score(np.r_[np.zeros(len(originals)), np.ones(len(g))], np.r_[originals.score, g.score])), 4),
                          "edited_at_or_above_suspicious": round(float((g.score >= sus).mean()), 4),
                          "original_at_or_above_suspicious": round(float((originals.score >= sus).mean()), 4)}
        if len(extra):
            edits["real_tool_scores"] = loc[loc.domain.str.startswith("real_tool")][["domain", "score"]].round(4).values.tolist()
        summary[name] = {"A": round(r["criterion_A_phone_real_under_0.30"], 4), "B": round(r["criterion_B_unseen_fake_at_or_above_suspicious"], 4),
                         "thresholds": {k: round(v, 4) for k, v in r["thresholds"].items()}, "local_edits": edits}
        print(name, json.dumps(summary[name], ensure_ascii=False), flush=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
