"""Score val/test in the demo condition and check the two acceptance criteria.

Demo condition = inference.analyze (short side 512, overlapping crops, mean probability).
Thresholds are calibrated on the *validation* split in that same condition (thresholds.calibrate).

Criterion A (real phone photos): share of held-out team photographers' real photos with
             probability < 0.30 must be >= 0.90.
Criterion B (unseen generators): share of held-out-generator fakes at or above the
             "suspicious" threshold must be >= 0.80.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.models.model1_full_image.inference import analyze, load_detector
from src.models.model1_full_image.thresholds import calibrate, save_thresholds
from src.models.model1_full_image.utils import load_config


def sample(frame, per_domain, seed=0):
    """Cap large Defactify domains to keep scoring time reasonable; keep everything else."""
    parts = [g.sample(min(len(g), per_domain), random_state=seed) if d.startswith("defactify:") else g
             for d, g in frame.groupby("domain")]
    return pd.concat(parts)


def score(frame, root, detector):
    model, checkpoint, device = detector
    scores = []
    for path in tqdm(frame.image_path, desc="score"):
        with Image.open(Path(root) / path) as image:
            scores.append(analyze(model, checkpoint, device, image.convert("RGB"), with_heatmap=False)["fake_probability"])
    return frame.assign(score=scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-domain", type=int, default=400, help="Max images per Defactify domain per split")
    parser.add_argument("--team-photographer-prefix", default="team_real:")
    args = parser.parse_args()
    config = load_config(args.config)
    meta = pd.read_csv(config["data"]["metadata"], keep_default_na=False)
    detector = load_detector(args.checkpoint)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    val = score(sample(meta[meta.split == "val"], args.per_domain), config["data"]["root"], detector)
    test = score(sample(meta[meta.split == "test"], args.per_domain), config["data"]["root"], detector)
    report = criteria_report(val, test, output, {"checkpoint": str(Path(args.checkpoint).resolve()),
                                                 "condition": "demo (inference.analyze, short side 512)"}, args.team_photographer_prefix)
    print(json.dumps(report, indent=2))


def criteria_report(val, test, output, extra=None, phone_prefix="team_real:"):
    """Calibrate thresholds on val scores, then compute criteria A/B and per-domain rates on test."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    val.to_csv(output / "val_scores.csv", index=False)
    test.to_csv(output / "test_scores.csv", index=False)
    thresholds = calibrate(val.label, val.score)
    thresholds.update(extra or {})
    save_thresholds(output / "thresholds.json", thresholds)
    real, fake = test[test.label == 0], test[test.label == 1]
    phone = real[real.domain.str.startswith(phone_prefix)]
    unseen = fake[fake.generator_group == "holdout"]
    report = {"thresholds": {k: thresholds[k] for k in ["suspicious", "ai"]},
              "criterion_A_phone_real_under_0.30": float((phone.score < 0.30).mean()),
              "criterion_A_pass": bool((phone.score < 0.30).mean() >= 0.90),
              "criterion_B_unseen_fake_at_or_above_suspicious": float((unseen.score >= thresholds["suspicious"]).mean()),
              "criterion_B_pass": bool((unseen.score >= thresholds["suspicious"]).mean() >= 0.80),
              "real_by_domain": {}, "fake_by_domain": {}, "auc_vs_all_test_reals": {}}
    for domain, g in real.groupby("domain"):
        report["real_by_domain"][domain] = {"n": len(g), "median": float(g.score.median()), "under_0.30": float((g.score < 0.30).mean()),
                                            "flagged_ai": float((g.score >= thresholds["ai"]).mean())}
    for domain, g in fake.groupby("domain"):
        report["fake_by_domain"][domain] = {"n": len(g), "group": g.generator_group.iloc[0],
                                            "at_or_above_suspicious": float((g.score >= thresholds["suspicious"]).mean()),
                                            "at_or_above_ai": float((g.score >= thresholds["ai"]).mean())}
        both = pd.concat([real, g])
        report["auc_vs_all_test_reals"][domain] = float(roc_auc_score(both.label, both.score))
    (output / "criteria.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
