"""Break test predictions down by seen vs hold-out generator and by three-level verdict.

Each fake group is scored against the same set of real test images, so AUCs are comparable.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.model1_full_image.thresholds import LEVELS, load_thresholds, verdict


def group_report(real, fake, thresholds):
    scores = pd.concat([real.fake_probability, fake.fake_probability])
    labels = [0] * len(real) + [1] * len(fake)
    report = {"n_real": len(real), "n_fake": len(fake), "roc_auc": float(roc_auc_score(labels, scores))}
    for name in ["suspicious", "ai"]:
        report[f"fake_recall_at_{name}"] = float((fake.fake_probability >= thresholds[name]).mean())
    return report


def analyze(predictions_path, thresholds_path, output, holdout=("dalle3", "midjourney")):
    df = pd.read_csv(predictions_path, keep_default_na=False)
    thresholds = load_thresholds(thresholds_path)
    df["verdict"] = [verdict(s, thresholds["suspicious"], thresholds["ai"]) for s in df.fake_probability]
    holdout = set(holdout)
    real, fake = df[df.label == 0], df[df.label == 1]
    report = {"thresholds": {k: thresholds[k] for k in ["suspicious", "ai"]}, "holdout_generators": sorted(holdout),
              "real_verdict_rates": real.verdict.value_counts(normalize=True).reindex(LEVELS, fill_value=0).to_dict(),
              "groups": {}, "generators": {}}
    report["groups"]["seen"] = group_report(real, fake[~fake.domain.isin(holdout)], thresholds)
    report["groups"]["holdout"] = group_report(real, fake[fake.domain.isin(holdout)], thresholds)
    for generator, part in fake.groupby("domain"):
        report["generators"][generator] = group_report(real, part, thresholds)
    table = pd.crosstab(df.domain, df.verdict, normalize="index").reindex(columns=LEVELS, fill_value=0)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "generalization.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    table.to_csv(output / "verdict_rates_by_domain.csv")
    print(json.dumps(report, indent=2))
    print(table.round(3))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="test_predictions.csv from evaluate.py")
    parser.add_argument("--thresholds", required=True, help="thresholds.json from calibrate_thresholds.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--holdout", nargs="*", default=["dalle3", "midjourney"])
    args = parser.parse_args()
    analyze(args.predictions, args.thresholds, args.output, args.holdout)


if __name__ == "__main__":
    main()
