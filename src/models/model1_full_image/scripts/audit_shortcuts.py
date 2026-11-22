"""Measure how well file metadata alone separates Real from Fake (shortcut audit).

Trains a gradient-boosted classifier on width, height, aspect ratio, bytes per pixel and
JPEG quantization statistics only, never pixels. A high test AUC means the dataset carries
non-visual shortcuts that a CNN could exploit; the CNN's own score should be read against it.
Results are reported overall and separately for seen and hold-out generators.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def file_features(path):
    """Return non-pixel features for one image file."""
    with Image.open(path) as image:
        width, height = image.size
        tables = getattr(image, "quantization", None) or {}
    luma = np.array(tables.get(0, [0] * 64), dtype=float)
    chroma = np.array(tables.get(1, [0] * 64), dtype=float)
    return {"width": width, "height": height, "aspect": width / height, "is_square": float(width == height),
            "min_side": min(width, height), "bytes_per_pixel": Path(path).stat().st_size / (width * height),
            "q_luma_mean": luma.mean(), "q_luma_dc": luma[0], "q_chroma_mean": chroma.mean()}


def auc_or_none(labels, scores):
    return float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None


def audit(metadata_path, root, output):
    """Fit on train, report AUC on test; returns the report dict."""
    df = pd.read_csv(metadata_path, keep_default_na=False)
    root = Path(root)
    features = pd.DataFrame([file_features(root / p) for p in df.image_path])
    columns = list(features.columns)
    df = pd.concat([df.reset_index(drop=True), features], axis=1)
    train, test = df[df.split == "train"], df[df.split == "test"].copy()
    model = HistGradientBoostingClassifier(random_state=0).fit(train[columns], train.label)
    test["score"] = model.predict_proba(test[columns])[:, 1]
    real = test[test.label == 0]
    report = {"features": columns, "test_auc_overall": auc_or_none(test.label, test.score), "by_group": {}, "by_generator": {}}
    for group in ["seen", "holdout"]:
        part = pd.concat([real, test[test.generator_group == group]])
        report["by_group"][group] = auc_or_none(part.label, part.score)
    for generator in sorted(test.loc[test.label == 1, "domain"].unique()):
        part = pd.concat([real, test[test.domain == generator]])
        report["by_generator"][generator] = auc_or_none(part.label, part.score)
    report["single_feature_auc"] = {c: auc_or_none(test.label, test[c]) for c in columns}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "shortcut_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    df.drop(columns=["score"], errors="ignore").groupby("domain")[columns].median().to_csv(output / "feature_medians_by_domain.csv")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="outputs/model1/shortcut_audit")
    args = parser.parse_args()
    print(json.dumps(audit(args.metadata, args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
