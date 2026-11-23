"""Test the final CLIP detector on FraudBench's held-out review groups only.

Rows come from the v3 metadata (split == test, so no review group was seen in training or
threshold calibration). Each image is scored twice: the JPEG copy used everywhere else, and
the original downloaded file (fakes are often PNG) to check the result does not hinge on format.
Reports per generator, real reviews (Positive/Negative), category and three-tier verdicts.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.models.model1_full_image.clip_detector import analyze_clip, load_clip_detector
from src.models.model1_full_image.thresholds import load_thresholds, verdict

SUFFIXES = [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".PNG"]


def original(copy_path, mixed_root, fraudbench_root):
    rel = Path(copy_path).relative_to(Path(mixed_root) / "fraudbench")
    found = [p for p in (Path(fraudbench_root) / rel.parent / (rel.stem + s) for s in SUFFIXES) if p.exists()]
    if not found:
        raise FileNotFoundError(f"original of {copy_path} not found under {fraudbench_root}")
    return found[0]


def summarize(frame, col, th):
    real, fake = frame[frame.label == 0], frame[frame.label == 1]
    levels = lambda g: g[col].map(lambda s: verdict(s, th["suspicious"], th["ai"])).value_counts(normalize=True).reindex(["pass", "suspicious", "ai"], fill_value=0).round(4).to_dict()
    out = {"real": {"n": len(real), "under_0.30": round(float((real[col] < 0.30).mean()), 4), "verdicts": levels(real)}, "generators": {}}
    for name, part in real.groupby("review_kind"):
        out["real"][name] = {"n": len(part), "under_0.30": round(float((part[col] < 0.30).mean()), 4), "verdicts": levels(part)}
    for gen, g in fake.groupby("generator"):
        out["generators"][gen] = {"n": len(g), "group": g.generator_group.iloc[0], "auc": round(float(roc_auc_score(pd.concat([real, g]).label, pd.concat([real, g])[col])), 4),
                                  "at_or_above_suspicious": round(float((g[col] >= th["suspicious"]).mean()), 4), "verdicts": levels(g)}
    out["overall_auc"] = round(float(roc_auc_score(frame.label, frame[col])), 4)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", default="outputs/model1/clip_probe_v3data/clip_head.pt")
    parser.add_argument("--metadata", required=True, help="mixed_v3/metadata.csv")
    parser.add_argument("--fraudbench", required=True, help="Original FraudBench folder")
    parser.add_argument("--output", default="outputs/model1/fraudbench_test")
    args = parser.parse_args()
    meta = pd.read_csv(args.metadata, keep_default_na=False)
    rows = meta[(meta.split == "test") & meta.source.str.startswith("fraudbench")].copy()
    mixed_root = Path(args.metadata).parent
    parts = rows.image_path.map(lambda p: Path(p).relative_to(mixed_root / "fraudbench").parts)
    rows["category"], rows["review_kind"] = parts.str[0], parts.str[1]
    rows["generator"] = rows.domain.str.split(":").str[1]
    rows["original_path"] = rows.image_path.map(lambda p: str(original(p, mixed_root, args.fraudbench)))
    th = load_thresholds(Path(args.head).parent / "thresholds.json")
    detector = load_clip_detector(args.head)
    for col, path_col in [("score_jpeg_copy", "image_path"), ("score_original", "original_path")]:
        scores = []
        for p in tqdm(rows[path_col], desc=col):
            with Image.open(p) as image:
                scores.append(analyze_clip(detector, image, with_heatmap=False)["fake_probability"])
        rows[col] = scores
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "scores.csv", index=False)
    real = rows[rows.label == 0]
    by_category = {c: {"real_n": int((g.label == 0).sum()), "real_under_0.30": round(float((g[g.label == 0].score_original < 0.30).mean()), 4),
                       "fake_n": int((g.label == 1).sum()), "fake_at_or_above_suspicious": round(float((g[g.label == 1].score_original >= th["suspicious"]).mean()), 4)}
                   for c, g in rows.groupby("category")}
    report = {"thresholds": {k: th[k] for k in ["suspicious", "ai"]},
              "original_files": summarize(rows, "score_original", th), "jpeg_copies": summarize(rows, "score_jpeg_copy", th),
              "original_formats": rows.original_path.str.rsplit(".", n=1).str[-1].str.lower().groupby(rows.label).value_counts().to_dict(),
              "mean_abs_score_change_original_vs_copy": round(float((rows.score_original - rows.score_jpeg_copy).abs().mean()), 4),
              "by_category": by_category, "real_reviews_n": len(real)}
    report["original_formats"] = {f"{k[0]}:{k[1]}": int(v) for k, v in report["original_formats"].items()}
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["thresholds", "original_formats", "mean_abs_score_change_original_vs_copy"]}, ensure_ascii=False))
    print(json.dumps(report["original_files"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
