"""Add the v6 partial-edit detector to the type-B test and report the combined (CLIP or v6) verdict.

Reads eval_type_b.py's scores.csv (CLIP scores of each type-B fake and its real original), scans the
same images with v6 at native resolution (max tile logit, thresholds from local_native_v6), and takes
the more severe of the two verdicts, as the demo does with --local-checkpoint.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.models.model1_full_image.inference import load_detector
from src.models.model1_full_image.local_detector import scan

SEVERITY = {"pass": 0, "suspicious": 1, "ai": 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fakes", required=True)
    parser.add_argument("--metadata", default=r"C:\Users\차기원\cvtp-data\mixed_v3\metadata.csv")
    parser.add_argument("--local-checkpoint", default="outputs/model1/local_native_v6/checkpoints/best_model.pth")
    parser.add_argument("--output", default="outputs/model1/typeB_test")
    args = parser.parse_args()
    output = Path(args.output)
    df = pd.read_csv(output / "scores.csv", keep_default_na=False, na_values=[""])
    th = json.loads((Path(args.local_checkpoint).resolve().parent.parent / "thresholds.json").read_text(encoding="utf-8"))
    meta = pd.read_csv(args.metadata, keep_default_na=False)
    real = meta[meta.source == "team_real"].assign(stem=lambda d: d.image_path.map(lambda p: Path(p).stem.split("_", 1)[1]))
    real_path = real.drop_duplicates("stem").set_index("stem").image_path
    model, checkpoint, device = load_detector(args.local_checkpoint, "auto")
    level = lambda z: "ai" if z >= th["ai"] else "suspicious" if z >= th["suspicious"] else "pass"
    fake_logit, real_logit = [], []
    for i, stem in enumerate(df.stem):
        fake_logit.append(float(scan(model, checkpoint, device, Path(args.fakes) / f"{stem}.png")[0].max()))
        real_logit.append(float(scan(model, checkpoint, device, real_path[stem])[0].max()))
        print(i + 1, stem, round(fake_logit[-1], 1), round(real_logit[-1], 1), flush=True)
    df["fake_local_logit"], df["real_local_logit"] = fake_logit, real_logit
    for side in ["fake", "real"]:
        df[f"{side}_local_verdict"] = df[f"{side}_local_logit"].map(level)
        df[f"{side}_combined_verdict"] = [max(a, b, key=SEVERITY.get) for a, b in zip(df[f"{side}_verdict"], df[f"{side}_local_verdict"])]
    df.to_csv(output / "scores_combined.csv", index=False)

    def rates(part):
        out = {"n": len(part)}
        for name in ["", "local_", "combined_"]:
            key = name.rstrip("_") or "clip"
            out[key] = {"fake_at_or_above_suspicious": round(float((part[f"fake_{name}verdict"] != "pass").mean()), 4),
                        "fake_ai": round(float((part[f"fake_{name}verdict"] == "ai").mean()), 4),
                        "real_at_or_above_suspicious": round(float((part[f"real_{name}verdict"] != "pass").mean()), 4),
                        "real_ai": round(float((part[f"real_{name}verdict"] == "ai").mean()), 4)}
        return out
    result = {"local_thresholds_logit": {"suspicious": th["suspicious"], "ai": th["ai"]}, "all": rates(df),
              "by_original_split": {s: rates(g) for s, g in df.groupby("real_split")},
              "by_memo_restored": {str(k): rates(g) for k, g in df.groupby("memo_restored")}}
    (output / "summary_combined.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
