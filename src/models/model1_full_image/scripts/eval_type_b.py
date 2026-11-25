"""Score the team's type-B fakes and their real originals with the final CLIP model (3 crops).

Type B (Train-A-만들기/modal_qwen_edit.py + restore_memo.py): Qwen-Image-Edit-2509 redraws the
photo at long side 1344 with a different product, then the original memo region is pasted back.
Each fake is paired with its real original (same <photographer>_<object>_<idx> stem) from the v3
metadata, so the split of the original (train = seen in training, test = held-out photographers)
is reported next to the score.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from src.models.model1_full_image.clip_detector import analyze_clip, load_clip_detector
from src.models.model1_full_image.thresholds import load_thresholds, verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fakes", required=True, help="Folder of type-B PNGs named <photographer>_<object>_<idx>.png")
    parser.add_argument("--metadata", default=r"C:\Users\차기원\cvtp-data\mixed_v3\metadata.csv")
    parser.add_argument("--run", default="outputs/model1/clip_probe_v3data_3crop")
    parser.add_argument("--output", default="outputs/model1/typeB_test")
    args = parser.parse_args()
    run, output = Path(args.run), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    th = load_thresholds(run / "thresholds.json")
    views = int(th.get("views", 1))
    meta = pd.read_csv(args.metadata, keep_default_na=False)
    real = meta[meta.source == "team_real"].assign(stem=lambda d: d.image_path.map(lambda p: Path(p).stem.split("_", 1)[1]))
    real = real.drop_duplicates("stem").set_index("stem")
    report = pd.read_csv(Path(args.fakes) / "restore_report.csv").set_index("stem") if (Path(args.fakes) / "restore_report.csv").exists() else None
    detector = load_clip_detector(run / "clip_head.pt")
    rows = []
    for path in sorted(Path(args.fakes).glob("*.png")):
        stem = path.stem
        row = {"stem": stem, "photographer": stem.split("_")[0], "memo_restored": int(report.found[stem]) if report is not None and stem in report.index else -1}
        row["fake_score"] = analyze_clip(detector, Image.open(path), with_heatmap=False, views=views)["fake_probability"]
        if stem in real.index:
            row["real_split"] = real.split[stem]
            row["real_score"] = analyze_clip(detector, Image.open(real.image_path[stem]), with_heatmap=False, views=views)["fake_probability"]
        rows.append(row)
        print(len(rows), stem, round(row["fake_score"], 3), round(row.get("real_score", float("nan")), 3), flush=True)
    df = pd.DataFrame(rows)
    df["fake_verdict"] = df.fake_score.map(lambda s: verdict(s, th["suspicious"], th["ai"]))
    df["real_verdict"] = df.real_score.map(lambda s: verdict(s, th["suspicious"], th["ai"]) if pd.notna(s) else "")
    df.to_csv(output / "scores.csv", index=False)

    def summary(part):
        paired = part.dropna(subset=["real_score"])
        return {"n": len(part), "fake_at_or_above_suspicious": round(float((part.fake_verdict != "pass").mean()), 4),
                "fake_ai": round(float((part.fake_verdict == "ai").mean()), 4), "fake_median": round(float(part.fake_score.median()), 4),
                "paired": len(paired), "real_under_0.30": round(float((paired.real_score < 0.30).mean()), 4) if len(paired) else None,
                "fake_higher_than_its_real": round(float((paired.fake_score > paired.real_score).mean()), 4) if len(paired) else None}
    result = {"thresholds": {"suspicious": th["suspicious"], "ai": th["ai"], "views": views}, "all": summary(df)}
    result["by_original_split"] = {s: summary(g) for s, g in df.groupby(df.real_split.fillna("unpaired"))}
    result["by_photographer"] = {p: summary(g) for p, g in df.groupby("photographer")}
    result["by_memo_restored"] = {str(k): summary(g) for k, g in df.groupby("memo_restored")}
    (output / "summary.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
