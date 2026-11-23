"""Re-score val/test with a CNN checkpoint keeping every crop probability, then compare poolings.

The demo averages the overlapping 256 crops. A local AI edit (e.g. only the memo's date rewritten,
~2% of pixels) lifts one or two crops but vanishes in the mean, so max / top-2 mean are compared
with the mean under the same criteria (thresholds from val each time).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.models.model1_full_image.inference import analyze, load_detector
from src.models.model1_full_image.scripts.evaluate_criteria import criteria_report, sample
from src.models.model1_full_image.utils import load_config

POOLS = {"mean": np.mean, "max": np.max, "top2": lambda p: np.sort(p)[-2:].mean()}


def crop_scores(frame, root, detector, cache):
    if cache.exists():
        stored = json.loads(cache.read_text())
        if stored["paths"] == list(frame.image_path):
            return stored["crops"]
    model, checkpoint, device = detector
    crops = []
    for path in tqdm(frame.image_path, desc=cache.stem, mininterval=30):
        with Image.open(Path(root) / path) as image:
            crops.append(analyze(model, checkpoint, device, image.convert("RGB"), with_heatmap=False)["crop_probabilities"])
    cache.write_text(json.dumps({"paths": list(frame.image_path), "crops": crops}))
    return crops


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/model1_mixed_v3.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    meta = pd.read_csv(config["data"]["metadata"], keep_default_na=False)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    detector = load_detector(args.checkpoint)
    frames = {s: sample(meta[meta.split == s], 400) for s in ["val", "test"]}
    crops = {s: crop_scores(f, config["data"]["root"], detector, output / f"{s}_crops.json") for s, f in frames.items()}
    summary = {}
    for name, pool in POOLS.items():
        scored = {s: frames[s].assign(score=[float(pool(np.array(c))) for c in crops[s]]) for s in frames}
        r = criteria_report(scored["val"], scored["test"], output / name, {"checkpoint": args.checkpoint, "pooling": name})
        summary[name] = {"A": r["criterion_A_phone_real_under_0.30"], "B": r["criterion_B_unseen_fake_at_or_above_suspicious"],
                         "thresholds": r["thresholds"], "real_under_0.30": {d: v["under_0.30"] for d, v in r["real_by_domain"].items()},
                         "real_flagged_ai": {d: v["flagged_ai"] for d, v in r["real_by_domain"].items()}}
        print(name, {k: summary[name][k] for k in ["A", "B", "thresholds"]}, flush=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
