"""Final CLIP head scored on three 224 crops that cover the whole photo (mean of logits).

CLIP's own preprocessing keeps only the center square, so a 3:4 phone photo loses ~25% of its
height. clip_layers.py cached, for every val/test image, the embedding of the center view
(CLIP preprocessing) and of the two end crops along the long side. This applies the *unchanged*
final head (clip_probe_v3data/clip_head.pt) to those views, averages the three logits, and
recalibrates thresholds on val. The new run folder holds the same head plus thresholds.json
with "views": 3, which the demo reads to score uploads the same way.
Chosen without the test split: val log loss 0.226 -> 0.220 and DALL-E 3 selection AUC
0.925 -> 0.931 (clip_layers_select.py) both favour the 3-crop mean.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.model1_full_image.scripts.evaluate_criteria import criteria_report, sample
from src.models.model1_full_image.thresholds import load_thresholds, save_thresholds
from src.models.model1_full_image.utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/model1_mixed_v3.yaml")
    parser.add_argument("--head", default="outputs/model1/clip_probe_v3data/clip_head.pt")
    parser.add_argument("--cache", default="outputs/model1/clip_layers", help="Folder with val/test_layers.npz from clip_layers.py")
    parser.add_argument("--output", default="outputs/model1/clip_probe_v3data_3crop")
    args = parser.parse_args()
    meta = pd.read_csv(load_config(args.config)["data"]["metadata"], keep_default_na=False)
    head = torch.nn.Linear(768, 1)
    head.load_state_dict(torch.load(args.head, map_location="cpu"))
    scored = {}
    for split in ["val", "test"]:
        frame = sample(meta[meta.split == split], 400)
        cached = np.load(Path(args.cache) / f"{split}_layers.npz")
        if [str(Path(p)) for p in cached["paths"]] != [str(Path(p)) for p in frame.image_path]:
            raise ValueError(f"{split}: cached features do not match the sampled rows")
        with torch.no_grad():
            logits = head(torch.as_tensor(cached["embeds"], dtype=torch.float32)).squeeze(-1).numpy()  # [N, 3 views]
        scored[split] = frame.assign(score=1 / (1 + np.exp(-logits.mean(1))), score_center=1 / (1 + np.exp(-logits[:, 0])))
    output = Path(args.output)
    report = criteria_report(scored["val"], scored["test"], output, {"head": str(Path(args.head).resolve()), "views": 3,
                                                                    "condition": "mean logit of CLIP center view + two end crops (short side 224)"})
    shutil.copyfile(args.head, output / "clip_head.pt")
    thresholds = load_thresholds(output / "thresholds.json")
    save_thresholds(output / "thresholds.json", {**thresholds, "views": 3})
    print(json.dumps({k: v for k, v in report.items() if k.startswith(("criterion", "thresholds"))}, indent=1))


if __name__ == "__main__":
    main()
