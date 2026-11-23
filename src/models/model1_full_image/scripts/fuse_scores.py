"""Combine two detectors' scores (e.g. CNN + CLIP) with a 2-feature logistic regression fit on val.

Both score folders must come from evaluate_criteria.py / clip_probe.py on the same metadata,
so their val/test rows match. The fused score goes through the same criteria report.
Val sources are equally weighted so no single dataset decides the blend.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.model1_full_image.scripts.evaluate_criteria import criteria_report

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load(folder, split):
    frame = pd.read_csv(Path(folder) / f"{split}_scores.csv", keep_default_na=False)
    return frame.set_index("image_path")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="Score folder of detector A")
    parser.add_argument("--b", required=True, help="Score folder of detector B")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frames = {}
    for split in ["val", "test"]:
        a, b = load(args.a, split), load(args.b, split)
        if set(a.index) != set(b.index):
            raise ValueError(f"{split} rows differ between the two score folders")
        frame = a.drop(columns=["score"]).assign(score_a=a.score, score_b=b.score.reindex(a.index)).reset_index()
        frames[split] = frame
    val = frames["val"]
    x = torch.as_tensor(np.stack([logit(val.score_a), logit(val.score_b)], 1), dtype=torch.float32)
    y = torch.as_tensor(val.label.to_numpy(), dtype=torch.float32)
    w = torch.as_tensor(val.domain.map(1 / val.domain.value_counts()).to_numpy(), dtype=torch.float32)
    w = w / w.mean()
    head = torch.nn.Linear(2, 1)
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(head(x).squeeze(1), y, reduction="none") * w).mean()
        loss.backward()
        return loss
    optimizer.step(closure)
    weights = head.weight.detach().numpy()[0].tolist() + [float(head.bias.detach())]
    scored = {}
    for split, frame in frames.items():
        z = weights[0] * logit(frame.score_a) + weights[1] * logit(frame.score_b) + weights[2]
        scored[split] = frame.assign(score=1 / (1 + np.exp(-z)))
    report = criteria_report(scored["val"], scored["test"], args.output,
                             {"fusion": "sigmoid(wa*logit(a) + wb*logit(b) + c) fit on val", "weights_a_b_c": weights,
                              "a": str(Path(args.a).resolve()), "b": str(Path(args.b).resolve())})
    print("weights a, b, bias:", [round(v, 3) for v in weights])
    print({k: v for k, v in report.items() if k.startswith("criterion")})


if __name__ == "__main__":
    main()
