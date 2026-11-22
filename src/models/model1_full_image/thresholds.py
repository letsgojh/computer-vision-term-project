"""Three-level verdicts (pass / suspicious / AI) from two editable thresholds.

thresholds.json holds:
  suspicious: score >= this -> at least "suspicious" (the lower of: validation AI recall reaches
              target_recall, or validation real FPR reaches review_fpr)
  ai:         score >= this -> "ai" (set where validation real false-positive rate drops to max_real_fpr)
The team can edit the JSON by hand at any time; the demo slider overrides it per session.
"""

import json
from pathlib import Path

import numpy as np

LEVELS = ("pass", "suspicious", "ai")
LABELS_KO = {"pass": "통과 (실제 촬영으로 보임)", "suspicious": "의심 (사람이 확인 필요)", "ai": "AI 생성으로 판정"}


def verdict(score, suspicious, ai):
    """Map a fake probability to one of LEVELS."""
    if not 0 <= suspicious <= ai <= 1:
        raise ValueError("Thresholds must satisfy 0 <= suspicious <= ai <= 1")
    if score >= ai:
        return "ai"
    return "suspicious" if score >= suspicious else "pass"


def threshold_for_recall(labels, scores, target_recall):
    """Largest threshold whose Fake recall is still >= target_recall."""
    fake = np.sort(np.asarray(scores)[np.asarray(labels) == 1])
    if fake.size == 0:
        raise ValueError("No Fake samples")
    index = int(np.floor((1 - target_recall) * fake.size))
    return float(fake[min(index, fake.size - 1)])


def threshold_for_fpr(labels, scores, max_fpr):
    """Smallest threshold whose Real false-positive rate is <= max_fpr."""
    real = np.sort(np.asarray(scores)[np.asarray(labels) == 0])
    if real.size == 0:
        raise ValueError("No Real samples")
    allowed = int(np.floor(max_fpr * real.size))
    if allowed >= real.size:
        return 0.0
    # Threshold just above the (allowed+1)-th highest real score.
    return float(np.nextafter(real[real.size - 1 - allowed], 1.0))


def calibrate(labels, scores, target_recall=0.95, max_real_fpr=0.01, review_fpr=0.05):
    """Return a thresholds dict with the rates each threshold achieves on the given data.

    suspicious takes the lower of (AI recall >= target_recall) and (real FPR <= review_fpr),
    so the review band keeps catching AI even when validation is near-perfect; without the
    review_fpr term both thresholds collapse to one value on an easy validation set.
    """
    labels, scores = np.asarray(labels), np.asarray(scores)
    suspicious = min(threshold_for_recall(labels, scores, target_recall), threshold_for_fpr(labels, scores, review_fpr))
    ai = max(suspicious, threshold_for_fpr(labels, scores, max_real_fpr))
    real, fake = scores[labels == 0], scores[labels == 1]
    rates = {name: {"fake_recall": float((fake >= t).mean()), "real_fpr": float((real >= t).mean())}
             for name, t in [("suspicious", suspicious), ("ai", ai)]}
    return {"suspicious": suspicious, "ai": ai, "target_recall": target_recall,
            "max_real_fpr": max_real_fpr, "review_fpr": review_fpr, "validation_rates": rates}


def load_thresholds(path):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    verdict(0.0, values["suspicious"], values["ai"])  # validates ordering
    return values


def save_thresholds(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")
