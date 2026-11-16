"""이진 분류 지표. 가짜(label 1)를 양성으로 둔다."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.common.schema import POSTPROCESSES
from src.evaluation.vocab import ALL_SCOPES


def _clean(labels, probs):
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probs, dtype=float)
    if y.size == 0:
        return y.astype(int), p
    mask = np.isfinite(y) & np.isfinite(p)
    return y[mask].astype(int), p[mask].astype(float)


def _div(num: int, den: int):
    if den == 0:
        return None
    return float(num / den)


def expected_calibration_error(y, p, n_bins: int = 10):
    """확률이 정답 빈도와 얼마나 어긋나는지. 낮을수록 보정이 좋다."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    if n == 0:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        total += (count / n) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def binary_metrics(labels, probs, threshold: float) -> dict:
    """prob >= threshold 를 가짜 판정으로 두고 지표를 계산한다.

    분모가 0이면 그 지표는 None이다. 예측을 한 번도 가짜로 두지 않았는데
    Precision을 0으로 적으면, 오탐이 없다는 뜻과 섞이기 때문이다.
    """
    y, p = _clean(labels, probs)
    pred = (p >= threshold).astype(int) if len(p) else np.array([], dtype=int)
    tp = int(((pred == 1) & (y == 1)).sum()) if len(y) else 0
    fp = int(((pred == 1) & (y == 0)).sum()) if len(y) else 0
    tn = int(((pred == 0) & (y == 0)).sum()) if len(y) else 0
    fn = int(((pred == 0) & (y == 1)).sum()) if len(y) else 0
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    specificity = _div(tn, tn + fp)
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = float(2 * precision * recall / (precision + recall))
    balanced = None
    youden = None
    if recall is not None and specificity is not None:
        balanced = float((recall + specificity) / 2)
        youden = float(recall + specificity - 1)
    auc = None
    ap = None
    if len(y) and len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, p))
        ap = float(average_precision_score(y, p))
    elif len(y) and int((y == 1).sum()) > 0 and len(np.unique(y)) == 1:
        ap = float(average_precision_score(y, p))
    brier = float(brier_score_loss(y, np.clip(p, 0, 1))) if len(y) else None
    return {
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()) if len(y) else 0,
        "n_neg": int((y == 0).sum()) if len(y) else 0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": float((tp + tn) / len(y)) if len(y) else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "specificity": specificity,
        "balanced_accuracy": balanced,
        "fpr": _div(fp, fp + tn),
        "average_precision": ap,
        "brier": brier,
        "ece": expected_calibration_error(y, p) if len(y) else None,
        "youden_j": youden,
        "mean_prob_pos": float(p[y == 1].mean()) if len(y) and (y == 1).any() else None,
        "mean_prob_neg": float(p[y == 0].mean()) if len(y) and (y == 0).any() else None,
        "prob_gap": (
            float(p[y == 1].mean() - p[y == 0].mean())
            if len(y) and (y == 1).any() and (y == 0).any()
            else None
        ),
        "uncertain_rate": float(np.mean(np.abs(p - threshold) <= 0.05)) if len(y) else None,
        "threshold": float(threshold),
    }


def scope_frame(frame: pd.DataFrame, scope: str, postprocess: str) -> pd.DataFrame:
    """한 칸에 들어갈 행. A/B/C는 그 가짜와 진짜만 남긴다."""
    part = frame[frame["postprocess"] == postprocess]
    if scope == "all":
        return part
    return part[part["fake_type"].isin([scope, "real"])]


def choose_threshold(labels, probs) -> float | None:
    """val에서 F1이 최대인 기준값. 동점이면 더 큰 값(가짜 판정을 더 줄이는 쪽)."""
    y, p = _clean(labels, probs)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    best_t = None
    best_f1 = -1.0
    for raw in np.unique(p):
        threshold = float(raw)
        f1 = binary_metrics(y, p, threshold)["f1"]
        score = -1.0 if f1 is None else f1
        if score > best_f1 or (score == best_f1 and (best_t is None or threshold > best_t)):
            best_t = threshold
            best_f1 = score
    return best_t


def _bootstrap(y, p, threshold: float, n_boot: int, seed: int) -> dict:
    empty = {"f1_ci_low": None, "f1_ci_high": None, "auc_ci_low": None, "auc_ci_high": None}
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if n_boot <= 0 or len(y) < 20 or len(np.unique(y)) < 2:
        return empty
    rng = np.random.default_rng(seed)
    f1s, aucs = [], []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys, ps = y[idx], p[idx]
        if len(np.unique(ys)) < 2:
            continue
        stats = binary_metrics(ys, ps, threshold)
        if stats["f1"] is not None:
            f1s.append(stats["f1"])
        if stats["auc"] is not None:
            aucs.append(stats["auc"])

    def ci(values):
        if len(values) < 10:
            return None, None
        lo, hi = np.percentile(values, [2.5, 97.5])
        return float(lo), float(hi)

    f_lo, f_hi = ci(f1s)
    a_lo, a_hi = ci(aucs)
    return {"f1_ci_low": f_lo, "f1_ci_high": f_hi, "auc_ci_low": a_lo, "auc_ci_high": a_hi}


def metric_rows(frame: pd.DataFrame, threshold_of, n_boot: int = 0) -> pd.DataFrame:
    """모델 × 유형 × 후처리 한 행씩. threshold_of(model, run) -> float."""
    rows = []
    if frame.empty:
        return pd.DataFrame(rows)
    known = list(POSTPROCESSES)
    extra = [pp for pp in sorted(frame["postprocess"].dropna().unique()) if pp not in known]
    identities = frame[["model", "run", "model_label"]].drop_duplicates()
    for ident in identities.itertuples(index=False):
        group = frame[(frame["model"] == ident.model) & (frame["run"] == ident.run)]
        threshold = float(threshold_of(ident.model, ident.run))
        for scope in ALL_SCOPES:
            for postprocess in known + extra:
                part = scope_frame(group, scope, postprocess)
                stats = binary_metrics(part["label"], part["prob_fake"], threshold)
                if stats["n"] >= 20 and stats["n_pos"] >= 5 and stats["n_neg"] >= 5:
                    stats.update(_bootstrap(part["label"], part["prob_fake"], threshold, n_boot, seed=0))
                else:
                    stats.update(_bootstrap([], [], threshold, 0, seed=0))
                rows.append(
                    {
                        "model": ident.model,
                        "run": ident.run,
                        "model_label": ident.model_label,
                        "fake_type": scope,
                        "postprocess": postprocess,
                        **stats,
                    }
                )
    return pd.DataFrame(rows)
