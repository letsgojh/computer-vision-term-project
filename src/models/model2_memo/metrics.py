"""모델 2 지표: 이진 분류 지표, 임계값 선택, 비율 표."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def binary_metrics(labels, probs, threshold: float) -> dict:
    """prob >= threshold를 가짜(1)로 예측해 지표를 계산한다."""
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    auc = float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else None
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": auc,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "threshold": float(threshold),
    }


def choose_threshold(labels, probs) -> float:
    """F1이 최대인 임계값을 고른다. 후보는 PR 곡선 임계값과 0.5 (모델 1과 같은 규칙).

    최대 F1(오차 1e-12) 동점이면 0.5에 가까운 값, 그다음 작은 값을 고른다.
    """
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs, dtype=float)
    if len(set(y.tolist())) < 2:
        raise ValueError("임계값을 고르려면 진짜와 가짜가 모두 필요합니다.")
    candidates = np.append(precision_recall_curve(y, p)[2], 0.5)
    f1 = np.array([f1_score(y, (p >= t).astype(int), zero_division=0) for t in candidates])
    best = candidates[f1 >= f1.max() - 1e-12]
    return float(sorted(best, key=lambda t: (abs(t - 0.5), t))[0])


def rate_table(pred: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """그룹별 개수, 가짜 예측 비율, 평균 가짜 확률."""
    g = pred.groupby(by, dropna=False)
    out = g.agg(n=("prediction", "size"), pred_fake_rate=("prediction", "mean"),
                mean_prob=("fake_probability", "mean"))
    return out.reset_index()
