"""모델 2 지표: 이진 분류 지표, 임계값 선택, 비율 표."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


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
    """고유 확률값 중 F1이 최대인 임계값을 고른다. 동점이면 큰 값."""
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs, dtype=float)
    if len(set(y.tolist())) < 2:
        raise ValueError("임계값을 고르려면 진짜와 가짜가 모두 필요합니다.")
    best_t, best_f1 = None, -1.0
    for t in np.unique(p):
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 >= best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def rate_table(pred: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """그룹별 개수, 가짜 예측 비율, 평균 가짜 확률."""
    g = pred.groupby(by, dropna=False)
    out = g.agg(n=("prediction", "size"), pred_fake_rate=("prediction", "mean"),
                mean_prob=("fake_probability", "mean"))
    return out.reset_index()
