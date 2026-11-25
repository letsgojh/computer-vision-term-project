"""점수 식, 실제 사진 기준 백분위, 판정.

점수가 클수록 실제 사진 쪽에 가깝다.
AeroBlade의 재구성 오차 f_AE, RDD의 S_image / S_latent, WaRPAD 코사인 유사도가 모두 이 방향이다.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# 높을수록 실제 사진에 가까운 점수 이름.
REALNESS_KEYS = ("s_image", "s_latent", "warpad", "f_ae")

VERDICT_AI = "AI 의심"
VERDICT_UNSURE = "애매"
VERDICT_REAL = "실제에 가까움"
VERDICT_NO_REFERENCE = "기준 없음"


def simage(f_x: float, f_lowpass: float, f_rotated: float, f_both: float, lambda_r: float) -> float:
    """RDD 이미지 디바이어싱 점수.

    S_image(x) = f(x) - f(F(x)) - λ f(R(x)) + λ f(F(R(x)))
    F는 가우시안 저역통과, R은 90도 회전, f는 VAE 재구성의 LPIPS.
    """
    return f_x - f_lowpass - lambda_r * f_rotated + lambda_r * f_both


def rdd_score(s_image_value: float, s_latent_value: float) -> float:
    """S_RDD = S_image × S_latent²."""
    return s_image_value * (s_latent_value**2)


def percentile_rank(value: float, reference: list[float] | np.ndarray) -> float:
    """기준 실제 사진 대비 백분위(0~100). 클수록 실제 사진 분포 안쪽에 있다.

    같은 점수는 중간에 둔다. 기준 점수가 모두 같을 때, 그 값과 같은 사진을
    분포의 맨 아래로 밀지 않기 위해서다.
    """
    if value is None or not math.isfinite(float(value)):
        return float("nan")
    ref = np.asarray(reference, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return float("nan")
    value = float(value)
    below = float(np.mean(ref < value))
    tied = float(np.mean(ref == value))
    return (below + 0.5 * tied) * 100.0


def verdict_from_percentile(percentile: float, low: float = 5.0, mid: float = 20.0) -> str:
    if not math.isfinite(percentile):
        return VERDICT_NO_REFERENCE
    if percentile < low:
        return VERDICT_AI
    if percentile < mid:
        return VERDICT_UNSURE
    return VERDICT_REAL


def fuse_percentiles(parts: dict[str, float]) -> float:
    values = [v for v in parts.values() if v is not None and math.isfinite(v)]
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def reference_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def describe_distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }
