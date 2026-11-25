"""2단계 직교 Haar 웨이블릿.

WaRPAD는 저주파를 0으로 두고 복원한 고주파 성분으로 DINOv2 입력을 흔든다.
아래 변환은 짝수 해상도에서 완전 복원되며, 고주파 잔차는
원본에서 2단계 저주파 복원을 뺀 값과 같다.
"""

from __future__ import annotations

import math

import torch

_SQRT2 = math.sqrt(2.0)


def _dwt2(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.shape[-1] % 2 or x.shape[-2] % 2:
        raise ValueError(f"Haar 변환은 짝수 해상도가 필요합니다. 받은 크기: {tuple(x.shape[-2:])}")
    a = x[..., :, 0::2]
    b = x[..., :, 1::2]
    low = (a + b) / _SQRT2
    high = (a - b) / _SQRT2

    a_low = low[..., 0::2, :]
    b_low = low[..., 1::2, :]
    a_high = high[..., 0::2, :]
    b_high = high[..., 1::2, :]
    ll = (a_low + b_low) / _SQRT2
    lh = (a_low - b_low) / _SQRT2
    hl = (a_high + b_high) / _SQRT2
    hh = (a_high - b_high) / _SQRT2
    return ll, lh, hl, hh


def _idwt2(
    ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor
) -> torch.Tensor:
    a_low = (ll + lh) / _SQRT2
    b_low = (ll - lh) / _SQRT2
    a_high = (hl + hh) / _SQRT2
    b_high = (hl - hh) / _SQRT2

    height, width = a_low.shape[-2], a_low.shape[-1]
    low = a_low.new_empty(a_low.shape[:-2] + (height * 2, width))
    high = a_low.new_empty(a_low.shape[:-2] + (height * 2, width))
    low[..., 0::2, :] = a_low
    low[..., 1::2, :] = b_low
    high[..., 0::2, :] = a_high
    high[..., 1::2, :] = b_high

    a = (low + high) / _SQRT2
    b = (low - high) / _SQRT2
    out = a.new_empty(a.shape[:-1] + (width * 2,))
    out[..., 0::2] = a
    out[..., 1::2] = b
    return out


def high_frequency(x: torch.Tensor, levels: int = 2) -> torch.Tensor:
    """저주파 근사를 제거한 고주파 복원. WaRPAD의 IDWT(0, high-pass)에 해당한다."""
    divisor = 2**levels
    if x.shape[-1] % divisor or x.shape[-2] % divisor:
        raise ValueError(
            f"{levels}단계 Haar는 가로·세로가 {divisor}의 배수여야 합니다. 받은 크기: {tuple(x.shape[-2:])}"
        )
    details: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    current = x
    for _ in range(levels):
        ll, lh, hl, hh = _dwt2(current)
        details.append((lh, hl, hh))
        current = ll
    current = torch.zeros_like(current)
    for lh, hl, hh in reversed(details):
        current = _idwt2(current, lh, hl, hh)
    return current
