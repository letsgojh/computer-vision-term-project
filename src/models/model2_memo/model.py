"""모델 2 분류기: EfficientNet 백본 + 로짓 1개."""
from __future__ import annotations

import timm
import torch
from torch import nn


class MemoClassifier(nn.Module):
    """메모 크롭 이미지를 받아 가짜 로짓 [B]를 낸다."""

    def __init__(self, name: str = "efficientnet_b0", pretrained: bool = True,
                 drop_rate: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=1,
                                          global_pool="avg", drop_rate=drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)


def build_model(model_cfg: dict, pretrained: bool | None = None) -> MemoClassifier:
    """설정으로 모델을 만든다. pretrained를 주면 설정값보다 우선."""
    if pretrained is None:
        pretrained = bool(model_cfg.get("pretrained", True))
    return MemoClassifier(model_cfg.get("name", "efficientnet_b0"), pretrained,
                          float(model_cfg.get("drop_rate", 0.2)))
