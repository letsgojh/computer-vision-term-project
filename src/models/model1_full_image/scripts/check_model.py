"""Check EfficientNet-B0 forward pass independently of training data."""

import argparse
import torch
from src.models.model1_full_image.utils import select_device
from src.models.model1_full_image.model import BinaryImageClassifier


def main():
    """Run a full-resolution batch and verify the binary logit contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", action="store_true", help="Download ImageNet weights; errors are not hidden")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    device = select_device(args.device)
    model = BinaryImageClassifier(pretrained=args.pretrained).to(device).eval()
    with torch.inference_mode():
        logits = model(torch.randn(2, 3, 384, 384, device=device))
    assert logits.shape == (2,) and torch.isfinite(logits).all()
    print(f"Forward OK: device={device}, input=(2,3,384,384), logits={tuple(logits.shape)}, pretrained={args.pretrained}")


if __name__ == "__main__":
    main()
