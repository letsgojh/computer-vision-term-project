"""Device routing and real CPU/MPS training, serialization and explanation checks."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.model1_full_image.explain import generate_gradcam
from src.models.model1_full_image.model import BinaryImageClassifier
from src.models.model1_full_image.train import run_epoch
from src.models.model1_full_image.utils import select_device


class DeviceSelectionTests(unittest.TestCase):
    def test_auto_prefers_cuda_then_mps_then_cpu(self):
        for cuda, mps, expected in [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]:
            with self.subTest(expected=expected), patch("torch.cuda.is_available", return_value=cuda), patch("torch.backends.mps.is_available", return_value=mps), patch("torch.cuda.get_device_name", return_value="test GPU"):
                self.assertEqual(select_device().type, expected)
                self.assertEqual(select_device("cpu").type, "cpu")

    def test_explicit_unavailable_accelerator_raises(self):
        with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=False):
            for name in ["cuda", "mps"]:
                with self.assertRaisesRegex(RuntimeError, "unavailable"):
                    select_device(name)
        with self.assertRaises(ValueError):
            select_device("unknown")


class BackendExecutionTests(unittest.TestCase):
    def check_backend(self, name):
        device = torch.device(name)
        model = BinaryImageClassifier(pretrained=False).to(device)
        before = next(model.parameters()).detach().cpu().clone()
        rows = [{"image": torch.randn(3, 64, 64), "label": torch.tensor(float(i % 2))} for i in range(4)]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss, labels, scores, updates = run_epoch(model, DataLoader(rows, batch_size=2), nn.BCEWithLogitsLoss(), device, optimizer)
        self.assertEqual(updates, 2)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(len(labels), 4)
        self.assertTrue(all(0 <= value <= 1 for value in scores))
        self.assertFalse(torch.equal(before, next(model.parameters()).detach().cpu()))
        model.eval()
        image = rows[0]["image"].unsqueeze(0).to(device)
        with torch.inference_mode():
            expected = model(image).cpu()
        explanation = generate_gradcam(model, image)
        self.assertAlmostEqual(explanation["fake_logit"], expected.item(), places=4)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights.pth"
            torch.save(model.state_dict(), path)
            restored = BinaryImageClassifier(pretrained=False).eval()
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            with torch.inference_mode():
                actual = restored(image.cpu())
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)

    def test_cpu_float32_training_and_checkpoint(self):
        self.check_backend("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "Apple MPS hardware unavailable")
    def test_mps_training_gradcam_and_cpu_checkpoint(self):
        self.check_backend("mps")
