import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models.model1_full_image.inference import analyze
from src.models.model1_full_image.model import build_model
from src.models.model1_full_image.predict import predict

CONFIG = {"name": "efficientnet_b0", "pretrained": False, "preprocess": "native_crop", "input_size": 64,
          "mean": [0.5] * 3, "std": [0.5] * 3, "drop_rate": 0.0}


class PredictNativeCropTest(unittest.TestCase):
    def test_explain_heatmap_matches_whole_photo(self):
        torch.manual_seed(0)
        model = build_model(CONFIG, pretrained=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            image_path = tmp / "photo.png"
            pixels = np.random.default_rng(0).integers(0, 256, (300, 500, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(image_path)
            checkpoint = tmp / "native.pth"
            torch.save({"model_state": model.state_dict(), "config": {"model": CONFIG}, "threshold": 0.5,
                        "is_smoke": False, "epoch": 1, "threshold_source": "fixed_config",
                        "protected_identities": {}, "optimizer_updates": 1}, checkpoint)
            result = predict(image_path, checkpoint, "cpu", explain=True, explain_dir=tmp / "out")
            # Same whole-photo score as the demo path.
            expected = analyze(model.eval(), {"config": {"model": CONFIG}}, torch.device("cpu"),
                               Image.open(image_path), with_heatmap=False)["fake_probability"]
            self.assertAlmostEqual(result["fake_probability"], expected, places=5)
            report = json.loads((tmp / "out" / "explanation.json").read_text(encoding="utf-8"))
            self.assertEqual(report["geometry"]["mode"], "native_crop_tiles")
            self.assertIsNone(report["letterbox"])
            heatmap = np.load(tmp / "out" / "heatmap_values.npy")
            self.assertEqual(heatmap.shape, (300, 500))  # aligned to the full original, not a center crop
            self.assertTrue(np.isfinite(heatmap).all() and heatmap.min() >= 0 and heatmap.max() <= 1)


if __name__ == "__main__":
    unittest.main()
