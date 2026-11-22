import unittest

import torch
from PIL import Image

from src.models.model1_full_image.inference import analyze, crop_grid
from src.models.model1_full_image.model import build_model

CONFIG = {"name": "efficientnet_b0", "pretrained": False, "preprocess": "native_crop", "input_size": 64,
          "mean": [0.5] * 3, "std": [0.5] * 3, "drop_rate": 0.0}


class CropGridTest(unittest.TestCase):
    def test_grid_covers_image(self):
        for width, height in [(683, 512), (512, 512), (64, 64), (1000, 300)]:
            corners = crop_grid(width, height, 256 if min(width, height) >= 256 else 64)
            size = 256 if min(width, height) >= 256 else 64
            self.assertEqual(max(x for x, _ in corners) + size, max(width, size))
            self.assertEqual(max(y for _, y in corners) + size, max(height, size))
            self.assertEqual(min(x for x, _ in corners), 0)


class AnalyzeTest(unittest.TestCase):
    def test_whole_photo_score_and_heatmap(self):
        model = build_model(CONFIG, pretrained=False).eval()
        checkpoint = {"config": {"model": CONFIG}}
        image = Image.new("RGB", (400, 300), (120, 80, 40))
        result = analyze(model, checkpoint, torch.device("cpu"), image, work_short=128)
        self.assertTrue(0 <= result["fake_probability"] <= 1)
        self.assertEqual(result["work_size"], (171, 128))
        self.assertEqual(len(result["crop_probabilities"]), result["crops"])
        self.assertEqual(result["heatmap"].size, result["work_size"])


if __name__ == "__main__":
    unittest.main()
