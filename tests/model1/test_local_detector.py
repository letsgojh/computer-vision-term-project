import unittest

import numpy as np
import torch
from PIL import Image

from src.models.model1_full_image.local_detector import TILE, positions, scan, tile_heatmap


class MeanBrightness(torch.nn.Module):
    """Stand-in model: logit = mean pixel value of the (normalised) tile."""

    def forward(self, x):
        return x.mean(dim=(1, 2, 3))


class LocalDetectorTest(unittest.TestCase):
    def test_positions_cover_edges_without_overlap_gaps(self):
        self.assertEqual(positions(200), [0])
        self.assertEqual(positions(600), [0, 256, 344])
        for length in [256, 511, 512, 1000, 4032]:
            starts = positions(length)
            self.assertEqual(starts[0], 0)
            self.assertEqual(starts[-1] + TILE, length)
            self.assertTrue(all(b - a <= TILE for a, b in zip(starts, starts[1:])))

    def test_scan_finds_the_bright_tile_and_heatmap_matches_photo(self):
        pixels = np.zeros((600, 800, 3), np.uint8)
        pixels[344:600, 512:768] = 255  # exactly the bottom-edge-aligned tile at (512, 344)
        image = Image.fromarray(pixels)
        checkpoint = {"config": {"model": {"mean": [0, 0, 0], "std": [1, 1, 1]}}}
        logits, corners, scanned = scan(MeanBrightness(), checkpoint, torch.device("cpu"), image)
        self.assertEqual(len(logits), len(positions(800)) * len(positions(600)))
        self.assertEqual(scanned.size, (800, 600))
        self.assertEqual(corners[int(logits.argmax())], (512, 344))
        heat = tile_heatmap(scanned, logits, corners, low=0.1, high=0.9)
        self.assertEqual(heat.size, (800, 600))

    def test_small_image_is_padded_up_to_one_tile(self):
        checkpoint = {"config": {"model": {"mean": [0, 0, 0], "std": [1, 1, 1]}}}
        logits, corners, scanned = scan(MeanBrightness(), checkpoint, torch.device("cpu"), Image.new("RGB", (100, 300)))
        self.assertEqual(min(scanned.size), TILE)
        self.assertEqual(len(logits), len(corners))


if __name__ == "__main__":
    unittest.main()
