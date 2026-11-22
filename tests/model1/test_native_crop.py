import random
import unittest

import torch
from PIL import Image

from src.models.model1_full_image.dataset import NativeCropTransform, build_transform

MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


def noise(width, height, seed=0):
    generator = torch.Generator().manual_seed(seed)
    pixels = torch.randint(0, 256, (height, width, 3), generator=generator, dtype=torch.uint8)
    return Image.fromarray(pixels.numpy())


class NativeCropTest(unittest.TestCase):
    def test_output_shape_for_any_size(self):
        for size in [(640, 480), (480, 640), (1024, 1024), (270, 270), (100, 300)]:
            for train in [False, True]:
                out = NativeCropTransform(256, MEAN, STD, train)(noise(*size))
                self.assertEqual(tuple(out.shape), (3, 256, 256))

    def test_eval_is_deterministic_center_crop_without_resampling(self):
        image = noise(640, 480)
        out = NativeCropTransform(256, MEAN, STD)(image)
        expected = NativeCropTransform(256, MEAN, STD).tensor(image.crop((192, 112, 448, 368)))
        self.assertTrue(torch.equal(out, expected))

    def test_non_square_input_gets_no_padding(self):
        # Letterbox would fill borders with the mean colour; native crop must use only image pixels.
        image = Image.new("RGB", (640, 480), (255, 0, 0))
        out = NativeCropTransform(256, MEAN, STD)(image)
        self.assertTrue(torch.allclose(out[0], torch.ones(256, 256)))

    def test_eval_resize_short(self):
        transform = NativeCropTransform(256, MEAN, STD, eval_resize_short=512)
        self.assertEqual(tuple(transform(noise(1024, 1024)).shape), (3, 256, 256))

    def test_train_augmentation_varies(self):
        random.seed(0)
        transform = NativeCropTransform(256, MEAN, STD, train=True)
        image = noise(640, 480)
        self.assertFalse(torch.equal(transform(image), transform(image)))

    def test_build_transform_selects_mode(self):
        config = {"input_size": 256, "mean": MEAN, "std": STD}
        self.assertNotIsInstance(build_transform(config), NativeCropTransform)
        self.assertIsInstance(build_transform({**config, "preprocess": "native_crop"}), NativeCropTransform)
        with self.assertRaises(ValueError):
            build_transform({**config, "preprocess": "bogus"})


if __name__ == "__main__":
    unittest.main()
