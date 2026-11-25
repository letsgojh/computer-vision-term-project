import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from detector.haar import high_frequency
from detector.images import list_images, native_center_square, open_rgb
from detector.rdd import combine_image_scores, reconstruction_views
from detector.scores import (
    fuse_percentiles,
    percentile_rank,
    rdd_score,
    simage,
    verdict_from_percentile,
)


class ScoreFormulaTest(unittest.TestCase):
    def test_simage_matches_rdd_equation(self):
        value = simage(f_x=0.50, f_lowpass=0.40, f_rotated=0.80, f_both=0.70, lambda_r=0.5)
        self.assertAlmostEqual(value, 0.05)

    def test_rdd_squares_latent_score(self):
        self.assertAlmostEqual(rdd_score(0.2, -0.5), 0.05)
        self.assertAlmostEqual(rdd_score(0.2, 0.5), 0.05)

    def test_percentile_and_verdict(self):
        reference = [0.1, 0.2, 0.3, 0.4]
        self.assertAlmostEqual(percentile_rank(0.05, reference), 0.0)
        self.assertAlmostEqual(percentile_rank(0.35, reference), 75.0)
        self.assertAlmostEqual(percentile_rank(0.2, [0.2, 0.2, 0.2, 0.2]), 50.0)
        self.assertEqual(verdict_from_percentile(4.9), "AI 의심")
        self.assertEqual(verdict_from_percentile(12), "애매")
        self.assertEqual(verdict_from_percentile(40), "실제에 가까움")
        self.assertAlmostEqual(fuse_percentiles({"a": 10.0, "b": 30.0}), 20.0)

    def test_combine_uses_min_error_and_mean_simage(self):
        combined = combine_image_scores(
            [{"f_ae": 0.2, "s_image": 0.1}, {"f_ae": 0.4, "s_image": 0.3}],
            s_latent_value=0.5,
        )
        self.assertAlmostEqual(combined["f_ae"], 0.2)
        self.assertAlmostEqual(combined["s_image"], 0.2)
        self.assertAlmostEqual(combined["s_rdd"], 0.05)


class HaarTest(unittest.TestCase):
    def test_high_frequency_is_perfect_reconstruction_residual(self):
        torch.manual_seed(0)
        image = torch.randn(2, 3, 32, 32)
        residual = high_frequency(image, levels=2)
        restored_low = image - residual
        # 고주파를 다시 더하면 원본이 된다.
        self.assertTrue(torch.allclose(restored_low + residual, image, atol=1e-5))
        # 2단계 저주파만 남긴 영상은 고주파 잔차와 거의 직교에 가깝게 에너지가 나뉜다.
        self.assertGreater(float(residual.pow(2).mean()), 0.0)
        self.assertTrue(torch.allclose(high_frequency(restored_low, levels=2), torch.zeros_like(image), atol=1e-4))


class ImagePrepTest(unittest.TestCase):
    def test_large_photo_is_center_cropped_without_downscale(self):
        image = Image.new("RGB", (1000, 800), (10, 20, 30))
        # 중앙 픽셀만 다르게 칠해서 크롭이 중앙인지 확인한다.
        image.putpixel((500, 400), (255, 0, 0))
        cropped = native_center_square(image, 512)
        self.assertEqual(cropped.size, (512, 512))
        self.assertEqual(cropped.getpixel((256, 256)), (255, 0, 0))

    def test_views_cover_four_rdd_transforms(self):
        image = torch.zeros(1, 3, 16, 16)
        image[:, :, :8, :] = 1
        views = reconstruction_views(image, kernel_size=3, sigma=0.8)
        self.assertEqual(tuple(views.shape), (4, 3, 16, 16))
        self.assertTrue(torch.equal(views[0], image[0]))

    def test_list_images_skips_text(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new("RGB", (8, 8), "white").save(root / "a.jpg")
            (root / "note.txt").write_text("skip", encoding="utf-8")
            found = list_images(root)
            self.assertEqual([path.name for path in found], ["a.jpg"])
            opened = open_rgb(root / "a.jpg")
            self.assertEqual(opened.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
