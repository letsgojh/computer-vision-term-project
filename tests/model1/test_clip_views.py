import unittest

from PIL import Image

from src.models.model1_full_image.clip_detector import SIZE, view_layout


class ClipViewLayoutTest(unittest.TestCase):
    def test_three_crops_cover_the_whole_long_side(self):
        for size in [(3024, 4032), (5712, 4284), (640, 480), (1024, 1024)]:
            small, offsets = view_layout(Image.new("RGB", size))
            self.assertEqual(min(small.size), SIZE)
            (cx, cy), start, end = offsets
            long_axis = 0 if small.width >= small.height else 1
            self.assertEqual(start[long_axis], 0)
            self.assertEqual(end[long_axis] + SIZE, small.size[long_axis])
            self.assertEqual((cx, cy)[long_axis], (small.size[long_axis] - SIZE) // 2)
            self.assertTrue(all(o[1 - long_axis] == 0 for o in offsets))


if __name__ == "__main__":
    unittest.main()
