import unittest

import torch
from torch.utils.data import WeightedRandomSampler


class WeightedSamplerTest(unittest.TestCase):
    def test_source_shares_follow_weights(self):
        # 1000 rows of a big source and 10 of a small one, weighted to an equal share each.
        weights = torch.tensor([0.5 / 1000] * 1000 + [0.5 / 10] * 10, dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=20000, replacement=True, generator=torch.Generator().manual_seed(0))
        small_share = sum(i >= 1000 for i in sampler) / 20000
        self.assertAlmostEqual(small_share, 0.5, delta=0.03)


if __name__ == "__main__":
    unittest.main()
