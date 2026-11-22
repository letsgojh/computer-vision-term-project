import unittest

from src.models.model1_full_image.thresholds import calibrate, threshold_for_fpr, threshold_for_recall, verdict


class ThresholdTest(unittest.TestCase):
    def test_verdict_levels(self):
        self.assertEqual(verdict(0.1, 0.3, 0.8), "pass")
        self.assertEqual(verdict(0.3, 0.3, 0.8), "suspicious")
        self.assertEqual(verdict(0.8, 0.3, 0.8), "ai")
        with self.assertRaises(ValueError):
            verdict(0.5, 0.9, 0.2)

    def test_recall_threshold_keeps_target(self):
        labels = [1] * 100
        scores = [i / 100 for i in range(100)]
        t = threshold_for_recall(labels, scores, 0.95)
        self.assertGreaterEqual(sum(s >= t for s in scores) / 100, 0.95)

    def test_fpr_threshold_bounds_false_positives(self):
        labels = [0] * 100
        scores = [i / 100 for i in range(100)]
        t = threshold_for_fpr(labels, scores, 0.05)
        self.assertLessEqual(sum(s >= t for s in scores) / 100, 0.05)

    def test_calibrate_orders_thresholds(self):
        labels = [0] * 50 + [1] * 50
        scores = [i / 100 for i in range(50)] + [0.3 + i / 100 for i in range(50)]
        values = calibrate(labels, scores, 0.95, 0.01)
        self.assertLessEqual(values["suspicious"], values["ai"])
        self.assertGreaterEqual(values["validation_rates"]["suspicious"]["fake_recall"], 0.95)
        self.assertLessEqual(values["validation_rates"]["ai"]["real_fpr"], 0.01)

    def test_near_perfect_validation_keeps_a_review_band(self):
        # Perfectly separated scores used to collapse suspicious == ai.
        labels = [0] * 100 + [1] * 100
        scores = [i / 1000 for i in range(100)] + [0.9 + i / 1000 for i in range(100)]
        values = calibrate(labels, scores, 0.95, 0.01, 0.05)
        self.assertLess(values["suspicious"], values["ai"])
        self.assertLessEqual(values["validation_rates"]["suspicious"]["real_fpr"], 0.05)
        self.assertEqual(values["validation_rates"]["suspicious"]["fake_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
