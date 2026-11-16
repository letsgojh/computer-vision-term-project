"""평가 표가 규약과 노션 양식에 맞게 나오는지 확인한다."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.demo import build_demo
from src.evaluation.load import normalize_prediction
from src.evaluation.metrics import binary_metrics, metric_rows
from src.evaluation.run import run_evaluation


class MetricTests(unittest.TestCase):
    def test_known_confusion(self):
        stats = binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.7, 0.9], 0.5)
        self.assertEqual(stats["tp"], 2)
        self.assertEqual(stats["fp"], 1)
        self.assertEqual(stats["tn"], 1)
        self.assertEqual(stats["fn"], 0)
        self.assertAlmostEqual(stats["accuracy"], 0.75)
        self.assertAlmostEqual(stats["precision"], 2 / 3)
        self.assertAlmostEqual(stats["recall"], 1.0)
        self.assertAlmostEqual(stats["f1"], 0.8)
        self.assertAlmostEqual(stats["specificity"], 0.5)
        self.assertAlmostEqual(stats["auc"], 1.0)

    def test_precision_undefined_when_never_predicts_fake(self):
        stats = binary_metrics([0, 1, 1], [0.1, 0.2, 0.3], 0.5)
        self.assertIsNone(stats["precision"])
        self.assertEqual(stats["recall"], 0.0)
        self.assertIsNone(stats["f1"])

    def test_type_slice_drops_other_fakes(self):
        frame = pd.DataFrame(
            [
                ["model1", "default", "모델 1", "a.jpg", "original", 0, "real", 0.1],
                ["model1", "default", "모델 1", "b.png", "original", 1, "A", 0.9],
                ["model1", "default", "모델 1", "c.png", "original", 1, "B", 0.1],
            ],
            columns=["model", "run", "model_label", "filename", "postprocess", "label", "fake_type", "prob_fake"],
        )
        table = metric_rows(frame, lambda *_: 0.5, n_boot=0)
        row = table[(table["fake_type"] == "A") & (table["postprocess"] == "original")].iloc[0]
        self.assertEqual(int(row["n_pos"]), 1)
        self.assertEqual(int(row["n_neg"]), 1)
        self.assertEqual(int(row["n"]), 2)


class LoadTests(unittest.TestCase):
    def test_model_branch_columns_and_percent_scores(self):
        frame = pd.DataFrame(
            {
                "image_path": ["data/processed/original/jaehwan_mouse_001.jpg"],
                "fake_probability": [82],
                "domain": ["kakao"],
                "label": [0],
            }
        )
        out, notes = normalize_prediction(frame, Path("outputs/predictions/model1_baseline.csv"))
        self.assertEqual(out.loc[0, "filename"], "jaehwan_mouse_001.jpg")
        self.assertAlmostEqual(out.loc[0, "prob_fake"], 0.82)
        self.assertEqual(out.loc[0, "postprocess"], "kakao")
        self.assertEqual(out.loc[0, "model"], "model1")
        self.assertTrue(any("100" in note for note in notes))

    def test_unknown_domain_stays_original(self):
        frame = pd.DataFrame(
            {
                "image_path": ["a.jpg", "b.jpg"],
                "fake_probability": [0.2, 0.8],
                "domain": ["tigas", "diquid"],
            }
        )
        out, notes = normalize_prediction(frame, Path("model2/test_predictions.csv"))
        self.assertTrue((out["postprocess"] == "original").all())
        self.assertTrue(any("original" in note for note in notes))


class PipelineTests(unittest.TestCase):
    def test_demo_writes_notion_table_and_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            built = build_demo(root / "in")
            preview = pd.read_csv(built["predictions"] / "model1_baseline.csv")
            top_fp = preview[preview["filename"].str.endswith(".jpg") & (preview["postprocess"] == "original")]
            top_name = top_fp.sort_values("prob_fake", ascending=False).iloc[0]["filename"]
            image = root / "processed" / "original" / top_name
            image.parent.mkdir(parents=True)
            image.write_bytes(b"not-an-image")
            result = run_evaluation(
                files := sorted((built["predictions"]).glob("*.csv")),
                built["split"],
                built["real_metadata"],
                built["fake_mapping"],
                root / "processed",
                root / "results",
                n_boot=0,
                demo=True,
            )
            report = (root / "results" / "report.md").read_text(encoding="utf-8")
            bundle = (root / "results" / "tables" / "bundle.md").read_text(encoding="utf-8")
            self.assertIn("모델 1 – A", bundle)
            self.assertIn("모델 2 – C", bundle)
            self.assertIn("원본", bundle)
            self.assertIn("카톡 압축", bundle)
            self.assertIn("리사이즈", bundle)
            self.assertIn("재촬영", bundle)
            self.assertIn("형식 확인용", report)
            feedback = result["feedback"]
            self.assertIn("train", feedback)
            self.assertIn("모델 2", feedback)
            self.assertIn("유형 B", feedback)
            self.assertIn("iPhone15", feedback)
            metrics = result["metrics"]
            m1a = _recall(metrics, "model1", "A", "original")
            m1b = _recall(metrics, "model1", "B", "original")
            m2a = _recall(metrics, "model2", "A", "original")
            m2b = _recall(metrics, "model2", "B", "original")
            self.assertAlmostEqual(m1a, 0.5)
            self.assertAlmostEqual(m1b, 1.0)
            self.assertGreater(m2a, m1a)
            self.assertLess(m2b, 0.2)
            slices = pd.read_csv(root / "results" / "errors" / "metadata_slices.csv")
            hot = slices[(slices["concentrated"]) & (slices["error"] == "FP") & (slices["model"] == "model1")]
            self.assertTrue(((hot["field"] == "pen") & (hot["value"] == "highlighter")).any())
            copied = list((root / "results" / "cases").rglob(f"*{top_name}"))
            self.assertTrue(copied)
            self.assertTrue((root / "results" / "tables" / "f1_val_threshold.md").is_file())
            self.assertTrue(files)

    def test_leakage_and_reinforcement_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split = root / "split.csv"
            pd.DataFrame(
                [
                    ["jaehwan_mouse_001.jpg", 0, "real", "mouse", "jaehwan_mouse_001", "jaehwan", "test"],
                    ["fakeA_jaehwan_mouse_001_s1.png", 1, "A", "mouse", "jaehwan_mouse_001", "jaehwan", "train"],
                    ["fakeA_jaehwan_mouse_002_s1.png", 1, "A", "mouse", "shared", "jaehwan", "test"],
                    ["jaehwan_mouse_002.jpg", 0, "real", "mouse", "shared", "jaehwan", "val"],
                ],
                columns=["filename", "label", "fake_type", "object", "group_id", "photographer", "split"],
            ).to_csv(split, index=False)
            mapping = root / "fake_mapping.csv"
            pd.DataFrame(
                [
                    {
                        "fake_filename": "fakeA_jaehwan_mouse_002_s1.png",
                        "fake_type": "A",
                        "source_filename": "jaehwan_mouse_002.jpg",
                        "is_reinforcement": 1,
                    }
                ]
            ).to_csv(mapping, index=False)
            pred = root / "model1_baseline.csv"
            pd.DataFrame(
                {
                    "filename": ["jaehwan_mouse_001.jpg", "fakeA_jaehwan_mouse_002_s1.png"],
                    "postprocess": ["original", "original"],
                    "prob_fake": [0.2, 0.8],
                }
            ).to_csv(pred, index=False)
            result = run_evaluation(
                [pred],
                split,
                None,
                mapping,
                None,
                root / "out",
                n_boot=0,
            )
            text = "\n".join(result["notes"])
            self.assertIn("group_id", text)
            self.assertIn("is_reinforcement", text)


def _recall(metrics, model, scope, postprocess) -> float:
    hit = metrics[
        (metrics["model"] == model)
        & (metrics["fake_type"] == scope)
        & (metrics["postprocess"] == postprocess)
    ]
    return float(hit.iloc[0]["recall"])


if __name__ == "__main__":
    unittest.main()
