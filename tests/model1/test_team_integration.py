"""Check relocated roots and the team's object-level split contract."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from src.models.model1_full_image.dataset import read_metadata
from src.models.model1_full_image.scripts.download_tigas import PROJECT_ROOT
from src.models.model1_full_image.utils import load_config


class TeamIntegrationTests(unittest.TestCase):
    def test_relocated_config_resolves_team_root(self):
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(PROJECT_ROOT, root)
        config = load_config(root / "configs/model1_team.yaml")
        self.assertEqual(Path(config["data"]["root"]), root)
        self.assertEqual(Path(config["data"]["metadata"]), root / "data/splits/model1_metadata.csv")
        self.assertEqual(Path(config["training"]["output_dir"]), root / "outputs/model1")

    def test_team_photographer_metadata_and_item_leakage(self):
        config = load_config(PROJECT_ROOT / "configs/model1_team.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "data/processed/original"
            processed.mkdir(parents=True)
            rows = []
            for index, split in enumerate(["train", "val", "test"]):
                path = processed / f"{index}.jpg"
                Image.new("RGB", (32, 32)).save(path)
                rows.append(dict(image_path=path.relative_to(root).as_posix(), label=0,
                                 fake_type="real", source_id=f"s{index}", split=split,
                                 domain="trade", item_id=f"item{index}", photographer="same_person"))
            csv = root / "metadata.csv"
            pd.DataFrame(rows).to_csv(csv, index=False)
            result = read_metadata(csv, root, config["data"]["leakage_columns"])
            self.assertEqual(len(result), 3)
            self.assertEqual(result.photographer.tolist(), ["same_person"] * 3)
            rows[1]["item_id"] = rows[0]["item_id"]
            pd.DataFrame(rows).to_csv(csv, index=False)
            with self.assertRaisesRegex(ValueError, "item_id"):
                read_metadata(csv, root, config["data"]["leakage_columns"])
