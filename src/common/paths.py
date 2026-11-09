"""프로젝트 경로. Colab에서는 환경변수 CVTP_DATA_ROOT로 드라이브 경로를 지정한다."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("CVTP_DATA_ROOT", ROOT / "data"))

REAL = DATA / "real"
FAKE = DATA / "fake"
MASKS = FAKE / "masks"
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"
SPLIT_CSV = DATA / "splits" / "split.csv"

METADATA = ROOT / "metadata"
REAL_METADATA_CSV = METADATA / "real_metadata.csv"
FAKE_MAPPING_CSV = METADATA / "fake_mapping.csv"

OUTPUTS = ROOT / "outputs"
CHECKPOINTS = OUTPUTS / "checkpoints"
PREDICTIONS = OUTPUTS / "predictions"
RESULTS = OUTPUTS / "results"
