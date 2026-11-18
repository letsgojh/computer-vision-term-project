"""프로젝트 경로. Colab에서는 환경변수 CVTP_DATA_ROOT로 드라이브 경로를 지정한다.

드라이브에는 이미지만 둔다. split.csv 같은 CSV는 Git으로 관리하므로 항상 레포 안의 경로를 쓴다.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("CVTP_DATA_ROOT", ROOT / "data"))  # 이미지 (드라이브)

REAL = DATA / "real"
FAKE = DATA / "fake"
MASKS = FAKE / "masks"
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"

SPLITS = ROOT / "data" / "splits"  # CSV (Git)
SPLIT_CSV = SPLITS / "split.csv"

METADATA = ROOT / "metadata"
REAL_METADATA_CSV = METADATA / "real_metadata.csv"
FAKE_MAPPING_CSV = METADATA / "fake_mapping.csv"

OUTPUTS = ROOT / "outputs"
CHECKPOINTS = OUTPUTS / "checkpoints"
PREDICTIONS = OUTPUTS / "predictions"
RESULTS = OUTPUTS / "results"
