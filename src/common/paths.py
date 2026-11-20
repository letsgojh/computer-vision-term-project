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
OBJECTS_CSV = METADATA / "objects.csv"
FAKE_MAPPING_CSV = METADATA / "fake_mapping.csv"

def real_files(data_root: Path = DATA) -> list[Path]:
    """data/real 아래 모든 파일. 드라이브에서는 사람별 하위 폴더에 있으므로 하위 폴더까지 찾는다.

    숨김 파일·폴더(.DS_Store 등)는 뺀다. 규칙상 이름은 schema.real_name(p.name)으로 얻는다.
    """
    real = data_root / "real"
    if not real.is_dir():
        return []
    return sorted(
        p for p in real.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(real).parts)
    )


OUTPUTS = ROOT / "outputs"
CHECKPOINTS = OUTPUTS / "checkpoints"
PREDICTIONS = OUTPUTS / "predictions"
RESULTS = OUTPUTS / "results"
