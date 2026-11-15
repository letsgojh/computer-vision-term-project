"""model 2 테스트 공용 fixture."""
from pathlib import Path

import pytest

from src.models.model2_memo.scripts.make_smoke_data import make_smoke_data


@pytest.fixture(scope="session")
def smoke_data(tmp_path_factory) -> Path:
    """작은 합성 데이터셋을 만들고 CSV 경로를 돌려준다."""
    out = tmp_path_factory.mktemp("smoke_root") / "smoke"
    return make_smoke_data(out, items=6, per_item=2, c_per_item=1, size=(160, 120), seed=0)
