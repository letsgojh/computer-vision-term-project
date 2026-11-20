"""역할 간 데이터 규약. docs/data_contract.md와 내용을 같게 유지할 것."""

import re

FAKE_TYPES = ("A", "B", "C")
POSTPROCESSES = ("original", "kakao", "resize", "recapture")
SPLITS = ("train", "val", "test")

# 파일명 규칙
# 메모 없는 사진은 번호 앞에 nomemo_가 붙는다 (이슈 #5). 가짜 B는 원본 이름을 따라가서 자동으로 붙는다.
NOMEMO = "nomemo"
REAL_NAME = re.compile(
    r"^(?P<photographer>[a-z0-9]+)_(?P<object>[a-z0-9]+)_(?:(?P<nomemo>nomemo)_)?(?P<idx>\d{3})\.jpe?g$"
)
FAKE_AB_NAME = re.compile(
    r"^fake(?P<type>[AB])_(?P<source>[a-z0-9]+_[a-z0-9]+_(?:nomemo_)?\d{3})_s(?P<seed>\d+)\.png$"
)
FAKE_C_NAME = re.compile(r"^fakeC_(?P<object>[a-z0-9]+)_(?:(?P<nomemo>nomemo)_)?(?P<idx>\d{3})_s(?P<seed>\d+)\.png$")
# 구글 폼으로 올리면 드라이브가 파일명 뒤에 " - 제출자 이름"을 붙인다 (예: giwon_smartwatch_014 - 차기원.jpg)
_FORM_SUFFIX = re.compile(r" - [^/]*(?=\.[^./]+$)")

# CSV 컬럼
# 진짜 사진 메타데이터는 사람이 쓰지 않고 파일명·EXIF로 자동 생성한다 (src/preprocess/metadata.py).
# 촬영 조건(펜·종이·조명·배경)은 기록하지 않기로 했다 (이슈 #5).
# memo: 메모가 있으면 1, 없으면 0 (파일명의 nomemo_로 판단)
REAL_METADATA_COLUMNS = ["filename", "photographer", "object", "memo", "phone_model", "width", "height", "taken_at"]
OBJECTS_COLUMNS = ["object", "name_ko"]
FAKE_MAPPING_COLUMNS = [
    "fake_filename", "fake_type", "source_filename", "mask_filename", "object",
    "prompt", "negative_prompt", "seed", "model", "created_by", "is_reinforcement",
]
SPLIT_COLUMNS = ["filename", "label", "fake_type", "object", "group_id", "photographer", "split"]
PREDICTION_COLUMNS = ["filename", "postprocess", "prob_fake"]
# 모델 1(Model_A) 학습 입력 형식. split.csv에서 만든다 (src/preprocess/split.py export).
# photographer는 물건 단위 split이라 여러 split에 걸칠 수 있어서, 모델 1이 누수 검사하는 photographer_id 이름을 쓰지 않는다.
MODEL1_COLUMNS = ["image_path", "label", "fake_type", "source_id", "split", "domain", "item_id", "photographer", "postprocess"]


def fake_ab_name(fake_type: str, source_filename: str, seed: int) -> str:
    stem = source_filename.rsplit(".", 1)[0]
    return f"fake{fake_type}_{stem}_s{seed}.png"


def mask_name(fake_filename: str) -> str:
    return fake_filename.rsplit(".", 1)[0] + "_mask.png"


def fake_c_name(obj: str, idx: int, seed: int, memo: bool = True) -> str:
    tag = "" if memo else f"{NOMEMO}_"
    return f"fakeC_{obj}_{tag}{idx:03d}_s{seed}.png"


def real_name(filename: str) -> str:
    """드라이브에 올라간 진짜 사진 이름을 규칙상의 이름으로 바꾼다.

    구글 폼이 붙인 " - 이름"을 떼고 소문자로 바꾼다 (아이폰 확장자 .JPG, 대문자로 쓴 닉네임).
    CSV와 후처리 결과에는 이 이름을 쓰고, 원본 파일을 열 때는 paths.real_files()로 실제 경로를 찾는다.
    """
    return _FORM_SUFFIX.sub("", filename).lower()


def has_memo(filename: str) -> bool:
    """메모가 있는 사진인지 파일명으로 판단한다. 진짜·가짜 A·B·C 모두에 쓸 수 있다."""
    return f"_{NOMEMO}_" not in filename


def group_id(filename: str) -> str:
    """같은 원본에서 나온 진짜·가짜를 같은 split에 묶기 위한 키."""
    m = FAKE_AB_NAME.match(filename)
    if m:
        return m["source"]
    return filename.rsplit(".", 1)[0]
