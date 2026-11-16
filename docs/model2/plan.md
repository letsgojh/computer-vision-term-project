# 모델 2 (메모 영역 중심) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 메모 영역을 자동으로 찾아 고해상도로 잘라 진짜/가짜를 분류하는 모델 2 파이프라인(위치 찾기 → 학습 → 평가 → 모델 1 비교)을 만들고, 합성 데이터로 끝까지 도는 것을 확인한다.

**Architecture:** `locate.py`가 zero-shot 탐지기(Grounding DINO)로 이미지마다 메모 박스를 찾아 CSV 캐시에 저장한다. `dataset.py`가 그 박스로 크롭해 EfficientNet-B0 이진 분류기(`model.py`)에 넣는다. 학습·평가 방식과 입출력 CSV 형식은 모델 1과 같게 맞추고, `compare.py`가 두 모델의 예측을 A/B/C × 후처리 표로 비교한다.

**Tech Stack:** Python 3.11, PyTorch 2.11 (cu128), timm, transformers (Grounding DINO), pandas, Pillow, scikit-learn, PyYAML, pytest

**Spec:** `docs/model2/design.md` (실행자는 이 계획과 설계 문서를 함께 읽는다)

## Global Constraints

- 실행 환경: `PY=~/.venvs/cvtp-model2/bin/python` (Python 3.11, `torch==2.11.0`, `torchvision==0.26.0` cu128 설치됨). 모든 명령은 레포 루트에서 실행한다.
- 테스트: `$PY -m pytest tests/model2 -q`. **네트워크·GPU 없이** 통과해야 한다. 테스트에서 모델은 `pretrained=False`로 만든다. 실제 탐지기 테스트는 `MODEL2_RUN_DETECTOR=1`일 때만 실행한다.
- 수정 금지: 루트 `requirements.txt`, `.gitignore`, `src/common/`, `docs/data_contract.md`, `src/models/model1_full_image/`, `src/preprocess/`, `src/generation/`, `src/evaluation/`. 모델 1 코드를 import하지 않는다.
- 새 파일은 다음 위치에만 만든다: `src/models/model2_memo/`, `configs/model2_*.yaml`, `tests/__init__.py`, `tests/model2/`, `metadata/memo_boxes_manual.csv`, `docs/model2/`.
- 값 규칙: `label` 진짜 `0`, 가짜 `1`. `fake_type` ∈ {`real`, `A`, `B`, `C`, `unknown`}이고 `real` ↔ 0, 나머지 ↔ 1. `split` ∈ {`train`, `val`, `test`}. `postprocess` ∈ {`original`, `kakao`, `resize`, `recapture`}.
- 박스는 `ImageOps.exif_transpose` 적용 후 이미지 기준으로 0~1 정규화한 `(x0, y0, x1, y1)`이며, `x0 < x1`, `y0 < y1`이다.
- 입력 CSV 컬럼(PR #2 규약 3-5): `image_path, label, fake_type, source_id, split, domain, item_id, photographer, postprocess`. 필수는 앞의 5개다.
- 기본값(설계 문서): `exclude_from_training: [B]`, `train_postprocess: [original]`, `manual_splits: [train]`, `crop.margin 0.15`, `jitter_shift 0.05`, `jitter_scale [0.9, 1.1]`, `input_size 384`, `efficientnet_b0`, `drop_rate 0.2`, AdamW `lr 1e-4` `wd 0.01`, epochs 20, patience 5, batch 16, 경고 기준 10%p.
- 커밋: `<type>: <요약>` (type은 `feat`/`fix`/`data`/`docs`/`chore`), 끝에 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` 줄을 넣는다. 작업마다 로컬 커밋만 하고 **push하지 않는다**. 이미지·체크포인트·`outputs/`는 커밋하지 않는다.
- docstring·주석은 짧은 한국어로 쓴다 (`src/common/`과 같은 스타일).

## Review Focus

1. **박스 캐시에 없는 이미지:** 평가용 CSV를 새로 export하고 `locate`를 안 돌린 경우다. 데이터셋이 개수, 앞의 경로 3개, `locate --metadata` 안내를 담은 `ValueError`를 내야 한다 → Task 3 `test_dataset_missing_boxes_raises`
2. **아주 작거나 가장자리에 붙은 박스:** 예를 들어 `resize` 이미지에서 폭 0.001짜리 박스다. 크롭이 최소 1×1 px이 되고 `[3, S, S]` 텐서가 나와야 한다 → Task 1 `test_to_pixels_min_one_pixel`, Task 3 `test_transform_tiny_edge_box`
3. **B 제외 + original 필터 후 val에 한 클래스만 남는 경우:** 학습을 시작하기 전에 "both"가 들어간 `ValueError`가 나야 한다 → Task 5 `test_train_requires_both_classes_in_val`
4. **모델 1 예측 CSV에 `postprocess` 컬럼이 없거나 두 파일의 행이 다른 경우:** `image_path`로 후처리 종류를 추론하고, 공통 행만 비교하고, 빠진 개수를 보고해야 한다 → Task 7 `test_compare_infers_postprocess_and_counts_missing`
5. **RGB가 아닌 이미지(흑백 `L`, `RGBA`):** 크롭 전에 RGB로 바꿔야 한다 → Task 3 `test_transform_handles_non_rgb`

---

### Task 1: 환경 파일, 공용 유틸, 박스 유틸

**Files:**
- Create: `src/models/model2_memo/requirements.txt`
- Create: `src/models/model2_memo/utils.py`
- Create: `src/models/model2_memo/boxes.py`
- Create: `tests/__init__.py` (빈 파일), `tests/model2/__init__.py` (빈 파일)
- Test: `tests/model2/test_boxes.py`, `tests/model2/test_utils.py`

**Interfaces:**
- Produces (`utils.py`):
  - `load_config(path: str | Path) -> dict`: YAML을 읽는다. 파일이 없으면 `FileNotFoundError`
  - `select_device(name: str = "auto") -> torch.device`: `auto`는 CUDA → MPS → CPU 순서로 고른다. `cuda`/`mps`를 지정했는데 쓸 수 없으면 `RuntimeError`
  - `seed_everything(seed: int) -> None`: `random`, `numpy`, `torch` 시드를 고정한다
  - `save_json(path, obj) -> None`: UTF-8, `ensure_ascii=False`, indent 2. 상위 폴더를 만든다
- Produces (`boxes.py`):
  - `Box = tuple[float, float, float, float]`, `FULL_BOX: Box = (0.0, 0.0, 1.0, 1.0)`
  - `BOX_COLUMNS = ["image_path", "x0", "y0", "x1", "y1", "score", "source", "detector"]`, `BOX_SOURCES = ("auto", "manual", "fallback")`
  - `BoxRecord`: `{"image_path": str, "box": Box, "score": float, "source": str, "detector": str}` 형태의 dict
  - `validate_box(box) -> Box`: float로 바꾸고, `0 ≤ x0 < x1 ≤ 1`, `0 ≤ y0 < y1 ≤ 1`이 아니면 `ValueError`
  - `expand_box(box: Box, margin: float) -> Box`: 좌우는 박스 폭 × margin, 상하는 박스 높이 × margin만큼 넓히고 [0, 1]로 자른다
  - `jitter_box(box: Box, shift: float, scale_range: tuple[float, float], generator: torch.Generator | None = None) -> Box`: 중심을 (폭, 높이) × U(-shift, shift)만큼 옮기고 폭·높이를 각각 U(scale_range)배로 바꾼 뒤 [0, 1]로 자른다. 결과는 항상 유효해야 한다. 난수는 `torch.rand`로 뽑아서 DataLoader 시드를 따른다
  - `to_pixels(box: Box, width: int, height: int) -> tuple[int, int, int, int]`: `left=floor(x0·W)`, `right=ceil(x1·W)`(위아래도 같음), [0, W]×[0, H]로 자른다. 폭·높이가 1 px 미만이면 이미지 안에서 1 px로 넓힌다
  - `read_boxes(path) -> dict[str, BoxRecord]`: `image_path`가 키다. 파일이 없으면 `{}`, 컬럼이나 `source` 값이 틀리면 `ValueError`
  - `write_boxes(records: Iterable[BoxRecord], path) -> None`: `BOX_COLUMNS` 순서로 펼쳐서 `image_path` 순으로 정렬해 쓴다. 상위 폴더를 만든다
- `requirements.txt`: `numpy, pandas, pillow, pyyaml, tqdm, scikit-learn, timm, transformers, pytest`. 맨 위 주석에 torch 설치 명령(`--index-url https://download.pytorch.org/whl/cu128`, `torch==2.11.0 torchvision==0.26.0`)을 적는다

- [ ] **Step 1: 실패하는 테스트 작성** (`tests/model2/test_boxes.py`, `tests/model2/test_utils.py`)

```python
def test_expand_box_adds_margin_and_clamps():
    assert expand_box((0.4, 0.4, 0.6, 0.6), 0.15) == pytest.approx((0.37, 0.37, 0.63, 0.63))
    assert expand_box((0.0, 0.9, 0.2, 1.0), 0.5) == pytest.approx((0.0, 0.85, 0.3, 1.0))

@pytest.mark.parametrize("bad", [(0.5, 0.1, 0.4, 0.2), (-0.1, 0, 0.5, 0.5), (0, 0, 1.2, 1), (0.2, 0.2, 0.2, 0.4)])
def test_validate_box_rejects_bad(bad):
    with pytest.raises(ValueError):
        validate_box(bad)

def test_jitter_box_stays_valid_and_varies():
    g = torch.Generator().manual_seed(0)
    boxes = [jitter_box((0.4, 0.4, 0.6, 0.6), 0.05, (0.9, 1.1), g) for _ in range(50)]
    for b in boxes:
        validate_box(b)
        assert b[0] >= 0.38 - 1e-9 and b[2] <= 0.62 + 1e-9
    assert len(set(boxes)) > 1
    validate_box(jitter_box(FULL_BOX, 0.05, (0.9, 1.1), g))
    assert jitter_box((0.4, 0.4, 0.6, 0.6), 0.0, (1.0, 1.0), g) == pytest.approx((0.4, 0.4, 0.6, 0.6))

def test_to_pixels_min_one_pixel():
    l, t, r, b = to_pixels((0.5, 0.5, 0.5001, 0.5001), 160, 120)
    assert r - l >= 1 and b - t >= 1
    assert to_pixels((0.99, 0.99, 1.0, 1.0), 100, 100) == (99, 99, 100, 100)
    assert to_pixels(FULL_BOX, 64, 48) == (0, 0, 64, 48)
    assert to_pixels((0.25, 0.5, 0.75, 1.0), 100, 10) == (25, 5, 75, 10)

def test_boxes_csv_roundtrip(tmp_path):
    recs = [{"image_path": "b.jpg", "box": (0.1, 0.2, 0.3, 0.4), "score": 0.9, "source": "auto", "detector": "d"},
            {"image_path": "a.jpg", "box": FULL_BOX, "score": 0.0, "source": "fallback", "detector": "d"}]
    write_boxes(recs, tmp_path / "x" / "boxes.csv")
    got = read_boxes(tmp_path / "x" / "boxes.csv")
    assert set(got) == {"a.jpg", "b.jpg"} and got["b.jpg"]["box"] == pytest.approx((0.1, 0.2, 0.3, 0.4))
    assert read_boxes(tmp_path / "missing.csv") == {}
    # source 값이 틀린 파일은 ValueError
```

```python
def test_select_device_cpu_and_auto():
    assert select_device("cpu").type == "cpu"
    assert select_device("auto").type in {"cuda", "mps", "cpu"}

def test_select_device_unavailable_cuda_raises():
    if torch.cuda.is_available():
        pytest.skip("CUDA 있음")
    with pytest.raises(RuntimeError):
        select_device("cuda")

def test_load_config_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
```

- [ ] **Step 2: 실패 확인** — `$PY -m pytest tests/model2/test_boxes.py tests/model2/test_utils.py -q` → ImportError로 FAIL
- [ ] **Step 3: 위 Interfaces대로 `utils.py`, `boxes.py`, `requirements.txt`, 빈 `__init__.py` 두 개를 작성한다**
- [ ] **Step 4: 통과 확인** — 같은 명령 → 전부 PASS
- [ ] **Step 5: 커밋** — `git add src/models/model2_memo tests/__init__.py tests/model2 && git commit -m "feat: 모델 2 박스 유틸과 공용 유틸 추가"`

---

### Task 2: 합성 데이터 생성기

**Files:**
- Create: `src/models/model2_memo/scripts/__init__.py`, `src/models/model2_memo/scripts/make_smoke_data.py`
- Create: `tests/model2/conftest.py`
- Test: `tests/model2/test_smoke_data.py`

**Interfaces:**
- Consumes: `write_boxes`, `BoxRecord` (Task 1)
- Produces:
  - `make_smoke_data(out: Path, items: int = 6, per_item: int = 4, c_per_item: int = 2, size: tuple[int, int] = (320, 240), seed: int = 0) -> Path`: `out/model1_metadata.csv` 경로를 돌려준다. `items >= 3`이 아니면 `ValueError`
  - 출력: `out/data/processed/{original,kakao,resize}/{stem}.jpg`, `out/model1_metadata.csv`, `out/memo_boxes.csv`
  - CSV 컬럼은 Global Constraints의 입력 CSV 컬럼 순서 그대로다. `image_path`는 `out` 기준 상대 경로(`data/processed/kakao/<stem>.jpg`)이고, `domain="smoke"`다
  - stem 규칙(팀 파일명 규칙을 따름):
    - 진짜: `p{k}_item{i}_{n:03d}` (k = i % 3)
    - A/B: `fake{A|B}_<진짜 stem>_s{seed}`
    - C: `fakeC_item{i}_{n:03d}_s{seed}`
  - `source_id`는 진짜·A·B가 진짜 stem이고, C는 자기 stem이다. `item_id = item{i}`, `photographer = p{k}`(C는 빈칸)
  - split은 물건 단위로 나눈다: 마지막 물건 → `test`, 그 앞 물건 → `val`, 나머지 → `train`
  - 후처리 버전:
    - original: JPEG 품질 95
    - kakao: 긴 변을 0.75배로 줄이고 JPEG 품질 40
    - resize: 가로·세로를 각각 0.5배로 줄이고 JPEG 품질 95
  - 박스: 이미지마다 같은 정규화 메모 박스, `source="manual"`, `score=1.0`, `detector="smoke-gt"`
  - 그림 규칙 (모델 2가 배울 수 있어야 함):
    - 진짜: 노이즈 배경 + "제품" 사각형 + 밝은 종이 사각형에 선명한 글씨(`"2026.11.04 p{k}"`, `ImageFont.load_default()`)
    - A: 원본과 같고, 메모 안쪽만 다른 글씨로 다시 쓴 뒤 `GaussianBlur(1.5)`와 따뜻한 색조를 준다
    - B: 원본과 같고, 제품 사각형만 색을 바꾸고 블러한다. **메모 픽셀은 원본과 같다**
    - C: 위치를 새로 정해 독립적으로 그린 뒤 이미지 전체에 `GaussianBlur(1.0)`
  - CLI: `python -m src.models.model2_memo.scripts.make_smoke_data OUT [--items --per-item --c-per-item --width --height --seed]`
  - `conftest.py`: session 범위 fixture `smoke_data(tmp_path_factory) -> Path`. `make_smoke_data(<tmp>/smoke, items=6, per_item=2, c_per_item=1, size=(160, 120), seed=0)`을 부르고 CSV 경로를 돌려준다

- [ ] **Step 1: 실패하는 테스트 작성** (`tests/model2/test_smoke_data.py`)

```python
COLS = ["image_path", "label", "fake_type", "source_id", "split", "domain", "item_id", "photographer", "postprocess"]

def test_smoke_csv_schema(smoke_data):
    df = pd.read_csv(smoke_data, keep_default_na=False)
    assert list(df.columns) == COLS
    assert set(df.fake_type) == {"real", "A", "B", "C"}
    assert set(df.split) == {"train", "val", "test"}
    assert set(df.postprocess) == {"original", "kakao", "resize"}
    assert ((df.label == 0) == (df.fake_type == "real")).all()
    assert len(df) == (6 * 2 * 3 + 6 * 1) * 3   # (진짜+A+B + C) × 후처리 3종 = 126

def test_smoke_files_and_boxes_exist(smoke_data):
    df = pd.read_csv(smoke_data, keep_default_na=False)
    assert all((smoke_data.parent / p).is_file() for p in df.image_path)
    boxes = read_boxes(smoke_data.parent / "memo_boxes.csv")
    assert set(boxes) == set(df.image_path)
    assert {r["source"] for r in boxes.values()} == {"manual"}

def test_smoke_split_has_no_leakage(smoke_data):
    df = pd.read_csv(smoke_data, keep_default_na=False)
    assert (df.groupby("source_id").split.nunique() == 1).all()
    assert (df.groupby("item_id").split.nunique() == 1).all()

def test_fake_b_keeps_memo_and_fake_a_changes_it(smoke_data):
    # original 버전에서 B와 그 원본 진짜의 메모 박스 영역 평균 절대차 < 3, A와 원본은 > 10
    ...

def test_resize_variant_is_half_size(smoke_data):
    # 같은 stem의 resize 이미지 크기 == original 크기 // 2 (±1 px)
    ...
```
(`...` 두 개는 주석에 적은 단언을 그대로 코드로 쓴다.)

- [ ] **Step 2: 실패 확인** — `$PY -m pytest tests/model2/test_smoke_data.py -q` → FAIL
- [ ] **Step 3: `make_smoke_data`, CLI, `conftest.py`를 작성한다**
- [ ] **Step 4: 통과 확인** — 같은 명령 → PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: 모델 2 파이프라인 점검용 합성 데이터 생성기 추가"`

---

### Task 3: 메타데이터 검증과 메모 크롭 데이터셋

**Files:**
- Create: `src/models/model2_memo/dataset.py`
- Test: `tests/model2/test_dataset.py`

**Interfaces:**
- Consumes: `Box`, `expand_box`, `jitter_box`, `to_pixels`, `BoxRecord` (Task 1). fixture `smoke_data` (Task 2)
- Produces:
  - `REQUIRED_COLUMNS = ["image_path", "label", "fake_type", "source_id", "split"]`, `POSTPROCESSES = ("original", "kakao", "resize", "recapture")`
  - `infer_postprocess(image_path: str) -> str`: `\`를 `/`로 바꾼 뒤 `(?:^|/)processed/(original|kakao|resize|recapture)/`에 맞으면 그 값, 아니면 `"original"`
  - `read_metadata(csv_path, root, leakage_columns: Sequence[str] = ("item_id",), verify_images: bool = True) -> pd.DataFrame`
    - 텍스트 컬럼은 str, `keep_default_na=False`로 읽는다. `label`은 int로 바꾼다
    - 추가 컬럼:
      - `postprocess`: 컬럼이 있고 비어 있지 않으면 그 값, 아니면 `infer_postprocess`
      - `domain`: 없으면 `""`
      - `_resolved_path`: `root / image_path`를 resolve한 문자열
    - `ValueError` 조건:
      - 필수 컬럼이 없거나 필수 값이 빈칸
      - CSV에 행이 없음
      - label이 0/1이 아니거나, `fake_type`이 label과 맞지 않음
      - split이나 postprocess 값이 규칙 밖
      - resolve한 경로가 중복
      - `source_id`나 (있는) `leakage_columns` 값이 여러 split에 걸침 (메시지에 `"leak"` 포함. leakage 컬럼의 빈 값은 검사하지 않음)
    - `FileNotFoundError` 조건: CSV나 이미지가 없음
    - `verify_images=True`면 PIL `verify()`를 실행한다
  - `select_rows(df, split: str, exclude_types: Sequence[str] = (), postprocesses: Sequence[str] | None = None) -> pd.DataFrame`: 거른 뒤 `reset_index(drop=True)`. 비면 `ValueError`
  - `class MemoCropTransform(input_size: int, mean, std, margin: float, train: bool = False, jitter_shift: float = 0.0, jitter_scale: tuple[float, float] = (1.0, 1.0))`, `__call__(image: Image.Image, box: Box) -> torch.Tensor`
    - 순서: `exif_transpose` → RGB → (train이면 `jitter_box`) → `expand_box(margin)` → `to_pixels` → crop → `ImageOps.pad((S, S), BICUBIC, color=round(255·mean))` → `ToTensor` → `Normalize`
    - 좌우 반전은 없다. 평가 모드는 결정적이다
  - `build_transform(config: dict, train: bool) -> MemoCropTransform`: `config["model"]`의 `input_size`/`mean`/`std`와 `config["crop"]`의 `margin`/`jitter_shift`/`jitter_scale`을 쓴다. jitter는 `train=True`일 때만 켠다
  - `class MemoCropDataset(frame: pd.DataFrame, boxes: dict[str, BoxRecord], transform: MemoCropTransform)`
    - `frame.image_path` 중 `boxes`에 없는 것이 있으면 `ValueError`. 메시지에 개수, 앞의 경로 3개, `"python -m src.models.model2_memo.locate --metadata"` 안내를 넣는다
    - `__getitem__` → `{"image": Tensor[3,S,S], "label": float32 tensor, "image_path", "fake_type", "source_id", "postprocess", "box_source", "domain"}`

- [ ] **Step 1: 실패하는 테스트 작성** (`tests/model2/test_dataset.py`)

```python
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def test_infer_postprocess():
    assert infer_postprocess("data/processed/kakao/a.jpg") == "kakao"
    assert infer_postprocess("data\\processed\\resize\\a.jpg") == "resize"
    assert infer_postprocess("x/processed/recapture/b.jpg") == "recapture"
    assert infer_postprocess("data/real/a.jpg") == "original"

def test_read_metadata_smoke_ok(smoke_data):
    df = read_metadata(smoke_data, smoke_data.parent)
    assert len(df) == 126 and "_resolved_path" in df
    assert set(df.postprocess) == {"original", "kakao", "resize"}

# 아래 경우마다 smoke CSV를 고친 복사본을 tmp_path에 써서 확인한다
# label=2 → ValueError / 첫 진짜 행 fake_type="A" → ValueError / split="dev" → ValueError
# test 행 하나의 source_id를 train source_id로 → ValueError(match="leak")
# test 행 하나의 item_id를 train item_id로 → ValueError(match="leak")
# split 컬럼 삭제 → ValueError / 없는 이미지 경로 → FileNotFoundError
def test_read_metadata_rejects(...): ...

def test_read_metadata_infers_postprocess_when_column_missing(smoke_data, tmp_path):
    # postprocess 컬럼을 지운 CSV → df.postprocess == infer_postprocess(image_path) (전부)
    ...

def test_select_rows_excludes_b_and_filters_postprocess(smoke_data):
    df = read_metadata(smoke_data, smoke_data.parent)
    tr = select_rows(df, "train", ["B"], ["original"])
    assert "B" not in set(tr.fake_type) and set(tr.postprocess) == {"original"}
    te = select_rows(df, "test")
    assert "B" in set(te.fake_type) and set(te.postprocess) == {"original", "kakao", "resize"}

def test_transform_eval_shape_and_deterministic():
    img = Image.new("RGB", (160, 120), (120, 80, 40))
    t = MemoCropTransform(64, MEAN, STD, 0.15)
    a, b = t(img, (0.2, 0.2, 0.6, 0.7)), t(img, (0.2, 0.2, 0.6, 0.7))
    assert a.shape == (3, 64, 64) and torch.equal(a, b)

def test_transform_train_jitter_varies():
    # 가로 그라데이션 이미지에 train=True, jitter_shift=0.05로 두 번 호출 → 결과가 다름
    ...

def test_transform_tiny_edge_box():
    out = MemoCropTransform(64, MEAN, STD, 0.15)(Image.new("RGB", (160, 120)), (0.999, 0.999, 1.0, 1.0))
    assert out.shape == (3, 64, 64) and torch.isfinite(out).all()

def test_transform_handles_non_rgb():
    t = MemoCropTransform(64, MEAN, STD, 0.15)
    for mode in ["L", "RGBA"]:
        assert t(Image.new(mode, (50, 40)), (0.1, 0.1, 0.9, 0.9)).shape == (3, 64, 64)

def test_crop_uses_box_region():
    # 100×100 이미지, 왼쪽 절반 검정·오른쪽 절반 흰색. margin=0, 정사각 박스라 패딩 없음
    # 박스 (0.6, 0.2, 0.9, 0.5) → 역정규화 평균 > 0.9 / 박스 (0.1, 0.2, 0.4, 0.5) → < 0.1
    ...

def test_dataset_missing_boxes_raises(smoke_data):
    df = read_metadata(smoke_data, smoke_data.parent)
    boxes = read_boxes(smoke_data.parent / "memo_boxes.csv")
    boxes.pop(df.image_path[0])
    with pytest.raises(ValueError, match="locate"):
        MemoCropDataset(df, boxes, MemoCropTransform(64, MEAN, STD, 0.15))

def test_dataset_item_fields(smoke_data):
    df = read_metadata(smoke_data, smoke_data.parent)
    ds = MemoCropDataset(df, read_boxes(smoke_data.parent / "memo_boxes.csv"), MemoCropTransform(64, MEAN, STD, 0.15))
    item = ds[0]
    assert {"image", "label", "image_path", "fake_type", "source_id", "postprocess", "box_source", "domain"} <= set(item)
    assert item["label"].dtype == torch.float32 and item["box_source"] == "manual"
```
(`...`로 둔 테스트는 주석의 단언을 그대로 코드로 쓴다.)

- [ ] **Step 2: 실패 확인** — `$PY -m pytest tests/model2/test_dataset.py -q` → FAIL
- [ ] **Step 3: 위 Interfaces대로 `dataset.py`를 작성한다**
- [ ] **Step 4: 통과 확인** — 같은 명령 → PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: 모델 2 메타데이터 검증과 메모 크롭 데이터셋 추가"`

---

### Task 4: 메모 위치 찾기 (`locate.py`)와 팀 데이터 설정

**Files:**
- Create: `src/models/model2_memo/locate.py`
- Create: `configs/model2_memo.yaml`
- Create: `metadata/memo_boxes_manual.csv` (헤더만: `stem,postprocess,x0,y0,x1,y1,note`)
- Test: `tests/model2/test_locate.py`

**Interfaces:**
- Consumes: `Box`, `FULL_BOX`, `validate_box`, `read_boxes`, `write_boxes`, `BoxRecord`, `load_config`, `select_device` (Task 1). `read_metadata`, `POSTPROCESSES` (Task 3)
- Produces:
  - `Detection = tuple[Box, float]`
  - `class Detector(Protocol)`: `name: str`, `detect(self, image: Image.Image) -> list[Detection]`
  - `class HFZeroShotDetector(model_id: str, queries: Sequence[str], box_threshold: float, text_threshold: float, device: str = "auto")`
    - `AutoProcessor`, `AutoModelForZeroShotObjectDetection`을 쓴다
    - 텍스트는 쿼리를 소문자로 바꿔 `". "`로 잇고 끝에 `"."`를 붙인다 (Grounding DINO 형식)
    - `processor.post_process_grounded_object_detection(..., target_sizes=[(H, W)])`로 픽셀 박스를 받아 정규화하고 [0, 1]로 자른 뒤, 유효하지 않은 박스는 버린다
    - 이미지는 `exif_transpose` + RGB로 바꾼 뒤 넣는다. `name = model_id`
    - 설치된 transformers는 5.x다. `post_process_grounded_object_detection`의 인자 이름(예: `box_threshold` → `threshold`)은 설치된 버전의 시그니처를 확인해서 맞춘다
  - `choose_box(detections: list[Detection], min_area_frac: float, max_area_frac: float) -> Detection | None`: 면적이 범위 안에 있는 것 중 점수가 가장 높은 것 (동점이면 앞의 것)
  - `read_manual_boxes(path) -> dict[tuple[str, str], Box]`: 키는 `(stem, postprocess 또는 "")`. 파일이 없으면 `{}`. 키 중복, postprocess 값 오류, 박스 오류는 `ValueError`
  - `manual_box_for(manual, image_path: str, postprocess: str) -> Box | None`: `(stem, postprocess)`를 먼저 찾고, 없으면 postprocess가 `original`/`kakao`/`resize`일 때만 `(stem, "")`를 찾는다
  - `locate_frame(df, get_detector: Callable[[], Detector], manual, cached: dict[str, BoxRecord], manual_splits: Sequence[str], min_area_frac: float, max_area_frac: float, overwrite: bool = False) -> list[BoxRecord]` — `df`의 행마다 정확히 한 레코드를 만든다:
    1. 행의 `split`이 `manual_splits`에 있고 수동 박스가 있으면 → `source="manual"`, `score=1.0`, `detector="manual"`
    2. 아니면, `cached`에 `source`가 `auto`/`fallback`인 레코드가 있고 `overwrite=False`면 → 재사용 (`manual` 캐시 레코드는 재사용하지 않고 다시 탐지)
    3. 아니면 탐지 → `choose_box` → 있으면 `auto`, 없으면 `fallback`(`FULL_BOX`, `score=0.0`)
    - `get_detector()`는 탐지가 필요할 때 **처음 한 번만** 부른다 (필요 없으면 부르지 않음 → 모델 다운로드 없음)
  - `locate_report(df, boxes: dict[str, BoxRecord], threshold: float = 0.10) -> tuple[pd.DataFrame, list[str]]`
    - 표 컬럼: `split, fake_type, postprocess, n, auto_rate, manual_rate, fallback_rate`
    - 경고: 같은 `(split, postprocess)` 안에서 `real`이 아닌 각 유형 t에 대해 `|fallback_rate(t) − fallback_rate(real)|` 또는 `|manual_rate(t) − manual_rate(real)|`가 threshold를 넘으면 문장 하나를 만든다. 문장에는 split, postprocess, t, 비율 이름(`fallback`/`manual`), 두 비율을 넣는다
  - `run(config_path, metadata: str | None = None, overwrite: bool = False, device: str | None = None, detector: Detector | None = None) -> pd.DataFrame`
    - 설정을 읽고 `read_metadata`를 부른다 (`metadata`가 있으면 설정의 CSV 대신 쓴다)
    - 캐시와 수동 박스를 읽고 `locate_frame`을 돌린다. `detector`가 없으면 설정값으로 `HFZeroShotDetector`를 lazy하게 만든다
    - 기존 캐시의 다른 `image_path` 레코드는 유지한 채 병합해서 `write_boxes`로 쓴다
    - `<boxes_csv 폴더>/locate_report.csv`를 쓰고, `locate_report(..., threshold=locate.warn_rate_diff)`의 경고를 출력하고, 표를 돌려준다
  - `main()`: `--config`(기본 `configs/model2_memo.yaml`), `--metadata`, `--overwrite`, `--device {auto,cuda,mps,cpu}`
- `configs/model2_memo.yaml` (값은 설계 문서 그대로):

```yaml
seed: 42
data: {metadata: data/splits/model1_metadata.csv, root: ., leakage_columns: [item_id], verify_images: true,
       exclude_from_training: [B], train_postprocess: [original]}
locate: {boxes_csv: outputs/model2/memo_boxes.csv, manual_csv: metadata/memo_boxes_manual.csv, manual_splits: [train],
         model_id: IDEA-Research/grounding-dino-tiny,
         queries: [a handwritten note, a sticky note, a piece of paper with handwriting],
         box_threshold: 0.3, text_threshold: 0.25, min_area_frac: 0.002, max_area_frac: 0.8, warn_rate_diff: 0.10}
crop: {margin: 0.15, jitter_shift: 0.05, jitter_scale: [0.9, 1.1]}
model: {name: efficientnet_b0, pretrained: true, input_size: 384, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], drop_rate: 0.2}
training: {device: auto, epochs: 20, batch_size: 16, learning_rate: 0.0001, weight_decay: 0.01, num_workers: 2, amp: true,
           patience: 5, output_dir: outputs/model2}
```
(파일에는 블록 스타일 YAML로 쓰고, 섹션마다 한 줄 한국어 주석을 단다.)

- [ ] **Step 1: 실패하는 테스트 작성** (`tests/model2/test_locate.py`)
  - 테스트 파일 안에 `FakeDetector`를 정의한다: stem이 `fakeA`로 시작하면 `[]`, 아니면 `[((0.2, 0.2, 0.5, 0.5), 0.8)]`을 돌려주고 호출 횟수를 센다

```python
def test_choose_box_picks_highest_valid_area():
    dets = [(FULL_BOX, 0.9), ((0.1, 0.1, 0.2, 0.2), 0.5), ((0.5, 0.5, 0.7, 0.7), 0.7)]
    assert choose_box(dets, 0.002, 0.8) == ((0.5, 0.5, 0.7, 0.7), 0.7)
    assert choose_box([((0.1, 0.1, 0.11, 0.11), 0.9)], 0.05, 0.8) is None
    assert choose_box([], 0.002, 0.8) is None

def test_read_manual_boxes_and_lookup(tmp_path):
    # 행: "s1,,0.1,0.1,0.3,0.3," 와 "s1,recapture,0.2,0.2,0.4,0.4,"
    # manual_box_for(m, "data/processed/kakao/s1.jpg", "kakao") == (0.1, 0.1, 0.3, 0.3)
    # manual_box_for(m, "data/processed/recapture/s1.jpg", "recapture") == (0.2, 0.2, 0.4, 0.4)
    # manual_box_for(m, "data/processed/original/s2.jpg", "original") is None
    # 같은 (stem, postprocess) 행 두 개 → ValueError / postprocess "foo" → ValueError
    ...

def test_locate_frame_manual_only_in_train_and_fallback(smoke_data):
    # train 진짜 stem 하나와 test 진짜 stem 하나에 (stem, "") 수동 박스를 넣는다
    # → train 행들: source "manual" / test 행들: source "auto" (수동 박스 무시)
    # → fakeA 행들: source "fallback", box == FULL_BOX, score == 0.0
    # → 레코드 수 == len(df), image_path 중복 없음
    ...

def test_locate_frame_uses_cache_and_lazy_detector(smoke_data):
    # 모든 image_path가 auto로 캐시돼 있으면 get_detector를 부르지 않는다 (부르면 AssertionError를 내는 factory)
    # overwrite=True면 탐지기를 부른다
    ...

def test_locate_report_warns_on_rate_gaps():
    # test/original: real 4장 전부 auto, A 4장 중 3장 fallback → 경고 1개, "A"와 "fallback" 포함
    # 비율이 같으면 경고 0개 / manual 비율 차이 0.5 → "manual" 경고
    ...

def test_run_writes_cache_and_report(tmp_path, smoke_data):
    # 임시 설정: data.metadata=smoke CSV, root=smoke 폴더, boxes_csv=tmp/boxes.csv, manual_csv=tmp/없음.csv
    # boxes.csv에 "other.jpg" 레코드를 미리 써 둔다
    # run(cfg, detector=FakeDetector()) 후: 레코드 수 == len(df) + 1 ("other.jpg" 유지), tmp/locate_report.csv 존재
    ...

@pytest.mark.skipif(os.environ.get("MODEL2_RUN_DETECTOR") != "1", reason="모델 다운로드 필요")
def test_hf_detector_real(smoke_data):
    # smoke original 이미지 하나에 HFZeroShotDetector(설정값).detect → list이고, 모든 box가 validate_box 통과
    ...
```

- [ ] **Step 2: 실패 확인** — `$PY -m pytest tests/model2/test_locate.py -q` → FAIL
- [ ] **Step 3: `locate.py`, `configs/model2_memo.yaml`, `metadata/memo_boxes_manual.csv`를 작성한다**
- [ ] **Step 4: 통과 확인** — 같은 명령 → PASS (`test_hf_detector_real`은 SKIP)
- [ ] **Step 5: 커밋** — `git commit -m "feat: 모델 2 메모 위치 찾기(zero-shot 탐지, 수동 박스, 유형별 리포트) 추가"`

---

### Task 5: 모델, 지표, 학습

**Files:**
- Create: `src/models/model2_memo/model.py`, `src/models/model2_memo/metrics.py`, `src/models/model2_memo/train.py`
- Create: `configs/model2_smoke.yaml`
- Create: `tests/model2/helpers.py` (`make_cfg`, Task 6도 사용)
- Test: `tests/model2/test_metrics.py`, `tests/model2/test_train.py`

**Interfaces:**
- Consumes: `load_config`, `select_device`, `seed_everything`, `save_json`, `read_boxes` (Task 1). `read_metadata`, `select_rows`, `build_transform`, `MemoCropDataset` (Task 3). 설정 키는 Task 4의 `configs/model2_memo.yaml`
- Produces (`metrics.py`):
  - `binary_metrics(labels, probs, threshold: float) -> dict`
    - 키: `n, accuracy, precision, recall, f1, roc_auc, tp, fp, tn, fn, threshold`
    - 예측 = `prob >= threshold`. `zero_division=0`. 한 클래스만 있으면 `roc_auc=None`
  - `choose_threshold(labels, probs) -> float`: 후보는 정렬한 고유 확률값. F1이 최대인 값을 고르고, 동점이면 큰 값. 한 클래스만 있으면 `ValueError`
  - `rate_table(pred: pd.DataFrame, by: list[str]) -> pd.DataFrame`: `by` + `n, pred_fake_rate, mean_prob`. `pred_fake_rate`는 `prediction` 평균, `mean_prob`는 `fake_probability` 평균
- Produces (`model.py`):
  - `class MemoClassifier(nn.Module)(name: str = "efficientnet_b0", pretrained: bool = True, drop_rate: float = 0.2)`: `timm.create_model(name, pretrained, num_classes=1, global_pool="avg", drop_rate)`. `forward` → `[B]` logits
  - `build_model(model_cfg: dict, pretrained: bool | None = None) -> MemoClassifier`
- Produces (`train.py`): `train(config_path: str | Path, device_name: str | None = None) -> Path` (출력 폴더를 돌려줌)
  1. 설정을 읽고, 시드를 고정하고, 장치를 고른다. `read_metadata`, `read_boxes(locate.boxes_csv)`
  2. 학습·검증 행: `select_rows(df, "train"|"val", data.exclude_from_training, data.train_postprocess)`
  3. val에 진짜와 가짜가 모두 없으면 `ValueError("val needs both real and fake ...")`
  4. `<output_dir>/checkpoints/{best,last}_model.pth`가 이미 있으면 `FileExistsError`
  5. 출력 폴더에 `config.yaml` 복사본과 `data_summary.csv`(컬럼 `split, fake_type, postprocess, box_source, n`, 실제로 쓴 train·val 행 기준)를 쓴다. val 행 중 `box_source == "manual"`이 있으면 반목표 위반 가능성을 알리는 경고를 출력한다 (합성 데이터는 정답 박스라 예외이므로 오류로 멈추지는 않는다)
  6. DataLoader:
     - train은 시드 고정 generator로 shuffle한다
     - 학습 행 수 % batch == 1이면 `drop_last`
     - `num_workers`는 설정값을 쓴다
  7. 학습 설정:
     - AdamW, `CosineAnnealingLR(T_max=epochs)`, `BCEWithLogitsLoss`
     - CUDA에서 `training.amp`가 true면 autocast + GradScaler. val은 float32
  8. epoch마다:
     - val 확률로 `choose_threshold`를 구하고, 그 임계값으로 `binary_metrics`를 계산한다
     - val F1이 이전 최고보다 **클 때만** `best_model.pth`를 저장한다. `last_model.pth`는 매번 저장한다
     - `logs/training.csv`에 한 줄씩 쓴다: `epoch, lr, train_loss, val_loss, val_f1, val_auc, threshold`
  9. `patience` epoch 동안 개선이 없으면 멈춘다. test는 읽지 않는다
  - 체크포인트 dict 키: `model_state, config, epoch, threshold, val_metrics, best_f1, protected_ids`
  - `main()`: `--config`(기본 `configs/model2_memo.yaml`), `--device {auto,cuda,mps,cpu}`. `ValueError`/`FileNotFoundError`/`FileExistsError`는 `parser.exit(1, "Error: ...")`로 처리한다
- `configs/model2_smoke.yaml`: `model2_memo.yaml`과 같은 키에서 다음만 바꾼다
  - `data.metadata: outputs/model2_smoke_data/model1_metadata.csv`, `data.root: outputs/model2_smoke_data`
  - `locate.boxes_csv: outputs/model2_smoke_data/memo_boxes.csv`
  - `model.input_size: 128`, `training.epochs: 5`, `training.num_workers: 0`, `training.output_dir: outputs/model2_smoke`

- [ ] **Step 1: 실패하는 테스트 작성** (`tests/model2/test_metrics.py`, `tests/model2/test_train.py`)

```python
def test_binary_metrics_values():
    m = binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1)
    assert m["accuracy"] == m["precision"] == m["recall"] == m["f1"] == pytest.approx(0.5)
    assert m["roc_auc"] == pytest.approx(0.75)

def test_binary_metrics_single_class_auc_none():
    assert binary_metrics([1, 1], [0.2, 0.9], 0.5)["roc_auc"] is None

def test_choose_threshold_maximizes_f1():
    assert choose_threshold([0, 0, 1, 1], [0.1, 0.3, 0.6, 0.8]) == pytest.approx(0.6)
    assert choose_threshold([0, 0, 1, 1], [0.2, 0.7, 0.7, 0.9]) == pytest.approx(0.7)
    with pytest.raises(ValueError):
        choose_threshold([1, 1], [0.2, 0.9])

def test_rate_table():
    pred = pd.DataFrame({"fake_type": ["A", "A", "real"], "postprocess": ["original"] * 3,
                         "prediction": [1, 0, 1], "fake_probability": [0.9, 0.3, 0.6]})
    t = rate_table(pred, ["fake_type", "postprocess"]).set_index("fake_type")
    assert t.loc["A", "n"] == 2 and t.loc["A", "pred_fake_rate"] == pytest.approx(0.5)
    assert t.loc["real", "pred_fake_rate"] == pytest.approx(1.0)
```

```python
# tests/model2/helpers.py의 make_cfg(tmp_path, smoke_data, **override): configs/model2_memo.yaml을 읽어
# metadata/root/boxes_csv를 smoke로, input_size=64, pretrained=False, epochs=2, batch_size=8,
# num_workers=0, device=cpu, output_dir=tmp_path/"out"으로 바꾼 yaml을 써서 경로를 돌려준다
def test_build_model_output_shape():
    m = build_model({"name": "efficientnet_b0", "pretrained": False, "drop_rate": 0.2})
    assert m(torch.zeros(2, 3, 64, 64)).shape == (2,)

def test_train_end_to_end(tmp_path, smoke_data):
    out = train(make_cfg(tmp_path, smoke_data))
    assert (out / "checkpoints/best_model.pth").is_file() and (out / "checkpoints/last_model.pth").is_file()
    assert 1 <= len(pd.read_csv(out / "logs/training.csv")) <= 2
    ckpt = torch.load(out / "checkpoints/best_model.pth", weights_only=False)
    assert 0.0 <= ckpt["threshold"] <= 1.0 and ckpt["config"]["model"]["input_size"] == 64
    summary = pd.read_csv(out / "data_summary.csv", keep_default_na=False)
    assert "B" not in set(summary.fake_type) and set(summary.postprocess) == {"original"}
    assert set(summary.split) == {"train", "val"}

def test_train_refuses_existing_output(tmp_path, smoke_data):
    cfg = make_cfg(tmp_path, smoke_data, epochs=1)
    train(cfg)
    with pytest.raises(FileExistsError):
        train(cfg)

def test_train_requires_both_classes_in_val(tmp_path, smoke_data):
    # smoke CSV에서 val의 진짜 행을 모두 지운 복사본으로 설정 → ValueError(match="both")
    ...
```

- [ ] **Step 2: 실패 확인** — `$PY -m pytest tests/model2/test_metrics.py tests/model2/test_train.py -q` → FAIL
- [ ] **Step 3: `metrics.py`, `model.py`, `train.py`, `configs/model2_smoke.yaml`을 작성한다**
- [ ] **Step 4: 통과 확인** — 같은 명령 → PASS (CPU에서 1분 이내)
- [ ] **Step 5: 커밋** — `git commit -m "feat: 모델 2 분류기·지표·학습 스크립트 추가"`

---

### Task 6: 평가 (`evaluate.py`)

**Files:**
- Create: `src/models/model2_memo/evaluate.py`
- Modify: `tests/model2/conftest.py` (fixture 추가)
- Test: `tests/model2/test_evaluate.py`

**Interfaces:**
- Consumes: Task 1 유틸, Task 3 `read_metadata`/`select_rows`/`build_transform`/`MemoCropDataset`, Task 5 `build_model`/`binary_metrics`/`rate_table`, 체크포인트 키(Task 5)
- Produces:
  - `PREDICTION_COLUMNS = ["image_path", "label", "fake_type", "fake_probability", "prediction", "source_id", "domain", "postprocess", "box_source"]`
  - `evaluate(checkpoint_path, config_path: str | None = None, metadata: str | None = None, output_dir: str | None = None, device_name: str | None = None) -> dict`
    - 모델·크롭 설정과 임계값은 **체크포인트의 `config`와 `threshold`** 를 쓴다
    - `config_path`가 있으면 그 설정의 `data`/`locate`/`training(device, batch_size, num_workers)`를 쓰고, `metadata`가 있으면 `data.metadata`를 대체한다
    - test split **전체**를 쓴다: 유형 제외와 postprocess 필터 없음
    - test 행 중 `box_source == "manual"`이 있으면 개수와 함께 경고를 출력한다 (val/test 수동 박스는 반목표)
    - 출력 폴더(기본 `<training.output_dir>/predictions`)에 쓰는 파일:
      - `test_predictions.csv` (`PREDICTION_COLUMNS` 순서, `prediction = int(prob >= threshold)`)
      - `test_metrics.json`: `{checkpoint, epoch, threshold, overall, by_postprocess: {pp: binary_metrics}, by_fake_type: [...], by_fake_type_postprocess: [...]}`
      - `test_rates.csv`: `rate_table(by=["fake_type", "postprocess"])`
    - 돌려주는 값은 metrics dict다
  - `main()`: `--checkpoint`(필수), `--config`, `--metadata`, `--output-dir`, `--device`
  - `conftest.py`에 session fixture `trained_model(tmp_path_factory, smoke_data) -> tuple[Path, Path]` (설정 경로, 출력 폴더)를 추가한다. `tests/model2/helpers.py`의 `make_cfg(..., epochs=1)`로 학습을 한 번 한다

- [ ] **Step 1: 실패하는 테스트 작성** (`tests/model2/test_evaluate.py`)

```python
def test_evaluate_writes_predictions(tmp_path, smoke_data, trained_model):
    cfg, out = trained_model
    report = evaluate(out / "checkpoints/best_model.pth", cfg, output_dir=tmp_path / "pred")
    df = pd.read_csv(tmp_path / "pred/test_predictions.csv", keep_default_na=False)
    src = pd.read_csv(smoke_data, keep_default_na=False)
    assert list(df.columns) == PREDICTION_COLUMNS
    assert len(df) == (src.split == "test").sum()          # B와 후처리 3종 모두 포함
    assert "B" in set(df.fake_type) and set(df.postprocess) == {"original", "kakao", "resize"}
    assert df.fake_probability.between(0, 1).all()
    assert (df.prediction == (df.fake_probability >= report["threshold"]).astype(int)).all()
    assert (tmp_path / "pred/test_metrics.json").is_file() and (tmp_path / "pred/test_rates.csv").is_file()

def test_evaluate_output_dir_does_not_touch_default(tmp_path, trained_model):
    cfg, out = trained_model
    evaluate(out / "checkpoints/best_model.pth", cfg, output_dir=tmp_path / "p2")
    assert not (out / "predictions/test_predictions.csv").exists()

def test_evaluate_metadata_override(tmp_path, smoke_data, trained_model):
    # test·original 행만 남긴 CSV(같은 폴더 구조 기준 경로)를 metadata로 넘김 → 예측 행 수 == 그 부분집합 크기
    ...
```

- [ ] **Step 2: 실패 확인** — `$PY -m pytest tests/model2/test_evaluate.py -q` → FAIL
- [ ] **Step 3: `evaluate.py`와 fixture를 작성한다**
- [ ] **Step 4: 통과 확인** — `$PY -m pytest tests/model2 -q` → 전부 PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: 모델 2 평가 스크립트 추가 (고정 임계값, 유형·후처리별 지표)"`

---

### Task 7: 모델 1 vs 모델 2 비교표 (`compare.py`)

**Files:**
- Create: `src/models/model2_memo/compare.py`
- Test: `tests/model2/test_compare.py`

**Interfaces:**
- Consumes: `infer_postprocess` (Task 3)
- Produces:
  - `compare(model1_csv, model2_csv) -> tuple[pd.DataFrame, dict]`
    - 두 파일 모두 `image_path, label, fake_type, fake_probability, prediction` 컬럼이 있어야 한다. 없으면 빠진 컬럼 이름을 담은 `ValueError`
    - `postprocess`는 두 파일 모두 `infer_postprocess(image_path)`로 정한다 (파일의 postprocess 컬럼은 쓰지 않음)
    - `image_path`로 inner join한다. `label`이나 `fake_type`이 다르면 `ValueError`
    - `info = {"n_joined", "missing_in_model1", "missing_in_model2"}`
    - 표 행 순서: postprocess (`original, kakao, resize, recapture` 중 있는 것) × fake_type (`real, A, B, C, unknown` 중 있는 것)
    - 표 컬럼: `postprocess, fake_type, n, metric, model1_rate, model2_rate, model1_auc, model2_auc`
      - `metric`은 `real`이면 `"fpr"`, 나머지는 `"recall"`이다. rate는 `prediction` 평균이다
      - AUC는 가짜 행에만 계산한다: 같은 postprocess의 real 행과 비교하고, real이 없으면 NaN이다. real 행의 AUC는 NaN이다
  - `write_comparison(table, info, out_dir) -> None`: `comparison.csv`와 `comparison.md`를 쓴다
    - md 첫 줄: original에 A가 있으면 `A 유형 탐지율 (original): 모델1 {:.1%} → 모델2 {:.1%}`
    - 그다음 Markdown 표, 그다음 빠진 행 수
  - `main()`: `--model1`, `--model2`(필수), `--out`(기본 `outputs/model2/comparison`)

- [ ] **Step 1: 실패하는 테스트 작성** (`tests/model2/test_compare.py`)

```python
def _write(path, rows):  # rows: (image_path, label, fake_type, prob, pred)
    pd.DataFrame(rows, columns=["image_path", "label", "fake_type", "fake_probability", "prediction"]).to_csv(path, index=False)

def test_compare_rates_and_auc(tmp_path):
    o = "data/processed/original/"
    _write(tmp_path / "m1.csv", [(o + "r1.jpg", 0, "real", 0.1, 0), (o + "r2.jpg", 0, "real", 0.2, 0),
                                 (o + "a1.jpg", 1, "A", 0.3, 0), (o + "a2.jpg", 1, "A", 0.9, 1)])
    _write(tmp_path / "m2.csv", [(o + "r1.jpg", 0, "real", 0.1, 0), (o + "r2.jpg", 0, "real", 0.6, 1),
                                 (o + "a1.jpg", 1, "A", 0.8, 1), (o + "a2.jpg", 1, "A", 0.9, 1)])
    table, info = compare(tmp_path / "m1.csv", tmp_path / "m2.csv")
    a = table.set_index(["postprocess", "fake_type"]).loc[("original", "A")]
    assert a.n == 2 and a.metric == "recall"
    assert a.model1_rate == pytest.approx(0.5) and a.model2_rate == pytest.approx(1.0)
    assert a.model1_auc == pytest.approx(1.0) and a.model2_auc == pytest.approx(1.0)
    r = table.set_index(["postprocess", "fake_type"]).loc[("original", "real")]
    assert r.metric == "fpr" and r.model1_rate == 0.0 and r.model2_rate == pytest.approx(0.5) and math.isnan(r.model1_auc)
    assert info == {"n_joined": 4, "missing_in_model1": 0, "missing_in_model2": 0}

def test_compare_infers_postprocess_and_counts_missing(tmp_path):
    # 모델 1 CSV에는 postprocess 컬럼이 없고 경로는 data/processed/kakao/...
    # 모델 2 CSV에는 postprocess 컬럼이 있고 행이 하나 더 있음
    # → 표의 postprocess == {"kakao"}, info["missing_in_model1"] == 1, n_joined == 공통 행 수
    ...

def test_compare_label_mismatch_raises(tmp_path):
    # 같은 image_path인데 label이 다르면 ValueError
    ...

def test_write_comparison(tmp_path):
    # 위 첫 테스트의 결과로 write_comparison → comparison.csv와 comparison.md가 있고, md 첫 줄에 "A 유형"
    ...
```

- [ ] **Step 2: 실패 확인** — `$PY -m pytest tests/model2/test_compare.py -q` → FAIL
- [ ] **Step 3: `compare.py`를 작성한다**
- [ ] **Step 4: 통과 확인** — 같은 명령 → PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: 모델 1 vs 모델 2 비교표 스크립트 추가"`

---

### Task 8: README와 실제 실행 점검

**Files:**
- Modify: `src/models/model2_memo/README.md` (현재 7줄짜리 안내를 대체)

**Interfaces:**
- Consumes: Task 1~7의 CLI 전부

- [ ] **Step 1: README 작성 (한국어)** — 다음 내용을 담는다:
  - 목적: A 유형, 모델 1과의 차이
  - 설계 문서 링크(`docs/model2/design.md`)
  - 설치: `uv venv --python 3.11`, torch cu128 설치, `pip install -r src/models/model2_memo/requirements.txt`
  - 파이프라인 명령: 합성 데이터 → locate → train → evaluate → compare
  - 입력 CSV: PR #2 `split.py export`의 학습용 기본 출력과, 평가용 `--variants original kakao resize`·recapture export를 `--metadata`로 넘기는 방법
  - 수동 박스 파일 사용법: **train에만 적용됨**과 그 이유
  - 출력 위치
  - 주의사항: B를 학습에서 뺀 이유, 예측 파일 형식은 #3 결정을 따를 예정, 실제 사진 20장으로 탐지 쿼리·임계값부터 조정할 것
- [ ] **Step 2: 전체 테스트** — `$PY -m pytest tests/model2 -q` → 전부 PASS, `test_hf_detector_real` 1개 SKIP
- [ ] **Step 3: GPU 합성 데이터 실행** — 다음을 차례로 실행한다:

```bash
$PY -m src.models.model2_memo.scripts.make_smoke_data outputs/model2_smoke_data
$PY -m src.models.model2_memo.train --config configs/model2_smoke.yaml --device cuda
$PY -m src.models.model2_memo.evaluate --config configs/model2_smoke.yaml --checkpoint outputs/model2_smoke/checkpoints/best_model.pth
$PY -m src.models.model2_memo.compare --model1 outputs/model2_smoke/predictions/test_predictions.csv --model2 outputs/model2_smoke/predictions/test_predictions.csv --out outputs/model2_smoke/comparison
```
  - 기대 결과: 모든 명령이 오류 없이 끝나고 `test_rates.csv`, `comparison.md`가 생긴다
  - `test_rates.csv`의 A·C·B 탐지율을 보고서에 적는다. 단언하지는 않는다. 설계상 B는 메모가 진짜라서 낮게 나오는 것이 정상이다
  - 사전학습 가중치(약 20MB)는 이 단계에서 처음 내려받는다
- [ ] **Step 4: 커밋** — `git commit -m "docs: 모델 2 README에 사용법과 설계 요약 추가"`
