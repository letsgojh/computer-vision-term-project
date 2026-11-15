# ④ 모델 2 – 메모 영역 중심 분류기

인증샷에서 **메모 영역만 찾아 고해상도로 잘라** 진짜/가짜를 분류합니다. 설계 근거와 결정 사항은 [`docs/model2/design.md`](../../../docs/model2/design.md)에 있습니다.

## 개요

- **목적:** A 유형(메모 글씨만 인페인팅)은 이미지 전체에서 바뀐 영역이 아주 작습니다. 모델 1은 전체를 384px로 줄여 보기 때문에 이 흔적이 사라질 수 있습니다. 모델 2는 메모 크롭만 보고, "메모 중심 모델이 A를 더 잘 잡는가"를 모델 1과 비교합니다.
- **모델 1과의 차이:** 입력이 "전체 이미지"가 아니라 "메모 크롭"이라는 점 하나뿐입니다. 백본(`efficientnet_b0`), 입력 크기, 학습 방식, CSV 형식은 모델 1과 같습니다. 메모 위치는 Grounding DINO(zero-shot 탐지)로 찾습니다.
- **B를 학습에서 빼는 이유:** B는 메모가 진짜이고 제품만 가짜입니다. 메모 크롭에 "가짜" 라벨을 붙이면 진짜 메모를 가짜로 배웁니다. 그래서 `data.exclude_from_training: [B]`로 train과 val에서 뺍니다. test에는 B도 포함하며, "B는 못 잡는다"는 결과가 모델 1과의 역할 분담 근거가 됩니다.

## 설치

```bash
uv venv --python 3.11 ~/.venvs/cvtp-model2
source ~/.venvs/cvtp-model2/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0
uv pip install -r src/models/model2_memo/requirements.txt
```

모든 명령은 레포 루트에서 실행합니다. 처음 실행할 때 사전학습 가중치(약 20MB)와 Grounding DINO가 내려받아집니다.

## 빠른 점검 (합성 데이터, 실제 데이터 불필요)

학습 → 저장 → 예측 → 비교가 끝까지 도는지만 확인합니다. 성능 수치는 의미가 없습니다.

```bash
# 1. 합성 데이터 (정답 박스 memo_boxes.csv 포함, 탐지기 불필요)
python -m src.models.model2_memo.scripts.make_smoke_data outputs/model2_smoke_data
# 2. 학습
python -m src.models.model2_memo.train --config configs/model2_smoke.yaml --device cuda
# 3. 평가
python -m src.models.model2_memo.evaluate --config configs/model2_smoke.yaml --checkpoint outputs/model2_smoke/checkpoints/best_model.pth --device cuda
# 4. 비교 (여기서는 같은 파일을 두 번 넣어 명령이 도는지만 확인)
python -m src.models.model2_memo.compare --model1 outputs/model2_smoke/predictions/test_predictions.csv --model2 outputs/model2_smoke/predictions/test_predictions.csv --out outputs/model2_smoke/comparison
```

다시 실행하려면 `outputs/model2_smoke`를 먼저 지웁니다(학습은 기존 체크포인트를 덮어쓰지 않고 멈춥니다).

## 팀 데이터로 실행

### 입력 CSV

PR #2의 `src/preprocess/split.py export`가 만드는 CSV(데이터 규약 3-5)를 읽습니다. 필수 컬럼은 `image_path, label, fake_type, source_id, split`입니다. 정확한 옵션은 `python -m src.preprocess.split export --help`로 확인하세요(PR이 바뀔 수 있습니다).

| 용도 | CSV | 만드는 법 |
|---|---|---|
| 학습 (`data.metadata`) | `data/splits/model1_metadata.csv` | `python -m src.preprocess.split export` (기본, original만. 모델 1과 같은 파일) |
| 평가 (후처리 포함) | `data/splits/model1_metadata_eval.csv` | `split.py export --variants original kakao resize --out data/splits/model1_metadata_eval.csv` |
| 재촬영 평가 | `data/splits/model1_recapture.csv` | 별도 export (`--skip-missing`) |

### 실행 순서

```bash
# 1. 메모 위치 찾기 (학습 CSV). 결과는 outputs/model2/memo_boxes.csv에 캐시
python -m src.models.model2_memo.locate --config configs/model2_memo.yaml
# 평가용 CSV는 따로 locate를 돌려야 함 (같은 캐시에 추가됨)
python -m src.models.model2_memo.locate --config configs/model2_memo.yaml --metadata data/splits/model1_metadata_eval.csv

# 2. 학습 (val F1 최고 epoch + 그때의 임계값을 best_model.pth에 고정)
python -m src.models.model2_memo.train --config configs/model2_memo.yaml

# 3. 평가 (test 전체, B 포함, 고정 임계값)
python -m src.models.model2_memo.evaluate --config configs/model2_memo.yaml --checkpoint outputs/model2/checkpoints/best_model.pth --metadata data/splits/model1_metadata_eval.csv

# 재촬영은 결과 폴더를 따로 지정해 기본 평가를 덮어쓰지 않게 함
python -m src.models.model2_memo.locate --config configs/model2_memo.yaml --metadata data/splits/model1_recapture.csv
python -m src.models.model2_memo.evaluate --config configs/model2_memo.yaml --checkpoint outputs/model2/checkpoints/best_model.pth --metadata data/splits/model1_recapture.csv --output-dir outputs/model2/predictions_recapture

# 4. 모델 1과 비교 (공통 image_path만 사용, 빠진 행 수를 출력)
python -m src.models.model2_memo.compare --model1 <모델 1 test_predictions.csv> --model2 outputs/model2/predictions/test_predictions.csv
```

평가 CSV에 박스 캐시가 없는 이미지가 있으면 `evaluate`가 `locate --metadata` 안내와 함께 멈춥니다. `--device`는 모든 명령에서 `auto|cuda|mps|cpu`를 받습니다.

## 메모 위치 찾기와 수동 박스

우선순위는 **수동 박스(train만) > 자동 탐지 > 대체(전체 이미지)** 입니다. 탐지는 후처리 버전마다 그 이미지 자체에서 따로 합니다. 이미 캐시에 있는 `image_path`는 건너뛰고, `--overwrite`를 주면 다시 탐지합니다.

수동 박스 파일은 `metadata/memo_boxes_manual.csv`입니다.

```
stem,postprocess,x0,y0,x1,y1,note
```

- 좌표는 EXIF 방향을 적용한 이미지 기준 0~1 정규화 값입니다.
- `postprocess`가 비어 있으면 같은 stem의 original/kakao/resize에 모두 적용됩니다(구도가 같음). recapture는 구도가 다르므로 `recapture`로 따로 적습니다.
- 그 stem에만 적용됩니다. 진짜 사진의 박스를 같은 원본에서 나온 A·B에 옮겨 쓰지 않습니다.
- **train에만 적용됩니다**(`locate.manual_splits: [train]`). val/test까지 사람이 박스를 그려 주면 평가 결과가 실전 성능이 아니라 "사람이 메모를 찾아 준 경우"의 성능이 됩니다. val은 임계값을 정하므로 test와 조건이 같아야 합니다.

`locate_report.csv`는 `split × fake_type × postprocess`별 auto/manual/fallback 비율입니다. 같은 split·postprocess에서 진짜와 어떤 가짜 유형의 fallback 비율 또는 manual 비율이 10%p 넘게 다르면 경고를 출력합니다. 실패나 사람 손이 한쪽 라벨에만 몰리면 모델이 조작 흔적 대신 크롭 방식 차이를 배우기 때문입니다.

## 출력 위치

`outputs/`는 Git에서 제외됩니다(커밋 금지).

| 경로 | 내용 |
|---|---|
| `outputs/model2/memo_boxes.csv` | 박스 캐시 (`image_path, x0..y1, score, source, detector`) |
| `outputs/model2/locate_report.csv` | 탐지 출처 비율 리포트 |
| `outputs/model2/checkpoints/` | `best_model.pth`, `last_model.pth` |
| `outputs/model2/logs/` | 학습 로그 |
| `outputs/model2/predictions/` | `test_predictions.csv` (모델 1 형식 + `postprocess, box_source`), 지표 json/csv |
| `outputs/model2/comparison/` | 모델 1 vs 모델 2 비교표 (csv, md) |

smoke 실행은 같은 구조가 `outputs/model2_smoke/`, 합성 데이터는 `outputs/model2_smoke_data/`에 생깁니다.

## 주의사항

- **예측 파일 형식은 Issue #3 결정을 기다리는 중입니다.** 지금은 모델 1 형식에 `postprocess, box_source`를 더한 형식이며, 결정되면 맞춥니다.
- **실제 사진이 들어오면 20장 정도로 먼저** 탐지 쿼리(`locate.queries`)와 임계값(`box_threshold`, `text_threshold`, `min/max_area_frac`)을 조정하세요. 기본값은 합성 데이터 기준 초기값입니다.
- **유형별로 크롭 샘플을 눈으로 확인하세요.** fallback 비율이 유형 간에 같아도, 탐지가 맞는 정도는 유형별로 다를 수 있습니다(합성 데이터에서 실제로 관찰됨). `locate_report`만으로는 라벨과 엮인 크롭 차이를 놓칠 수 있습니다. real/A/B/C별로 크롭 이미지를 몇 장씩 열어 보세요.
- test는 epoch 선택이나 임계값 조정에 쓰지 않습니다. 임계값은 val에서 정해 체크포인트에 고정됩니다.
- 진짜와 가짜를 서로 다른 방법으로 크롭하거나, 추론 시 마스크·`fake_type` 같은 정답 정보로 위치를 찾지 마세요.

## 테스트

네트워크·GPU 없이 통과합니다. 실제 탐지기 테스트는 `MODEL2_RUN_DETECTOR=1`일 때만 실행됩니다.

```bash
python -m pytest tests/model2 -q
```
