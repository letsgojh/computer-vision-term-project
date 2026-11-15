# 모델 2 – 메모 영역 중심 분류기 설계

- 작성일: 2026-11-16
- 브랜치: `feat/JaehwanYoo-model2` (최신 `main`에서 분기)
- 관련: Issue #1(역할·주차별 체크리스트), #3(CSV·예측 형식 통일 논의), PR #2(후처리·split)

## 1. 목적과 이유

인증샷 진위 판별에서 **A 유형(메모 글씨만 인페인팅)** 은 이미지 전체에서 아주 작은 영역만 바뀝니다. 모델 1은 이미지 전체를 384px 정사각형으로 줄여서 보기 때문에 이 작은 흔적이 사라질 수 있습니다.

모델 2는 **메모 영역만 찾아서 고해상도로 잘라 분류**합니다. 이렇게 해서 "메모 중심 모델이 A 유형을 더 잘 잡는가"를 모델 1과 비교해 보여주는 것이 목표입니다.

두 모델의 차이가 "전체 이미지 vs 메모 크롭" 하나뿐이도록, 백본·입력 크기·학습 방식·입출력 CSV 형식은 모델 1과 똑같이 맞춥니다.

## 2. 목표 / 비목표 / 반목표

### 목표 (이번 작업 범위)
1. 메모 위치 찾기(`locate.py`): 자동 탐지 + 수동 박스 덮어쓰기 + 실패 시 대체 + 유형별 실패율 리포트
2. 메모 크롭 데이터셋(`dataset.py`): CSV 검증, 크롭, 전처리, 학습용 증강
3. 분류 모델·학습·평가(`model.py`, `train.py`, `evaluate.py`)
4. 모델 1 vs 모델 2 비교표(`compare.py`)
5. 합성 데이터 생성기(`scripts/make_smoke_data.py`)로 **학습 → 저장 → 예측이 끝까지 도는 것**을 확인
6. 테스트(`tests/model2/`)와 README

실제 팀 데이터는 아직 공유 드라이브에 없습니다. 이번 작업은 **실제 데이터가 들어오면 바로 학습할 수 있는 코드**까지가 범위입니다. 우리 데이터 학습은 3주차 일입니다.

### 비목표 (이번에 하지 않음)
- 조작 영역 탐지(마스크를 정답으로 쓰는 segmentation): 4주차 여유 시 추가
- OCR 기반 판별: 한글 손글씨 인식이 약해서 제외
- 모델 1 + 모델 2 결합(앙상블): 4주차 여유 시
- HF 데이터(TIGAS, DiQuID)로 학습: 메모가 없는 일반 AI 이미지라 모델 2에는 의미가 적음. 1주차 점검은 합성 데이터로 대체
- 모델 1·후처리 코드, `docs/data_contract.md`, `src/common/` 수정
- 원격 push, PR 생성 (사용자가 따로 요청할 때)

### 반목표 (돌아가더라도 실패로 보는 것)
- **진짜와 가짜를 서로 다른 방법으로 크롭하는 것.** 예: 가짜는 마스크로 자르고 진짜는 수동 박스로 자르기. 모델이 조작 흔적이 아니라 크롭 방식의 차이를 배웁니다.
- **추론 시 정답 정보(A·B 마스크, `fake_type`, 원본 대응표)로 메모 위치를 찾는 것.** 실전에는 이런 정보가 없습니다.
- **val/test에서 사람이 그린 수동 박스를 쓰는 것.** 그러면 test 결과가 실전 성능이 아니라 "사람이 메모를 찾아 준 경우"의 성능이 됩니다 (4-2).
- test split을 epoch 선택이나 임계값 조정에 쓰는 것
- 이미지·체크포인트·대용량 파일을 Git에 커밋하는 것

## 3. 제약

- **입력 이미지:** 학습·평가에는 `data/processed/{original,kakao,resize,recapture}/{stem}.jpg`만 씁니다 (PR #2 규칙). 진짜는 카메라 JPEG, 가짜는 생성기 PNG라서 원본을 쓰면 파일 형식만으로 구별될 위험이 있기 때문입니다. 모든 버전은 EXIF를 지운 JPEG입니다.
- **입력 CSV:** PR #2의 `python -m src.preprocess.split export`가 만드는 `data/splits/model1_metadata.csv` 형식(데이터 규약 3-5)을 그대로 읽습니다.
  - 컬럼: `image_path, label, fake_type, source_id, split, domain, item_id, photographer, postprocess`
  - 필수: `image_path, label, fake_type, source_id, split`. 나머지는 있으면 씁니다.
  - `postprocess`가 없으면 `image_path`의 `data/processed/{후처리}/` 부분에서 알아내고, 그것도 안 되면 `original`로 봅니다.
- **모델 1이 아직 `main`에 없음:** 모델 1 코드를 import하지 않습니다. 형식과 학습 방식만 맞춰 모델 2 폴더 안에 따로 작성합니다.
- **역할 폴더:** 코드는 `src/models/model2_memo/`, 설정은 `configs/model2_*.yaml`, 테스트는 `tests/model2/`에 둡니다. 역할 폴더 밖에 새로 만드는 파일은 `metadata/memo_boxes_manual.csv`와 이 문서(`docs/model2/`)뿐이며, PR 설명에 이유를 적습니다.
- **루트 파일 충돌 피하기:** PR #2가 루트 `requirements.txt`(pytest 추가)와 `pytest.ini`를 바꾸므로, 모델 2는 루트 파일을 건드리지 않습니다. 추가 의존성은 `src/models/model2_memo/requirements.txt`에 적고, 테스트는 `python -m pytest tests/model2`로 실행합니다.
- **이 PC 환경:** WSL2, RTX 5060 Ti 8GB(CUDA 12.8 이상 wheel 필요), WSL 메모리 7GB, 시스템 Python 3.14. 모델 1과 같은 Python 3.11 + `torch==2.11.0`/`torchvision==0.26.0`(cu128)을 `uv`로 설치합니다. 가상환경은 I/O가 느린 `/mnt/c` 대신 `~/.venvs/cvtp-model2`에 둡니다.
- **테스트는 네트워크·GPU 없이 돌아야 합니다.** 사전학습 가중치나 탐지 모델 다운로드가 필요한 테스트는 환경변수(`MODEL2_RUN_DETECTOR=1`)가 있을 때만 실행합니다.

## 4. 정해진 설계

### 4-1. 라벨과 B 유형 처리 (사용자 결정)
- 이미지 라벨은 그대로입니다: 진짜 `0`, 가짜 A·B·C `1`.
- **B는 학습에서 뺍니다.** B는 메모가 진짜라서 메모 크롭에 "가짜" 라벨을 붙이면 진짜 메모를 가짜로 배우게 되기 때문입니다.
  - 설정 `data.exclude_from_training: [B]`로 조절합니다.
  - 이 제외는 **train과 val 모두**에 적용합니다. val은 best epoch 선택과 임계값 조정에 쓰이므로 학습 목표와 같아야 하기 때문입니다. (구현자 판단으로 정한 값)
  - **test에는 B를 포함해 모든 유형을 넣습니다.** "B는 모델 2가 못 잡는다"는 결과 자체가 모델 1과의 역할 분담 근거가 됩니다.

### 4-2. 메모 위치 찾기 (`locate.py`)
- **방법:** 텍스트로 대상을 지정하는 zero-shot 객체 탐지 모델을 **모든 이미지에 같은 방식으로** 실행합니다.
  - 기본: Grounding DINO tiny (`IDEA-Research/grounding-dino-tiny`, `transformers`의 `AutoModelForZeroShotObjectDetection`)
  - 모델 ID와 쿼리 문장은 설정으로 바꿀 수 있습니다. 실제 사진이 들어오면 20장 정도로 먼저 시험해서 쿼리·임계값을 조정합니다 (이번 범위 밖).
  - 탐지기는 교체할 수 있는 인터페이스(`이미지 → [(box, score)]`)로 감싸서, 테스트에서는 가짜 탐지기를 넣습니다.
- **후처리 버전마다 따로 탐지합니다.** 실전에는 압축·축소된 사진만 있으므로, `kakao`·`resize` 이미지도 그 이미지 자체에서 메모를 찾습니다.
- **박스 고르기:** 점수 임계값을 넘고 면적 비율이 `[min_area_frac, max_area_frac]` 안에 있는 박스 중 점수가 가장 높은 것 하나를 고릅니다.
- **우선순위:** 수동 박스(**train split만**) > 자동 탐지 > 대체(전체 이미지).
  - 수동 박스 파일 `metadata/memo_boxes_manual.csv`(Git 관리): 컬럼 `stem, postprocess, x0, y0, x1, y1, note`. 좌표는 0~1로 정규화합니다.
  - `postprocess`가 비어 있으면 같은 stem의 `original`·`kakao`·`resize`에 모두 적용합니다. 이 세 버전은 구도가 같고 크기만 다르기 때문입니다. `recapture`는 구도가 달라서 `postprocess=recapture`로 따로 적어야 합니다.
  - **수동 박스는 설정 `locate.manual_splits`(기본 `[train]`)에 있는 split의 행에만 적용합니다.** val과 test는 실전과 같게 자동 탐지와 대체만 씁니다. val은 임계값을 정하는 곳이라 test와 조건이 같아야 하기 때문입니다. (구현자 판단으로 정한 값)
  - **수동 박스는 그 stem에만 적용합니다.** 진짜 사진의 수동 박스를 같은 원본에서 나온 A·B에 옮겨 쓰지 않습니다. 옮기려면 원본 대응표가 필요하기 때문입니다. (구현자 판단으로 정한 값)
- **대체:** 탐지에 실패하면 박스를 전체 이미지 `(0, 0, 1, 1)`로 두고 출처를 `fallback`으로 기록합니다.
- **결과 캐시:** `outputs/model2/memo_boxes.csv` (Git 제외, 다시 만들 수 있음). 컬럼은 `image_path, x0, y0, x1, y1, score, source(auto|manual|fallback), detector`입니다. 이미 있는 `image_path`는 건너뛰고, `--overwrite`일 때만 다시 탐지합니다.
- **리포트:** `split × fake_type × postprocess`별로 auto/manual/fallback 비율을 `outputs/model2/locate_report.csv`에 저장합니다.
  - 같은 split·postprocess 안에서 진짜와 어떤 가짜 유형의 **fallback 비율 또는 manual 비율** 차이가 10%p를 넘으면 경고를 출력합니다.
  - 실패나 사람 손이 가짜(또는 진짜)에만 몰리면, 크롭 방식 차이가 라벨과 엮여 지름길이 되기 때문입니다.
- **명령 옵션:** `--metadata`로 설정의 CSV 대신 다른 CSV(예: 평가용 후처리 버전 CSV)를 지정할 수 있습니다. 결과는 같은 `boxes_csv` 캐시에 `image_path` 기준으로 추가됩니다.

### 4-3. 크롭과 전처리 (`dataset.py`)
- 이미지를 열고 EXIF 방향을 적용한 뒤 RGB로 바꿉니다. 정규화 박스를 픽셀 좌표로 바꾸고, 각 변에 박스 크기의 `crop.margin`(기본 0.15)만큼 여백을 더한 뒤 이미지 안으로 자릅니다.
- 크롭을 비율을 유지한 채 `input_size` 정사각형에 letterbox하고(빈 곳은 mean 색), ImageNet mean/std로 정규화합니다. 모델 1의 전처리와 같은 방식입니다.
- **학습 증강:** 박스 위치를 박스 크기의 ±`crop.jitter_shift`(기본 0.05)만큼 옮기고, 크기를 `crop.jitter_scale`(기본 [0.9, 1.1]) 배로 바꿉니다. 탐지 오차에 강해지게 하기 위해서입니다.
  - 좌우 반전은 쓰지 않습니다. 글씨가 뒤집히기 때문입니다.
  - 평가·추론에서는 증강이 없고 결정적입니다.
- **CSV 검증:** 모델 1의 `read_metadata`와 같은 규칙을 적용합니다.
  - 라벨이 0/1인지, `fake_type`이 라벨과 맞는지(`real` ↔ 0, `A/B/C/unknown` ↔ 1), split 값이 맞는지, 이미지 파일이 있는지, 경로가 중복되지 않는지 확인합니다.
  - `source_id`와 `leakage_columns`(기본 `[item_id]`)가 여러 split에 걸치면 오류로 멈춥니다.
- **데이터셋 출력:** `image, label, image_path, fake_type, source_id, postprocess, box_source`

### 4-4. 모델·학습·평가
- **학습·평가 데이터 (모델 1·PR #2와 같은 방식):**
  - **학습과 val은 `original`만 씁니다.** 설정 `data.train_postprocess: [original]`로 train·val 행을 거릅니다.
  - 학습 CSV는 모델 1과 같은 파일(`data/splits/model1_metadata.csv`, PR #2 `split.py export`의 기본 출력 = original만)을 씁니다. 그래야 두 모델이 **같은 행으로 학습**해서, 차이가 "전체 vs 메모 크롭"뿐이 됩니다.
  - 평가는 후처리 버전을 포함해 따로 export한 CSV(예: `--variants original kakao resize`로 만든 `data/splits/model1_metadata_eval.csv`, recapture용 `model1_recapture.csv`)를 `evaluate --metadata`로 넘깁니다. 평가 단계에서는 postprocess로 거르지 않습니다.
- **모델:** `timm` `efficientnet_b0`, 출력 1개(logit), `drop_rate 0.2`, ImageNet 사전학습. 모델 1과 같습니다.
- **학습 (모델 1과 같은 방식):**
  - AdamW(lr 1e-4, wd 0.01), CosineAnnealing, `BCEWithLogitsLoss`, 최대 20 epoch, patience 5, batch 16
  - CUDA에서만 AMP를 쓰고, 장치는 `auto`(CUDA → MPS → CPU)로 고릅니다.
  - **val F1이 가장 좋은 epoch**를 `best_model.pth`로 저장하고, 그 epoch에서 **val F1을 최대로 하는 임계값**을 체크포인트에 고정합니다. `last_model.pth`도 저장합니다.
  - 학습 단계에서는 test를 읽지 않습니다.
  - 출력은 `outputs/model2/{checkpoints,logs,predictions}`이고, 이미 체크포인트가 있으면 덮어쓰지 않고 멈춥니다.
- **평가:**
  - 체크포인트에 고정된 임계값으로 test split 전체(B 포함, CSV에 있는 모든 후처리 버전)를 예측합니다.
  - 기본 저장 위치는 `outputs/model2/predictions/test_predictions.csv`입니다. `--output-dir`로 바꿀 수 있어서, recapture 평가가 기본 평가 결과를 덮어쓰지 않게 할 수 있습니다.
  - 컬럼은 모델 1 형식에 두 개를 더한 것입니다: `image_path, label, fake_type, fake_probability, prediction, source_id, domain, postprocess, box_source`
  - 지표(accuracy, precision, recall, F1, ROC-AUC)는 전체, `fake_type`별, `postprocess`별로 계산해 json/csv로 저장합니다.
  - 예측 파일 형식은 #3에서 논의 중입니다. 지금은 모델 1 형식을 따르고, 결정되면 맞춥니다.

### 4-5. 비교표 (`compare.py`)
- 모델 1과 모델 2의 `test_predictions.csv`를 `image_path`로 합칩니다. 두 파일에 모두 있는 행만 쓰고, 빠진 행 수를 출력합니다.
- `fake_type × postprocess` 칸마다 두 모델의 값을 나란히 놓습니다:
  - 가짜 칸: 탐지율(recall, 각 모델의 고정 임계값 기준)과, 같은 postprocess의 진짜와 비교한 ROC-AUC
  - 진짜 칸: 오탐률(FPR)
- CSV와 Markdown 표로 `outputs/model2/comparison/`에 저장합니다. A 유형 요약 한 줄을 맨 위에 둡니다.
- 공식 결과표는 평가 담당이 만듭니다. 이 표는 모델 2 산출물인 "모델 1과의 비교표"용입니다.

### 4-6. 합성 데이터 (`scripts/make_smoke_data.py`)
- 목적은 실제 데이터 없이 **학습 → 저장 → 예측 → 비교가 끝까지 도는지** 확인하는 것입니다. 성능 수치는 의미가 없습니다.
- 생성 내용:
  - 배경(노이즈 + 그라데이션), "제품" 사각형, "메모"(밝은 종이 사각형 + 날짜·닉네임 글씨)가 있는 이미지
  - A는 메모 글씨 영역을 다른 글씨 + 약한 블러·색 변화로 바꾸고, B는 제품 영역을 바꾸고, C는 전체를 다른 방식(더 매끈한 텍스처)으로 생성합니다. 모델이 배울 수 있는 차이가 있어야 합니다.
- 출력:
  - `OUT/data/processed/{original,kakao,resize}/{stem}.jpg` (kakao는 JPEG 품질을 낮춰서, resize는 축소해서)
  - `OUT/model1_metadata.csv` (규약 3-5 형식, `source_id`와 `item_id` 단위로 split이 겹치지 않음)
  - `OUT/memo_boxes.csv` (정답 박스, `source=manual`) — 합성 데이터에서는 탐지기 없이 학습할 수 있게 합니다.

### 4-7. 설정
- `configs/model2_memo.yaml`: 팀 데이터용. `data`(metadata, root, leakage_columns, exclude_from_training), `locate`(boxes_csv, manual_csv, model_id, queries, box_threshold, text_threshold, min/max_area_frac), `crop`(margin, jitter), `model`, `training`(모델 1과 같은 키, `output_dir: outputs/model2`)
- `configs/model2_smoke.yaml`: 합성 데이터용. epoch·입력 크기를 작게 잡습니다.

### 4-8. 명령 (모두 프로젝트 루트에서)
```bash
python -m src.models.model2_memo.scripts.make_smoke_data outputs/model2_smoke_data
python -m src.models.model2_memo.locate   --config configs/model2_memo.yaml [--metadata <CSV>] [--overwrite]
python -m src.models.model2_memo.train    --config configs/model2_memo.yaml [--device ...]
python -m src.models.model2_memo.evaluate --config configs/model2_memo.yaml --checkpoint outputs/model2/checkpoints/best_model.pth [--metadata <평가용 CSV>] [--output-dir <폴더>]
python -m src.models.model2_memo.compare  --model1 <모델1 test_predictions.csv> --model2 <모델2 test_predictions.csv>
python -m pytest tests/model2
```

## 5. 구현자에게 맡기는 것
- 함수·클래스 이름, 파일 안의 구조, 유틸 분리 방식
- 탐지 쿼리 문장과 임계값의 초기값 (설정으로 바꿀 수 있으면 됨)
- 합성 데이터의 구체적인 그림 방식과 장수 (테스트가 빠르게 돌 정도로 작게)
- 로그 출력 형식, 학습 곡선 그림 여부
- 커밋 단위 (작업 단계마다 로컬 커밋, push 없음)

## 6. 열린 문제 (이번 범위 밖, 팀 확인 필요)
- **#3 예측 파일 형식:** 모델 2도 모델 1 형식을 쓸 수 있다고 #3에 답하기 (사용자가 직접)
- **A·B 생성 방식:** SDXL inpainting 결과에 원본 픽셀을 다시 붙여 넣는지 A·B 담당에게 확인하기. 붙여 넣지 않으면 사진 전체(B의 메모 포함)에 VAE 흔적이 남아서 4-1의 B 제외 근거가 약해집니다.
- **가짜 해상도:** SDXL 출력(예: 1024px)과 진짜 원본(예: 3024×4032)의 해상도·비율이 다르면 그것만으로 구별될 수 있습니다. 생성·후처리 담당과 확인해야 합니다.
