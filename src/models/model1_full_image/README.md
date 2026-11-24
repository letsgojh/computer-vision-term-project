# 모델 1 — AI 생성 이미지 판별 (전체 이미지)

중고거래 물건 사진이 **AI로 생성된 이미지인지** 판단합니다. 결과는 세 가지입니다.

- AI 생성 확률 (0~1)
- 3단계 판정: 통과 / 의심(사람이 확인) / AI
- AI 판정 시 Grad-CAM 히트맵

| 항목 | 내용 |
|---|---|
| 담당 | 모델 1 (브랜치 `feat/GiwonCha-model1`) |
| 현재 버전 | **최종: CLIP ViT-L/14(고정) + 선형 분류기**, 공개 데이터(Defactify, FraudBench) + 팀 데이터로 학습 (`scripts/clip_probe.py`, `clip_detector.py`) |
| 개발 과정 | v1~v4 CNN, CLIP, 결합 실험의 구조, 결과, 한계, 변경 이유는 [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md)에 있습니다 |
| 알려진 한계 | 처음 보는 기종의 진짜 폰 사진 중 AI 확률 30% 미만은 88.6%입니다(목표 90%). "AI"로 판정된 것은 219장 중 1장입니다. 학습하지 않은 생성기는 75%만 "의심" 이상으로 잡습니다(목표 80%). 1차 선별기로 쓰고 모델 2와 함께 판단하세요([DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md)의 "시스템 차원의 정리") |

모델 1은 사진 전체를 봅니다. 메모 영역 탐지, OCR, 조작 영역 분할은 모델 2의 범위입니다.

---

## 1. 환경 설치

Python 3.11을 권장합니다. 아래 명령은 모두 **레포 루트**에서 실행합니다.

### Windows + NVIDIA GPU

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r src/models/model1_full_image/requirements.txt gradio
.\.venv\Scripts\python.exe -m src.models.model1_full_image.scripts.check_gpu
```

### Linux

`python3.11 -m venv .venv`, `source .venv/bin/activate`로 환경을 만들고 활성화한 뒤, 위와 같은 `pip install`을 실행합니다.

### Apple Silicon Mac (MPS)

arm64 Python 3.11에서 `pip install torch torchvision`을 기본 인덱스로 설치합니다. CUDA용 lock 파일은 Mac에서 쓰지 않습니다.

`check_gpu`는 패키지 import, 실제 GPU 연산, 장치 이름을 확인합니다. 설정의 `training.device: auto`는 **CUDA → MPS → CPU** 순서로 장치를 고릅니다. `--device cuda|mps|cpu`로 직접 지정할 수도 있습니다. CUDA에서는 학습에 AMP(float16)를 쓰고, 검증·평가·추론은 항상 float32로 합니다.

`requirements-lock.txt`에는 Windows/CUDA 검증 환경의 정확한 버전이 기록되어 있습니다.

## 2. 데이터 위치

이미지, 체크포인트, 결과 파일은 Git에 올리지 않습니다. 데이터는 레포 밖 폴더에 두고, 환경변수로 위치를 알려 줍니다.

```powershell
$env:CVTP_DATA_ROOT = "D:\cvtp-data"            # Defactify 등 학습 데이터
$env:TIGAS_ROOT     = "D:\tigas-download-root"  # TIGAS 교차 테스트용 (선택)
```

설정 파일의 `${CVTP_DATA_ROOT}`는 실행할 때 이 값으로 바뀝니다. 값이 비어 있으면 바로 오류가 납니다.

## 3. 데모 실행

학습된 체크포인트와 `thresholds.json`이 필요합니다. 둘 다 Git에 없으므로 공유 드라이브에서 받아 같은 폴더 구조로 둡니다.

**최종 모델 (CLIP)**: `clip_head.pt`와 같은 폴더의 `thresholds.json`을 씁니다. CLIP 가중치(약 1.7GB)는 처음 실행할 때 Hugging Face에서 내려받습니다(`pip install transformers` 필요).

```powershell
.\.venv\Scripts\python.exe -m app.model1_demo `
  --checkpoint outputs/model1/clip_probe_v3data_3crop/clip_head.pt `
  --version "CLIP (최종)"
```

CLIP은 224×224 정사각형만 입력받습니다. 그래서 사진의 짧은 변을 224로 줄인 뒤, 가운데와 긴 변 양쪽 끝의 224 크롭 3개로 사진 전체를 보고 세 점수(로짓)를 평균합니다. `thresholds.json`의 `"views": 3`이 이 방식을 켭니다. 히트맵도 세 크롭 위치에 그려 사진 전체를 덮습니다.

**이전 CNN 모델 (v2)**:

```powershell
.\.venv\Scripts\python.exe -m app.model1_demo `
  --checkpoint outputs/model1/defactify_v2_noise_aug/checkpoints/best_model.pth `
  --version "v2 — v1 + 노이즈·블러·샤프닝 증강"
```

브라우저에서 http://127.0.0.1:7860 을 엽니다.

- 사진을 올리고 **판별**을 누르면 AI 확률과 3단계 판정이 나옵니다. "AI"로 판정되면 히트맵도 나옵니다.
- 슬라이더로 의심 임계값과 AI 임계값을 바로 바꿀 수 있습니다. **임계값 저장**을 누르면 `thresholds.json`에 기록됩니다.
- CNN 모델은 사진의 짧은 변을 512로 맞춘 뒤, 겹치는 256×256 조각으로 전체를 덮어 점수를 평균냅니다. 조각별 Grad-CAM은 가장자리를 부드럽게 섞어 이어 붙입니다.

히트맵은 모델이 판단에 주목한 영역일 뿐, 생성·조작된 위치의 증거가 아닙니다. 해석 시 주의점은 [XAI.md](XAI.md)를 참고하세요.

단일 이미지를 명령줄에서 예측할 수도 있습니다.

```powershell
.\.venv\Scripts\python.exe -m src.models.model1_full_image.predict --image <사진.jpg> --checkpoint <best_model.pth>
```

판정에는 체크포인트에 저장된 단일 임계값(Real/Fake)을 씁니다. 3단계 판정은 데모에서 보세요.

`native_crop` 모델(v1~)은 데모와 같은 방식으로 채점합니다. 짧은 변을 512로 맞추고 겹치는 크롭으로 사진 전체를 덮어 점수를 평균냅니다. 그래서 같은 사진이면 데모와 같은 점수가 나옵니다.

`--explain`을 붙이면 Grad-CAM 결과물을 `outputs/explanations/` 아래에 저장합니다.
- 저장되는 파일은 `explanation.png`, `overlay.png`, `heatmap_values.npy`, `explanation.json`입니다.
- `native_crop` 모델은 크롭별 히트맵을 이어 붙여 원본 사진 크기에 맞춥니다. letterbox 모델은 여백을 제외하고 맞춥니다.
- `explanation.json`의 `geometry` 항목에 어떤 방식으로 맞췄는지 기록됩니다.

## 4. v2 재현 (학습부터 평가까지)

```powershell
# 1) Defactify 내려받기 (약 7.5GB)
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Rajarshi-Roy-research/Defactify_Image_Dataset', repo_type='dataset', local_dir=r'$env:CVTP_DATA_ROOT\defactify', allow_patterns=['data/*'])"

# 2) 이미지 파일과 metadata.csv로 변환 (DALL-E 3, Midjourney는 test 전용)
.\.venv\Scripts\python.exe -m src.models.model1_full_image.scripts.prepare_defactify --source $env:CVTP_DATA_ROOT/defactify --output $env:CVTP_DATA_ROOT/defactify_model1 --root $env:CVTP_DATA_ROOT/defactify_model1

# 3) 지름길 검사: 픽셀 없이 크기 등 메타데이터만으로 맞힐 수 있는지
.\.venv\Scripts\python.exe -m src.models.model1_full_image.scripts.audit_shortcuts --metadata $env:CVTP_DATA_ROOT/defactify_model1/metadata.csv --root $env:CVTP_DATA_ROOT/defactify_model1

# 4) 학습 (RTX 4070 SUPER 기준 약 50분)
.\.venv\Scripts\python.exe -m src.models.model1_full_image.train --config configs/model1_defactify_v2.yaml

# 5) 3단계 임계값 계산 -> outputs/model1/defactify_v2_noise_aug/thresholds.json
$CK = "outputs/model1/defactify_v2_noise_aug/checkpoints/best_model.pth"
.\.venv\Scripts\python.exe -m src.models.model1_full_image.scripts.calibrate_thresholds --checkpoint $CK --config configs/model1_defactify_v2.yaml

# 6) 테스트 평가 (원본 해상도 / 크기 맞춤) + 학습 생성기 vs hold-out 생성기 분석
$O = "outputs/model1/defactify_v2_noise_aug"
.\.venv\Scripts\python.exe -m src.models.model1_full_image.evaluate --checkpoint $CK --config configs/model1_defactify_v2.yaml --output-dir $O/eval_native
.\.venv\Scripts\python.exe -m src.models.model1_full_image.evaluate --checkpoint $CK --config configs/model1_defactify_v2.yaml --output-dir $O/eval_resize512 --eval-resize-short 512
.\.venv\Scripts\python.exe -m src.models.model1_full_image.scripts.analyze_generalization --predictions $O/eval_native/test_predictions.csv --thresholds $O/thresholds.json --output $O/eval_native

# 7) TIGAS 교차 테스트 (선택, 학습에 없던 데이터셋)
.\.venv\Scripts\python.exe -m src.models.model1_full_image.evaluate --checkpoint $CK --config configs/model1_tigas_crosstest.yaml --output-dir $O/eval_tigas
```

TIGAS를 받는 방법은 [TIGAS_DOWNLOAD.md](TIGAS_DOWNLOAD.md)에 있습니다. 교차 테스트용 `metadata.csv`는 모든 행을 `split=test`로 하고, `caption_id` 열을 포함해야 합니다(누수 검사용).

## 5. 설정 파일

| 설정 | 용도 |
|---|---|
| `configs/model1_defactify_v2.yaml` | 이전 CNN 버전(v2). v1과 같고, 진짜·가짜 모두에 노이즈·블러·샤프닝 증강을 적용합니다 |
| `configs/model1_defactify.yaml` | v1. Defactify, `preprocess: native_crop`, 256 크롭, 생성기 hold-out |
| `configs/model1_tigas_crosstest.yaml` | Defactify로 학습한 모델을 TIGAS로 평가합니다(평가 전용) |
| `configs/model1_tigas_*.yaml` | v0. TIGAS 학습, letterbox 384 |
| `configs/model1_team.yaml` | 팀 데이터(`split.csv` 기반) 학습용 |
| `configs/model1_baseline.yaml`, `configs/model1_smoke.yaml` | 기본값 예시 / 합성 데이터로 코드 동작만 확인 |

전처리는 `model.preprocess`로 고릅니다.
- `letterbox` (기본값, v0): 비율을 유지해 정사각형으로 축소하고 빈 곳을 채웁니다. 데이터의 진짜와 가짜가 크기·비율이 다르면, 채운 띠가 정답 단서가 됩니다.
- `native_crop` (v1~): 리사이즈 없이 원본 픽셀에서 잘라냅니다. 학습 증강은 `model.augment`에서 설정하며, 진짜와 가짜에 같은 확률로 적용됩니다.

## 6. 평가 원칙

- **임계값**: 학습 중 임계값과 best 체크포인트는 **검증셋으로만** 고릅니다. test에서는 체크포인트에 저장된 값을 그대로 쓰고 다시 찾지 않습니다.
- **3단계 판정**(`thresholds.py`): 검증셋으로 계산합니다.
  - 의심 = min(AI 재현율 95% 지점, 진짜 오탐률 5% 지점)
  - AI = 진짜 오탐률 1% 지점
  - 미탐(AI를 진짜로 통과)을 더 위험하게 보는 설정입니다.
- **누수 검사**: `source_id`와 `data.leakage_columns`(Defactify는 `caption_id`)가 split 사이에 겹치면 학습을 거부합니다. 평가 때도 학습·검증에 쓴 ID가 test에 섞이면 거부합니다.
- **일반화 평가**: 같은 데이터셋 점수만으로는 부족합니다.
  - 학습하지 않은 생성기(hold-out)
  - 학습하지 않은 데이터셋(TIGAS)
  - 크기 맞춤(`--eval-resize-short`)
  - 실제 폰 사진

  이 결과를 따로 보고합니다.
- **기록**: 수치는 측정한 것만 [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md)에 출처와 함께 적습니다.

## 7. 팀 데이터로 학습 (`split.csv` 기반)

후처리 담당의 `split.py export`로 `data/splits/model1_metadata.csv`를 만든 뒤 실행합니다. `image_path`는 `data/processed/{original,kakao,resize,recapture}/...`를 가리킵니다. 같은 물건(`item_id`)과 원본(`source_id`)은 같은 split에 있어야 합니다.

```powershell
.\.venv\Scripts\python.exe -m src.models.model1_full_image.scripts.validate_dataset --config configs/model1_team.yaml
.\.venv\Scripts\python.exe -m src.models.model1_full_image.train --config configs/model1_team.yaml
.\.venv\Scripts\python.exe -m src.models.model1_full_image.evaluate --config configs/model1_team.yaml --checkpoint <best_model.pth>
```

CSV 필수 열은 `image_path,label,fake_type,source_id,split`입니다. label은 진짜 0, 가짜 1이고, fake_type은 `real`, `A`, `B`, `C`, `unknown` 중 하나입니다.

## 8. 합성 데이터로 코드 동작 확인

도형과 문자로 만든 **합성 전용** 데이터입니다. 여기서 나온 지표는 탐지 성능이 아니므로, 성능으로 인용하면 안 됩니다.

```powershell
.\.venv\Scripts\python.exe -m src.models.model1_full_image.scripts.create_smoke_test_data
.\.venv\Scripts\python.exe -m unittest discover -s tests/model1 -v
.\.venv\Scripts\python.exe -m src.models.model1_full_image.train --config configs/model1_smoke.yaml
```

smoke 체크포인트를 평가하거나 추론하려면 `--allow-smoke`를 명시해야 합니다.

## 9. 파일 안내

| 경로 (`src/models/model1_full_image/` 기준) | 역할 |
|---|---|
| `dataset.py` | CSV 검증, 누수 검사, 전처리(`letterbox`, `native_crop`과 증강), Dataset |
| `model.py` | timm 백본 + 단일 logit 출력 |
| `train.py` | 학습, 검증 F1로 best 선택, early stopping, 학습 곡선 |
| `evaluate.py` | 저장된 임계값으로 test 평가, 이미지별 예측 저장, 크기 맞춤 옵션 |
| `predict.py` | 이미지 한 장 추론 (`--explain`으로 Grad-CAM 저장) |
| `inference.py` | 사진 전체를 크롭으로 덮어 채점하고 히트맵을 이어 붙입니다(데모가 사용) |
| `thresholds.py` | 3단계 판정과 임계값 계산 |
| `explain.py` | Grad-CAM 계산과 설명 저장 |
| `utils.py` | 설정(환경변수 포함), seed, 장치, 지표, 체크포인트 공통 처리 |
| `scripts/prepare_defactify.py` | Defactify → 이미지 + `metadata.csv`, 생성기 hold-out 분리 |
| `scripts/audit_shortcuts.py` | 메타데이터만으로 정답을 맞힐 수 있는지(지름길) 검사 |
| `scripts/calibrate_thresholds.py` | 검증셋으로 `thresholds.json` 생성 |
| `scripts/analyze_generalization.py` | 학습 생성기와 hold-out 생성기의 성능, 진짜 판정 비율 분석 |
| `scripts/download_tigas.py`, `scripts/visualize_tigas_samples.py` | TIGAS 선택 다운로드와 샘플 확인 |
| `scripts/check_gpu.py`, `check_model.py`, `check_checkpoint.py`, `check_gradcam.py` | 환경, 모델, 체크포인트, Grad-CAM 점검 |
| `scripts/validate_dataset.py`, `create_smoke_test_data.py` | 학습 전 데이터 검증 / 합성 fixture 생성 |
| `../../../app/model1_demo.py` | Gradio 데모 (`clip_head.pt`면 CLIP, 그 외는 CNN) |
| `clip_detector.py` | 최종 CLIP 모델의 점수와 ViT Grad-CAM 히트맵 (데모용) |
| `scripts/clip_probe.py` | CLIP 특징 추출 + 선형 분류기 학습·평가 |
| `../../../tests/model1/` | 단위 테스트 (`python -m unittest discover -s tests/model1`) |
| `DEVELOPMENT_LOG.md` | 버전별 개발 과정, 결과, 한계 |
| `XAI.md`, `VALIDATION.md`, `TIGAS_DOWNLOAD.md` | Grad-CAM 해석 / v0 구현 검증 기록 / TIGAS 다운로드 방법 |

학습 결과는 `training.output_dir` 아래에 저장됩니다(`checkpoints/`, `logs/training.csv`, `figures/`, `config.yaml`, `environment.json`). 실험마다 새 `output_dir`를 쓰세요. 이미 체크포인트가 있는 폴더에는 덮어쓰지 않습니다.
