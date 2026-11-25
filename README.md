# 인증샷 진위 판별 (Computer Vision Term Project)

중고거래 인증샷(물건 + 날짜·닉네임 손글씨 메모)이 실제 촬영인지, 생성형 AI로 만들거나 고친 것인지 판별합니다.

- 역할 분담과 주차별 체크리스트: [Issue #1](https://github.com/letsgojh/computer-vision-term-project/issues/1)
- 파일명·CSV 형식: [docs/data_contract.md](docs/data_contract.md)
- 브랜치·PR 규칙: [docs/branch_strategy.md](docs/branch_strategy.md)

## 가짜 유형

| 유형 | 만드는 방법 | 담당 |
|---|---|---|
| A | 진짜 사진의 **메모 글씨 영역**을 인페인팅 (SDXL inpainting) | 가짜 생성 A·B |
| B | 진짜 사진의 **제품 영역**을 인페인팅 (SDXL inpainting) | 가짜 생성 A·B |
| C | 처음부터 생성 (SDXL text-to-image) | 가짜 생성 C |

## 전체 흐름과 폴더

```
① 진짜 촬영 (전원)          → data/real/ + metadata/real_metadata.csv
② 가짜 생성 (A·B, C)        → data/fake/{A,B,C,masks}/ + metadata/fake_mapping.csv   [src/generation]
③ 후처리 · split            → data/processed/{original,kakao,resize,recapture}/ + data/splits/split.csv   [src/preprocess]
④ 모델 학습 (모델 1, 모델 2) → outputs/checkpoints/, outputs/predictions/   [src/models]
   모델 3은 학습 없이 점수로 판별 → src/models/model3_TrainingFree/
⑤ 평가 · 분석               → outputs/results/   [src/evaluation]
⑥ 약한 유형 보강 → 재학습    → ②로 돌아가 train에만 추가
```

```
.
├── configs/            # 학습·생성 설정 (yaml)
├── data/               # 이미지 (Git 제외, 공유 드라이브에 같은 구조로 보관)
├── docs/               # 규약 문서, 파트별 문서
├── metadata/           # 메타데이터 시트, 원본 대응표, 물건 목록 (Git 관리)
├── notebooks/          # Colab 노트북
├── outputs/            # 체크포인트·예측·결과표 (Git 제외)
└── src/
    ├── common/         # 경로·스키마 (전원 공용)
    ├── generation/     # ② 가짜 생성
    ├── preprocess/     # ③ 후처리 · split
    ├── models/
    │   ├── model1_full_image/      # ④ 전체 이미지 분류
    │   ├── model2_memo/            # ④ 메모 영역 중심
    │   └── model3_TrainingFree/    # 학습 없이 재구성·고주파 점수로 판별
    └── evaluation/     # ⑤ 평가 · 분석
```

## 모델 3 (`src/models/model3_TrainingFree`)

AeroBlade 재구성 오차, RDD 디바이어싱, WaRPAD를 한 검사기로 묶었습니다. 가짜 사진으로 학습하지 않고, 실제 사진 점수 분포와 비교해 판정합니다. 실행 방법은 해당 폴더의 README를 봅니다.

추가된 파일:

- `src/models/model3_TrainingFree/detect.py`
- `src/models/model3_TrainingFree/requirements.txt`
- `src/models/model3_TrainingFree/README.md`
- `src/models/model3_TrainingFree/detector/__init__.py`
- `src/models/model3_TrainingFree/detector/haar.py`
- `src/models/model3_TrainingFree/detector/images.py`
- `src/models/model3_TrainingFree/detector/pipeline.py`
- `src/models/model3_TrainingFree/detector/rdd.py`
- `src/models/model3_TrainingFree/detector/scores.py`
- `src/models/model3_TrainingFree/detector/warpad.py`
- `src/models/model3_TrainingFree/tests/test_scores.py`

## 환경 설정

Python 3.10 이상.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- 로컬(Apple Silicon)에서는 PyTorch `mps` 디바이스를 씁니다.
- Colab에서는 레포를 클론한 뒤 데이터 경로를 드라이브로 지정합니다.

```python
!git clone https://github.com/letsgojh/computer-vision-term-project.git
%cd computer-vision-term-project
!pip install -r requirements.txt
import os; os.environ["CVTP_DATA_ROOT"] = "/content/drive/MyDrive/cvtp/data"
```
