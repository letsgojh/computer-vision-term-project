# 데이터 규약 (Data Contract)

> 각자의 산출물이 다음 사람의 입력이 됩니다. 이 문서의 파일명·CSV 형식을 기준으로 맞춰 주세요.
> 1주차 회의에서 확정되면 상단 상태를 **확정**으로 바꾸고, 이후 변경은 PR로만 합니다.

**상태: 초안 (1주차 확정 필요)**

코드에서는 `src/common/schema.py`에 같은 내용이 상수로 정의되어 있습니다. 문서를 바꾸면 코드도 같이 바꿔 주세요.

---

## 1. 파일명 규칙

| 대상 | 형식 | 예시 |
|---|---|---|
| 진짜 사진 | `{닉네임}_{물건}_{번호3자리}.jpg` | `jaehwan_mouse_001.jpg` |
| 진짜 사진 (메모 없음) | `{닉네임}_{물건}_nomemo_{번호3자리}.jpg` | `jaehwan_mouse_nomemo_001.jpg` |
| 가짜 A·B (인페인팅) | `fake{유형}_{원본 stem}_s{seed}.png` | `fakeA_jaehwan_mouse_001_s42.png`, `fakeB_jaehwan_mouse_nomemo_001_s42.png` |
| 가짜 A·B 마스크 | `{가짜 stem}_mask.png` | `fakeA_jaehwan_mouse_001_s42_mask.png` |
| 가짜 C (text-to-image) | `fakeC_{물건}_{번호3자리}_s{seed}.png` | `fakeC_mouse_001_s7.png` |
| 가짜 C (메모 없음) | `fakeC_{물건}_nomemo_{번호3자리}_s{seed}.png` | `fakeC_mouse_nomemo_001_s7.png` |

- 닉네임·물건 이름은 **영문 소문자 + 숫자**만 씁니다. 공백, 한글, 하이픈(`-`)은 쓰지 않습니다.
- 물건 이름 목록은 `metadata/objects.csv`에 등록된 이름만 씁니다.
- 진짜 사진은 **원본 그대로, JPG로** 올립니다. 카카오톡으로 전송하면 재압축되므로 금지합니다. 아이폰은 설정 → 카메라 → 포맷 → 높은 호환성으로 바꿔야 JPG로 찍힙니다.
- 촬영자와 물건은 파일명에서 읽습니다. 파일명이 규칙과 다르면 메타데이터를 만들 수 없습니다.
- 메모 없는 사진은 번호 앞에 `nomemo_`를 붙입니다. 물건마다 메모 있는 사진 25장(`001`~`025`), 메모 없는 사진 5장(`nomemo_001`~`nomemo_005`)입니다 (이슈 #5).
  - 가짜 B는 원본 이름을 따라가므로 `nomemo_`가 저절로 붙습니다. 가짜 C는 생성할 때 붙입니다.
  - 가짜 A는 메모를 고치는 유형이라 메모 없는 원본으로 만들지 않습니다. 대응표에 있으면 split 생성이 멈춥니다.
  - 메모 유무는 `src.common.schema.has_memo(파일명)`으로 진짜·가짜 모두 판단할 수 있습니다.

## 2. 폴더 구조 (`data/`)

```
data/
├── real/                    # ① 진짜 원본 (전원)
├── fake/
│   ├── A/  B/  C/           # ② 가짜 이미지
│   └── masks/               # ② A·B 마스크
├── processed/               # ③ 후처리 결과 (진짜·가짜 모두 동일 적용)
│   ├── original/
│   ├── kakao/
│   ├── resize/
│   └── recapture/
├── external/                # HF 데이터셋 (DiQuID, TIGAS 등)
└── splits/split.csv         # ③ 확정 후 변경 금지
```

`processed/{후처리}/` 아래 파일은 **원래 파일의 stem + `.jpg`** 입니다 (예: `fakeA_jaehwan_mouse_001_s42.png` → `processed/kakao/fakeA_jaehwan_mouse_001_s42.jpg`). `split.csv`의 `filename`에서 확장자만 `.jpg`로 바꾸면 모든 후처리 버전을 찾을 수 있습니다.

진짜는 카메라 JPEG(EXIF 포함), 가짜는 생성기 PNG라서 그대로 학습하면 모델이 이미지 내용이 아니라 파일 형식으로 맞힐 수 있습니다. 그래서 `original`을 포함한 모든 후처리 결과는 **EXIF를 지우고 같은 방식으로 다시 저장한 JPEG**입니다. 모델 학습·평가에는 `data/real/`, `data/fake/`가 아니라 `data/processed/`만 씁니다.

이미지 파일은 Git에 올리지 않습니다 (`.gitignore` 처리). 구글 드라이브는 **사진 저장용으로만** 씁니다.

- 드라이브: `data/` 아래 이미지 폴더(`real/`, `fake/`, `processed/`, `external/`)만 위 구조 그대로 둡니다. Colab에서는 공유 폴더를 "드라이브에 바로가기 추가"한 뒤 `CVTP_DATA_ROOT`를 그 안의 `data/`로 지정합니다.
- Git: `data/splits/`의 CSV, `metadata/`, 코드. `CVTP_DATA_ROOT`를 바꿔도 CSV는 항상 레포 안의 것을 씁니다.
- 드라이브·구글 폼 링크는 공개 레포(이슈, README, 코드)에 쓰지 않습니다.

## 3. CSV 형식

### 3-1. `metadata/real_metadata.csv` — 진짜 사진 메타데이터 (자동 생성)

촬영자가 따로 쓰지 않습니다. 후처리 담당이 `python -m src.preprocess.metadata`로 파일명과 사진의 EXIF에서 만듭니다. 펜·종이·조명·배경 같은 촬영 조건은 기록하지 않기로 했습니다 (이슈 #5).

| 컬럼 | 설명 | 출처 | 예시 |
|---|---|---|---|
| `filename` | 파일명 | 파일 | `jaehwan_mouse_001.jpg` |
| `photographer` | 촬영자 닉네임 | 파일명 | `jaehwan` |
| `object` | 물건 | 파일명 | `mouse` |
| `memo` | 메모가 있으면 1, 없으면 0 | 파일명 (`nomemo_`) | `1` |
| `phone_model` | 촬영 기기 | EXIF | `iPhone 15` |
| `width`, `height` | 해상도(px), 회전 반영 | 이미지·EXIF | `3024`, `4032` |
| `taken_at` | 촬영 시각 | EXIF | `2026-11-17 14:03:21` |

- 파일명 규칙 위반, JPG가 아닌 파일, `objects.csv`에 없는 물건이 있으면 오류 목록을 보여 주고 저장하지 않습니다.
- EXIF가 없는 사진은 경고합니다. 카톡 전송본·스크린샷·편집본은 EXIF가 지워져 원본이 아닐 수 있습니다.

### 3-1-1. `metadata/objects.csv` — 물건 목록

| 컬럼 | 설명 | 예시 |
|---|---|---|
| `object` | 파일명에 쓰는 영문 이름 | `smartphone` |
| `name_ko` | 한국어 이름 | `휴대폰` |

이슈 #5 계획은 24종이고, 지금은 확정된 20종이 등록되어 있습니다. 추가 4종은 정해지면 넣습니다.
- 이름은 가짜 C 생성 코드와 같아야 합니다. 자동차(`car`)와 추가 4종은 C 코드에 아직 없습니다.
- `nomemo`는 물건 이름으로 쓸 수 없습니다 (메모 없는 사진 표시와 겹침).

### 3-2. `metadata/fake_mapping.csv` — 가짜 ↔ 원본 대응표 (생성 담당)

| 컬럼 | 설명 |
|---|---|
| `fake_filename` | 가짜 파일명 |
| `fake_type` | `A` / `B` / `C` |
| `source_filename` | 원본 진짜 파일명 (C는 빈칸) |
| `mask_filename` | 마스크 파일명 (C는 빈칸) |
| `object` | 물건 |
| `prompt`, `negative_prompt` | 생성 프롬프트 |
| `seed` | 시드 |
| `model` | 사용 모델 (예: `sdxl-inpainting-0.1`) |
| `created_by` | 생성 담당 닉네임 |
| `is_reinforcement` | 3주차 이후 보강용이면 `1` (train에만 추가) |

### 3-3. `data/splits/split.csv` — 학습/평가 분할 (후처리 담당)

| 컬럼 | 설명 |
|---|---|
| `filename` | 파일명 (진짜·가짜 공통) |
| `label` | `0` = 진짜, `1` = 가짜 |
| `fake_type` | `real` / `A` / `B` / `C` |
| `object` | 물건 — **split은 물건 단위로 나눔** |
| `group_id` | 같은 원본에서 나온 진짜·가짜를 묶는 키 (진짜 원본 stem, C는 자기 stem) |
| `photographer` | 촬영자 — 사람 단위 test 분리 시 사용 |
| `split` | `train` / `val` / `test` |

- 같은 `group_id`는 반드시 같은 `split`에 있어야 합니다. 이를 어기면 데이터 누수가 생깁니다.
- 2주차 말에 확정한 뒤에는 행을 **추가만** 할 수 있습니다. 보강 데이터는 `train`으로만 추가합니다.

### 3-4. `outputs/predictions/{model}_{run}.csv` — 예측 결과 (모델 담당 → 평가 담당)

| 컬럼 | 설명 |
|---|---|
| `filename` | 파일명 |
| `postprocess` | `original` / `kakao` / `resize` / `recapture` |
| `prob_fake` | 가짜일 확률 (0~1) |

평가 담당은 `filename`으로 `split.csv`와 조인해서 유형별 × 후처리별 결과표를 만듭니다.

### 3-5. `data/splits/model1_metadata.csv` — 모델 1 학습 입력 (후처리 담당, `split.py export`로 생성)

`split.csv`에서 자동으로 만드는 파생 파일이라 Git에 올리지 않습니다. 형식 통일은 이슈 #3에서 논의 중입니다.

| 컬럼 | 설명 | `split.csv`의 대응 |
|---|---|---|
| `image_path` | `data/processed/{후처리}/{stem}.jpg` | `filename` |
| `label` | `0` / `1` | `label` |
| `fake_type` | `real` / `A` / `B` / `C` (외부 데이터는 `unknown`) | `fake_type` |
| `source_id` | 같은 원본 묶음 | `group_id` |
| `split` | `train` / `val` / `test` | `split` |
| `domain` | `trade` (우리 인증샷), 외부 데이터는 `general` | — |
| `item_id` | 물건 | `object` |
| `photographer` | 촬영자 (C는 빈칸) | `photographer` |
| `postprocess` | 후처리 버전 | — |

- 후처리 버전마다 한 행이므로, 같은 사진의 `original`과 `kakao`는 `image_path`만 다르고 `source_id`·`split`은 같습니다.
- 촬영자 컬럼은 모델 1이 누수 검사하는 `photographer_id`가 아니라 `photographer`로 씁니다. split이 물건 단위라서 한 촬영자의 물건이 여러 split에 나뉘기 때문입니다. 촬영자 단위로 분리하려면 `configs/preprocess.yaml`의 `test_photographers`를 쓰세요.
