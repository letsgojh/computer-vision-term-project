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
| 가짜 A·B (인페인팅) | `fake{유형}_{원본 stem}_s{seed}.png` | `fakeA_jaehwan_mouse_001_s42.png` |
| 가짜 A·B 마스크 | `{가짜 stem}_mask.png` | `fakeA_jaehwan_mouse_001_s42_mask.png` |
| 가짜 C (text-to-image) | `fakeC_{물건}_{번호3자리}_s{seed}.png` | `fakeC_mouse_001_s7.png` |

- 닉네임·물건 이름은 **영문 소문자 + 숫자**만 씁니다. 공백, 한글, 하이픈(`-`)은 쓰지 않습니다.
- 물건 이름 목록은 `metadata/objects.csv`에 등록된 이름만 씁니다.
- 진짜 사진은 **원본 그대로** 올립니다. 카카오톡으로 전송하면 재압축되므로 금지합니다.

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

`processed/{후처리}/` 아래 파일명은 원래 파일명과 같게 유지합니다. 그래야 `split.csv` 한 장으로 모든 후처리 버전을 찾을 수 있습니다.

이미지 파일은 Git에 올리지 않습니다 (`.gitignore` 처리). 공유 드라이브에 위 구조 그대로 올리고, Git에는 CSV와 코드만 올립니다.

## 3. CSV 형식

### 3-1. `metadata/real_metadata.csv` — 진짜 사진 메타데이터 (전원 작성)

| 컬럼 | 설명 | 예시 |
|---|---|---|
| `filename` | 파일명 | `jaehwan_mouse_001.jpg` |
| `photographer` | 촬영자 닉네임 | `jaehwan` |
| `object` | 물건 | `mouse` |
| `memo_text` | 메모에 쓴 내용 | `2026.11.04 / 로드클` |
| `pen` | 펜 종류 | `ballpoint` / `namepen` / `pencil` / `highlighter` |
| `paper` | 종이 종류 | `postit` / `a4` / `notebook` / `receipt` |
| `lighting` | 조명 | `natural` / `fluorescent` / `dark` |
| `background` | 배경 | `desk` / `floor` / `bed` / `box` |
| `phone_model` | 촬영 기기 | `iPhone15` |
| `width`, `height` | 해상도(px) | `3024`, `4032` |

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
