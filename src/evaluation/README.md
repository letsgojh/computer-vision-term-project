# ⑤ 평가 · 분석

**담당:** 평가 · 분석 (1명)

가짜(label 1)를 양성으로 두고, 모델 예측을 `split.csv`에 붙여 유형 × 후처리 결과표와 오류 분석을 만듭니다.

| 입력 | 출력 |
|---|---|
| `outputs/predictions/{model}_{run}.csv` | `outputs/results/report.md` |
| `data/splits/split.csv` | `outputs/results/tables/` |
| `metadata/real_metadata.csv`, `metadata/fake_mapping.csv` | `outputs/results/errors/`, `feedback_weak_types.md` |

예측 CSV의 기본 열은 `filename`, `postprocess`, `prob_fake` 입니다. 모델 브랜치가 쓰는 `image_path`, `fake_probability`도 읽습니다. `postprocess`는 `original` / `kakao` / `resize` / `recapture` 입니다.

정답은 예측 파일이 아니라 `split.csv`를 기준으로 합니다. 유형 A·B·C 칸은 그 유형의 가짜와 진짜만 비교하고, 전체 칸은 모든 가짜를 합칩니다.

## 실행

저장소 루트에서:

```powershell
python -m src.evaluation.run
python -m src.evaluation.run --demo
```

`--demo`는 형식만 보여 주는 합성 데이터입니다. 성능 숫자가 아닙니다. 결과는 `outputs/results/format_preview/`에 씁니다.

실제 예측이 `outputs/predictions/`에 있으면 `--demo` 없이 실행하고, 결과는 `outputs/results/`에 씁니다. 이미지·결과 파일은 `.gitignore` 대상이라 Git에는 코드만 올립니다. 결과표는 공유 드라이브에 올리거나 다시 실행해 만듭니다.

## 결과표

`tables/bundle.md`가 발표 양식입니다.

| 모델 / 유형 | 원본 | 카톡 압축 | 리사이즈 | 재촬영 |
|---|---|---|---|---|
| 모델 1 – A | Acc · P · R · F1 · AUC · n |  |  |  |
| 모델 1 – B |  |  |  |  |
| 모델 1 – C |  |  |  |  |
| 모델 2 – A |  |  |  |  |
| 모델 2 – B |  |  |  |  |
| 모델 2 – C |  |  |  |  |

지표만 따로 붙이려면 `tables/accuracy.md`, `precision.md`, `recall.md`, `f1.md`, `auc.md`를 씁니다. `n`은 가짜/진짜 장수입니다.

판정 기준값은 0.5입니다 (`prob_fake >= 0.5`). AUC와 AP는 기준값과 무관합니다. val 예측이 있으면 val에서 F1이 최대인 기준값을 고른 뒤 test에만 적용한 표를 `tables/f1_val_threshold.md`에 따로 둡니다.

## 같이 보는 값

다섯 지표만으로는 모델이 정말 나누는지 알기 어려워 아래에 같이 적습니다.

| 값 | 볼 때 |
|---|---|
| Specificity | 진짜를 진짜로 두는지. 오탐 규모 |
| Balanced Accuracy | 장수 차이 때문에 Accuracy가 부풀었는지 |
| AP | 가짜가 적어도 순위가 맞는지 |
| Brier, ECE | 확률을 확신도처럼 말해도 되는지 |
| 확률 간격, 점수 히스토그램 | 진짜·가짜 점수가 벌어져 있는지 |
| 유형 평균 Recall | 많은 유형이 전체 점수를 끌고 있는지 |
| 후처리 차이 | 원본 대비 압축·리사이즈·재촬영에서 얼마나 떨어지는지 |
| 모델 1·2 짝비교 | 같은 파일을 누가 맞혔는지. 유형 A를 우선 봄 |
| 메타데이터 집중 | 놓친 가짜·오탐 진짜가 펜·조명·폰 등에 몰리는지 |

## 오류와 피드백

- `errors/fn.csv`: 놓친 가짜
- `errors/fp.csv`: 오탐한 진짜
- `errors/metadata_slices.csv`: 조건별 비율. `concentrated`가 참이면 그 조건에 몰린 것
- `errors/gallery.md`: 칸마다 가장 확신한 오류. 이미지가 `data/processed/{후처리}/`에 있으면 `cases/`로 복사
- `feedback_weak_types.md`: 생성 담당에게 줄 약한 유형. 보강은 train에만 추가

A·B의 촬영 조건은 `fake_mapping.csv`의 원본 파일을 거쳐 `real_metadata.csv`와 연결합니다. C는 대응 원본이 없습니다.

```powershell
python -m unittest src.evaluation.test_pipeline
```
