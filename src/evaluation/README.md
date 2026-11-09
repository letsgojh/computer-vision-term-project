# ⑤ 평가 · 분석

**담당:** 평가 · 분석 (1명)

| 입력 | 출력 |
|---|---|
| `outputs/predictions/*.csv`, `data/splits/split.csv`, `metadata/real_metadata.csv` | `outputs/results/` (결과표, 오류 사례) |

- 지표: Accuracy, Precision, Recall, F1, AUC
- 결과표: 유형별(A/B/C) × 후처리별(original/kakao/resize/recapture)
- 오류 분석: 놓친 가짜, 오탐한 진짜 → 메타데이터(펜·종이·조명 등)와 연결해 생성·모델 담당에게 공유
