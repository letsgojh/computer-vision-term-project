# ④ 모델 1 – 전체 이미지 분류

**담당:** 모델 1 (1명)

| 입력 | 출력 |
|---|---|
| `data/external/` (1주차, DiQuID·TIGAS), `data/processed/` + `split.csv` (2주차~) | `outputs/checkpoints/model1_*`, `outputs/predictions/model1_{run}.csv` |

CNN 또는 ViT 파인튜닝 (`timm`). 예측 파일 형식은 `docs/data_contract.md` 3-4를 따릅니다.
비교 실험: HF 데이터만 학습 vs 우리 데이터로 학습.
