# 브랜치 · PR 규칙

## 브랜치

`main`을 중심으로 역할별 브랜치에서 작업하고, PR로 `main`에 합칩니다.

| 브랜치 | 역할 | 주로 건드리는 폴더 |
|---|---|---|
| `main` | 항상 실행 가능한 상태 유지 | — |
| `feat/data-collection` | 전원 – 진짜 촬영·메타데이터 | `metadata/real_metadata.csv`, `metadata/objects.csv` |
| `feat/generation-ab` | 가짜 생성 A·B | `src/generation/`, `metadata/fake_mapping.csv` |
| `feat/generation-c` | 가짜 생성 C | `src/generation/`, `metadata/fake_mapping.csv` |
| `feat/preprocess-split` | 후처리 · split 관리 | `src/preprocess/`, `data/splits/split.csv` |
| `feat/model1-full-image` | 모델 1 – 전체 이미지 분류 | `src/models/model1_full_image/` |
| `feat/model2-memo` | 모델 2 – 메모 영역 중심 | `src/models/model2_memo/` |
| `feat/evaluation` | 평가 · 분석 | `src/evaluation/` |

- 자기 역할 폴더 밖의 파일, 특히 `src/common/`과 `docs/data_contract.md`를 바꿀 때는 PR 설명에 이유를 적고 관련 담당자를 리뷰어로 지정합니다.
- 작업 전에 `git pull origin main`(또는 rebase)으로 최신 `main`을 반영합니다.

## 커밋 메시지

```
<type>: <요약>
```

`type`은 `feat`(기능), `fix`(버그), `data`(CSV·데이터 규약), `docs`(문서), `chore`(설정) 중 하나를 씁니다.

## 올리면 안 되는 것

- 이미지, 모델 체크포인트, 대용량 파일 → 공유 드라이브에 올립니다.
- API 키, HF 토큰 → `.env`에 두고 커밋하지 않습니다.
