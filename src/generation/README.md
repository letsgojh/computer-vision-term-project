# ② 가짜 생성

**담당:** 가짜 생성 A·B (1명), 가짜 생성 C (1명)

| | 입력 | 출력 |
|---|---|---|
| A·B (인페인팅) | `data/real/`, 마스크 | `data/fake/A/`, `data/fake/B/`, `data/fake/masks/` |
| C (text-to-image) | 프롬프트 템플릿 | `data/fake/C/` |
| 공통 | — | `metadata/fake_mapping.csv`에 한 장당 한 행 |

파일명은 `src/common/schema.py`의 `fake_ab_name()`, `mask_name()`, `fake_c_name()`으로 만듭니다.
3주차 이후 보강 데이터는 `is_reinforcement=1`로 표시합니다.
