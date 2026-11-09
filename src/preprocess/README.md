# ③ 후처리 · split

**담당:** 후처리 · split 관리 (1명)

| 입력 | 출력 |
|---|---|
| `data/real/`, `data/fake/`, `metadata/*.csv` | `data/processed/{original,kakao,resize,recapture}/`, `data/splits/split.csv` |

- 후처리는 진짜·가짜 모두에 똑같이 적용합니다.
- split은 물건 단위로 나누고, 같은 `group_id`(`src/common/schema.py`의 `group_id()`)는 같은 split에 둡니다.
- 2주차 말 확정 후에는 `split.csv`에 행 추가만 허용합니다 (보강 데이터 → `train`).
