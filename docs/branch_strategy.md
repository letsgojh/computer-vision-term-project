# 브랜치 · PR 규칙

## 브랜치

`main`은 항상 실행 가능한 상태로 두고, 각자 **자기 이름이 들어간 브랜치**에서 작업한 뒤 PR로 `main`에 합칩니다.

### 이름 규칙

```
feat/<영문이름>-<역할>
```

- `<영문이름>`: 이름+성, 단어 첫 글자만 대문자 (예: `TaeiKim`)
- `<역할>`: 아래 표의 역할 이름을 그대로 씁니다. 영문 소문자와 하이픈(`-`)만 씁니다.
- 한글, 공백, 밑줄(`_`)은 쓰지 않습니다. 한글 브랜치 이름은 URL·터미널(특히 Windows)에서 깨질 수 있습니다.
- `A`·`B`·`C`는 가짜 유형 이름이므로 모델 이름에 붙이지 않습니다 (모델은 `model1`, `model2`).
- 같은 사람이 역할 밖의 작업을 따로 할 때는 역할 자리에 작업 이름을 씁니다 (예: `feat/TaeiKim-model1-export`).

| 역할 | 브랜치 예시 | 주로 건드리는 폴더 |
|---|---|---|
| 가짜 생성 A·B | `feat/<이름>-generation-ab` | `src/generation/`, `metadata/fake_mapping.csv` |
| 가짜 생성 C | `feat/<이름>-generation-c` | `src/generation/`, `metadata/fake_mapping.csv` |
| 모델 1 – 전체 이미지 분류 | `feat/<이름>-model1` | `src/models/model1_full_image/` |
| 모델 2 – 메모 영역 중심 | `feat/<이름>-model2` | `src/models/model2_memo/` |
| 후처리 · split 관리 | `feat/TaeiKim-preprocess-split` | `src/preprocess/`, `data/splits/split.csv` |
| 평가 · 분석 | `feat/<이름>-evaluation` | `src/evaluation/` |

- 진짜 사진 메타데이터(`metadata/real_metadata.csv`)는 전원이 쓰므로, 각자 자기 브랜치에서 **자기 행만 추가**합니다. 기존 행을 고치거나 정렬하지 않아야 PR끼리 충돌이 적습니다.
- 브랜치는 반드시 최신 `main`에서 만듭니다. `main`과 상관없는 새 이력으로 시작하면 PR에서 충돌이 납니다.

  ```bash
  git switch main && git pull origin main
  git switch -c feat/<영문이름>-<역할>
  git push -u origin feat/<영문이름>-<역할>
  ```

- 이름이 규칙과 다른 브랜치는 커밋이 쌓이기 전에 바꾸는 게 쉽습니다. 본인 브랜치만 본인이 바꿉니다.

  ```bash
  git branch -m <새 이름>
  git push -u origin <새 이름>
  git push origin --delete <옛 이름>
  ```

### 공통 규칙

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
