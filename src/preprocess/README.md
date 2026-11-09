# ③ 후처리 · split

**담당:** 후처리 · split 관리 (1명)

| 입력 | 출력 |
|---|---|
| `data/real/`, `data/fake/`, `metadata/*.csv` | `data/processed/{original,kakao,resize,recapture}/`, `data/splits/split.csv` |

설정은 [`configs/preprocess.yaml`](../../configs/preprocess.yaml)에 있습니다.

## 후처리 — `postprocess.py`

```bash
python -m src.preprocess.postprocess run                    # original·kakao·resize 일괄 생성 (이미 있으면 건너뜀)
python -m src.preprocess.postprocess recapture <폴더>        # 화면 재촬영 사진 넣기 (파일명 = 원본 stem)
python -m src.preprocess.postprocess inspect <이미지> ...     # 해상도·JPEG 품질·EXIF 확인
```

| 후처리 | 하는 일 |
|---|---|
| `original` | EXIF 제거 + 회전 반영 + JPEG(q95) 재저장. 진짜·가짜의 파일 형식 차이를 없앰 |
| `kakao` | 긴 변 축소 + 낮은 품질 JPEG. 카톡 전송본 흉내 |
| `resize` | 짧은 변 축소. 마켓 업로드 리사이즈 흉내 |
| `recapture` | 모니터에 띄운 사진을 폰으로 다시 찍은 것. 사람이 찍어서 `recapture` 명령으로 넣음 |

**카톡 설정값 정하기:** 진짜 사진 몇 장을 실제 카톡으로 보내고(일반 화질), 받은 파일에 `inspect`를 돌려 나온 해상도와 품질을 `configs/preprocess.yaml`의 `kakao`에 적습니다.

## split — `split.py`

```bash
python -m src.preprocess.split build      # 2주차: 처음 만들기 (이미 행이 있으면 거부, --force로 덮어쓰기)
python -m src.preprocess.split append     # 3주차~: 새 행을 train에만 추가
python -m src.preprocess.split check      # 누수·형식 점검 + 유형별 장수 표
```

- 물건 단위로 나누고, 같은 `group_id`(같은 원본에서 나온 진짜·가짜)는 같은 split에 둡니다.
- `configs/preprocess.yaml`의 `test_photographers`에 닉네임을 넣으면 그 사람의 물건은 모두 test로 갑니다.
- 보강 데이터(`is_reinforcement=1`)는 train에만 들어갑니다. 원본이나 물건이 val/test에 있는 보강분은 넣으면 누수가 되므로 **건너뛰고 목록을 출력**합니다. 생성 담당에게 train 원본으로만 보강해 달라고 전달하세요.

## 더미 데이터로 미리 돌려보기

```bash
python -m src.preprocess.make_dummy /tmp/cvtp_dummy
python -m src.preprocess.postprocess --data-root /tmp/cvtp_dummy/data run
python -m src.preprocess.split --real-metadata /tmp/cvtp_dummy/metadata/real_metadata.csv \
    --fake-mapping /tmp/cvtp_dummy/metadata/fake_mapping.csv \
    --split-csv /tmp/cvtp_dummy/data/splits/split.csv build
python -m pytest            # 테스트
```
