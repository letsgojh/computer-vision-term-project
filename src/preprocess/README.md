# ③ 후처리 · split

**담당:** 후처리 · split 관리 (1명)

| 입력 | 출력 |
|---|---|
| `data/real/`, `data/fake/`, `metadata/objects.csv`, `metadata/fake_mapping.csv` | `metadata/real_metadata.csv`, `data/processed/{original,kakao,resize,recapture}/`, `data/splits/split.csv` |

설정은 [`configs/preprocess.yaml`](../../configs/preprocess.yaml)에 있습니다.

## 진짜 사진 메타데이터 — `metadata.py`

```bash
python -m src.preprocess.metadata          # data/real/ → metadata/real_metadata.csv
```

- 촬영자·물건·메모 유무는 파일명에서, 기종·해상도·촬영 시각은 EXIF에서 읽습니다. 사람이 따로 쓰는 칸은 없습니다.
- 저장 후 물건별·촬영자별 메모 있음/없음 장수 표를 보여 줍니다. 물건마다 25장 + 5장인지 확인하세요.
- 파일명 규칙 위반, JPG가 아닌 파일, `metadata/objects.csv`에 없는 물건이 있으면 오류 목록을 보여 주고 저장하지 않습니다. 해당 촬영자에게 고쳐 달라고 전달하세요.
- EXIF가 없는 사진은 경고만 합니다. 카톡 전송본이나 편집본일 수 있으니 원본인지 확인하세요.
- 사진이 새로 올라올 때마다 다시 실행하고 CSV를 커밋합니다.

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

### 모델 1 형식으로 내보내기 — `split.py export`

```bash
python -m src.preprocess.split export                                  # original → data/splits/model1_metadata.csv
python -m src.preprocess.split export --variants original kakao resize # 후처리별 평가용 (버전마다 한 행)
python -m src.preprocess.split export --variants recapture --skip-missing --out data/splits/model1_recapture.csv
```

- `split.csv`를 모델 1의 `metadata.csv` 형식(`image_path,label,fake_type,source_id,split,domain,...`)으로 바꿉니다. 형식은 [`docs/data_contract.md`](../../docs/data_contract.md) 3-5 참고.
- `image_path`는 `data/processed/{후처리}/{stem}.jpg`입니다. 모델 1 설정의 `data.root`를 `data/`가 들어 있는 폴더(로컬은 프로젝트 루트)로 두세요.
- `recapture`는 일부 사진만 있으므로 `--skip-missing`으로 없는 행을 뺍니다.

## 더미 데이터로 미리 돌려보기

```bash
python -m src.preprocess.make_dummy /tmp/cvtp_dummy   # 진짜 메타데이터도 metadata.py로 같이 만듦
python -m src.preprocess.postprocess --data-root /tmp/cvtp_dummy/data run
python -m src.preprocess.split --real-metadata /tmp/cvtp_dummy/metadata/real_metadata.csv \
    --fake-mapping /tmp/cvtp_dummy/metadata/fake_mapping.csv \
    --split-csv /tmp/cvtp_dummy/data/splits/split.csv build
python -m pytest tests/test_preprocess.py   # 후처리·split 테스트 (torch 없이 실행됨)
```

`python -m pytest`만 치면 `tests/` 아래 모델 1·2 테스트까지 모으는데, 그 테스트들은 torch가 필요해서 torch가 없으면 모으는 단계에서 멈춥니다. 후처리만 확인할 때는 위처럼 파일을 지정하세요.
