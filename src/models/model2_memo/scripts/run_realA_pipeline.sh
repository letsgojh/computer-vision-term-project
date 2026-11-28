#!/usr/bin/env bash
# 실험 1: 드라이브 진짜 사진 + C(팀원) + Qwen A(불량 69장 제외) → 학습·테스트·Grad-CAM·지름길 검사. 전부 로컬.
set -euo pipefail
W=${CVTP_WORK:-$HOME/cvtp_work}           # bad_old.txt, bad_new.txt 위치
PP=${CVTP_PP:-$W/pp}                      # feat/TaeiKim-preprocess-split(PR #2) 체크아웃
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
D=${CVTP_DATA:-$HOME/cvtp_data}
PY=${PY:-python}
OUT=outputs/model2_realAC2
BK=$D/backup_before_A2; mkdir -p $BK

echo "== 0. 백업"
cp --update=none $D/metadata/fake_mapping.csv $BK/ ; cp --update=none $D/data/splits/*.csv $BK/

echo "== 1. A 사진·마스크 합치기 (불량 제외)"
mkdir -p $D/data/fake/A $D/data/fake/masks
cp --update=none $D/fakeA_qwen/data/fake/A/*.png $D/data/fake/A/
cp --update=none $D/fakeA_qwen/data/fake/masks/*.png $D/data/fake/masks/
$PY - <<EOF
import pandas as pd, pathlib
m = pd.read_csv("$D/metadata/fake_mapping.csv", dtype=str)
a = pd.read_csv("$D/fakeA_qwen/metadata/fake_mapping_A.csv", dtype=str)[m.columns]
m = pd.concat([m[~m.fake_filename.isin(a.fake_filename)], a])
bad = set(pathlib.Path("$W/bad_old.txt").read_text().split()) | set(pathlib.Path("$W/bad_new.txt").read_text().split())
stem = m.source_filename.fillna("").str.replace(".jpg", "", regex=False)
print("불량 A 제외:", int(((m.fake_type == "A") & stem.isin(bad)).sum()))
m = m[~((m.fake_type == "A") & stem.isin(bad))]
m.to_csv("$D/metadata/fake_mapping.csv", index=False)
print("fake_mapping:", m.fake_type.value_counts().to_dict())
EOF

cd $D
echo "== 2. 후처리 (original·kakao·resize, 있는 건 건너뜀)"
PYTHONPATH=$PP $PY -m src.preprocess.postprocess --config $PP/configs/preprocess.yaml --data-root $D/data run

echo "== 3. split에 A 추가 (원본과 같은 split)"
PYTHONPATH=$PP $PY - <<PYEOF
import pandas as pd
from src.preprocess.split import build_rows, _read_csv, SPLIT_COLUMNS
sp = _read_csv("data/splits/split.csv")
rows = build_rows(_read_csv("metadata/real_metadata.csv"), _read_csv("metadata/fake_mapping.csv"))
new = rows[~rows.filename.isin(sp.filename) & (rows.fake_type == "A")].copy()
new["split"] = new.group_id.map(sp.drop_duplicates("group_id").set_index("group_id")["split"])
assert new.split.notna().all(), "원본 split을 못 찾은 A가 있음"
sp = pd.concat([sp, new[SPLIT_COLUMNS]], ignore_index=True)
keep = set(rows.filename)
before = len(sp); sp = sp[(sp.fake_type != "A") | sp.filename.isin(keep)]
print("split에서 뺀 A:", before - len(sp))
sp.to_csv("data/splits/split.csv", index=False)
print("최종 A split:", sp[sp.fake_type=="A"].split.value_counts().to_dict())
PYEOF
PYTHONPATH=$PP $PY -c "from src.preprocess.split import check,_read_csv; e=check(_read_csv('data/splits/split.csv')); print('누수 점검:', e or '이상 없음'); assert not e"
PYTHONPATH=$PP $PY -m src.preprocess.split --config $PP/configs/preprocess.yaml --split-csv $D/data/splits/split.csv export --out $D/data/splits/model1_metadata.csv --data-root $D
PYTHONPATH=$PP $PY -m src.preprocess.split --config $PP/configs/preprocess.yaml --split-csv $D/data/splits/split.csv export --variants original kakao resize --out $D/data/splits/model1_eval.csv --data-root $D

cd $REPO
echo "== 4. 설정"
mkdir -p $OUT
[ -f $OUT/memo_boxes.csv ] || cp outputs/model2_realAC/memo_boxes.csv $OUT/memo_boxes.csv
sed -e "s#outputs/model2_realC#$OUT#g" outputs/model2_realC/config_realC.yaml > $OUT/config.yaml

echo "== 5. 메모 위치 (새 A만 계산)"
$PY -m src.models.model2_memo.locate --config $OUT/config.yaml --metadata $D/data/splits/model1_eval.csv 2>&1 | grep -v "it/s\]" | tail -20

echo "== 6. 학습"
$PY -m src.models.model2_memo.train --config $OUT/config.yaml 2>&1 | grep -v "HF Hub" | tee $OUT/train.log

echo "== 7. 테스트 평가"
$PY -m src.models.model2_memo.evaluate --config $OUT/config.yaml --checkpoint $OUT/checkpoints/best_model.pth \
    --metadata $D/data/splits/model1_eval.csv 2>&1 | grep -v "HF Hub" | tee $OUT/eval.log

echo "== 8. Grad-CAM"
$PY -m src.models.model2_memo.explain --config $OUT/config.yaml --checkpoint $OUT/checkpoints/best_model.pth \
    --metadata $D/data/splits/model1_eval.csv --per-group 6 --output-dir $OUT/gradcam 2>&1 | tail -3

echo "== 9. 지름길 대조 실험"
$PY -m src.models.model2_memo.scripts.shortcut_tests --config $OUT/config.yaml \
    --checkpoint $OUT/checkpoints/best_model.pth --data-root $D 2>&1 | grep -v "HF Hub" | tail -15 || true

echo "== 끝"
cat $OUT/predictions/test_rates.csv 2>/dev/null || true
