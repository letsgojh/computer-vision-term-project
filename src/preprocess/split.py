"""③ train / val / test 분할.

    python -m src.preprocess.split build [--force]   # 2주차: 처음 만들기
    python -m src.preprocess.split append            # 3주차~: 새 행을 train에만 추가
    python -m src.preprocess.split check             # 누수 점검

입력:  metadata/real_metadata.csv, metadata/fake_mapping.csv
출력:  data/splits/split.csv

규칙 (docs/data_contract.md):
- 물건 단위로 나눈다. 같은 물건은 한 split에만 있다.
- 같은 원본에서 나온 진짜·가짜(group_id)는 같은 split에 있다.
- 확정 후에는 행 추가만 하고, 추가되는 행은 train으로만 간다.
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.common import paths
from src.common.schema import FAKE_TYPES, SPLIT_COLUMNS, SPLITS


def _stem(filename: str) -> str:
    return filename.rsplit(".", 1)[0]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def build_rows(real_meta: pd.DataFrame, fake_map: pd.DataFrame) -> pd.DataFrame:
    """메타데이터 시트와 대응표로 split 열을 뺀 행들을 만든다."""
    real = pd.DataFrame({
        "filename": real_meta["filename"],
        "label": "0",
        "fake_type": "real",
        "object": real_meta["object"],
        "group_id": real_meta["filename"].map(_stem),
        "photographer": real_meta["photographer"],
        "is_reinforcement": "0",
    })

    by_source = real_meta.set_index("filename")
    fake_rows = []
    for r in fake_map.itertuples(index=False):
        if r.fake_type not in FAKE_TYPES:
            raise ValueError(f"{r.fake_filename}: fake_type이 {FAKE_TYPES} 중 하나가 아님 ({r.fake_type!r})")
        if r.fake_type == "C":
            group, photographer, obj = _stem(r.fake_filename), "", r.object
        else:
            if r.source_filename not in by_source.index:
                raise ValueError(f"{r.fake_filename}: 원본 {r.source_filename!r}이 real_metadata.csv에 없음")
            src = by_source.loc[r.source_filename]
            if r.object and r.object != src["object"]:
                raise ValueError(f"{r.fake_filename}: object {r.object!r}가 원본의 {src['object']!r}와 다름")
            group, photographer, obj = _stem(r.source_filename), src["photographer"], src["object"]
        fake_rows.append({
            "filename": r.fake_filename,
            "label": "1",
            "fake_type": r.fake_type,
            "object": obj,
            "group_id": group,
            "photographer": photographer,
            "is_reinforcement": r.is_reinforcement or "0",
        })
    rows = pd.concat([real, pd.DataFrame(fake_rows, columns=real.columns)], ignore_index=True)
    dup = rows["filename"][rows["filename"].duplicated()]
    if len(dup):
        raise ValueError(f"파일명 중복: {sorted(set(dup))[:10]}")
    return rows


def assign_objects(rows: pd.DataFrame, ratios, seed: int, test_photographers=()) -> dict[str, str]:
    """물건 → split. 지정한 촬영자의 물건은 test로 보내고, 나머지는 장수 비율에 맞춰 배정한다."""
    counts = rows.groupby("object").size()
    held = set(rows.loc[rows["photographer"].isin(test_photographers), "object"])
    assignment = {obj: "test" for obj in held}

    total = counts.sum()
    target = {s: r * total for s, r in zip(SPLITS, ratios)}
    current = {s: counts[list(held)].sum() if s == "test" else 0 for s in SPLITS}

    rest = sorted(set(counts.index) - held)
    random.Random(seed).shuffle(rest)
    # 큰 물건부터 넣어야 비율이 덜 틀어진다
    rest.sort(key=lambda o: -counts[o])
    for obj in rest:
        split = max(SPLITS, key=lambda s: (target[s] - current[s]) / target[s] if target[s] else float("-inf"))
        assignment[obj] = split
        current[split] += counts[obj]
    return assignment


def build(rows: pd.DataFrame, ratios, seed: int, test_photographers=()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """처음 split을 만든다. (결과, 건너뛴 행)을 돌려준다.

    보강 데이터(is_reinforcement=1)는 train에만 들어갈 수 있어서, 원본이 val/test에 간 보강분은 빠진다.
    """
    base = rows[rows["is_reinforcement"] != "1"]
    obj_split = assign_objects(base, ratios, seed, test_photographers)
    out = base.assign(split=base["object"].map(obj_split))[SPLIT_COLUMNS]
    out, skipped = append(out, rows[rows["is_reinforcement"] == "1"])
    return out.reset_index(drop=True), skipped


def append(existing: pd.DataFrame, rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """확정된 split에 새 행을 train으로만 추가한다. (결과, 건너뛴 행)을 돌려준다.

    이미 val/test에 있는 물건이나 원본 그룹에 속한 새 행은 train에 넣으면 누수가 되므로 건너뛴다.
    """
    new = rows[~rows["filename"].isin(existing["filename"])].copy()
    group_split = existing.drop_duplicates("group_id").set_index("group_id")["split"]
    obj_split = existing.drop_duplicates("object").set_index("object")["split"]

    where = new["group_id"].map(group_split).fillna(new["object"].map(obj_split)).fillna("train")
    new["split"] = where
    new["skip_reason"] = ""
    blocked = where != "train"
    new.loc[blocked, "skip_reason"] = "같은 물건/원본이 " + where[blocked] + "에 있음"

    added = new.loc[~blocked, SPLIT_COLUMNS]
    return pd.concat([existing, added], ignore_index=True), new.loc[blocked, ["filename", "skip_reason"]]


def check(split: pd.DataFrame) -> list[str]:
    """누수와 형식 오류를 찾는다. 문제가 없으면 빈 리스트."""
    errors = []
    missing = set(SPLIT_COLUMNS) - set(split.columns)
    if missing:
        return [f"컬럼 없음: {sorted(missing)}"]

    for col, allowed in [("split", SPLITS), ("fake_type", ("real",) + FAKE_TYPES), ("label", ("0", "1"))]:
        bad = split.loc[~split[col].isin(allowed), "filename"]
        if len(bad):
            errors.append(f"{col} 값 오류 {len(bad)}건: {list(bad[:5])}")
    mismatch = split[(split["label"] == "0") != (split["fake_type"] == "real")]
    if len(mismatch):
        errors.append(f"label과 fake_type 불일치 {len(mismatch)}건: {list(mismatch['filename'][:5])}")

    dup = split["filename"][split["filename"].duplicated()]
    if len(dup):
        errors.append(f"파일명 중복 {len(dup)}건: {list(dup[:5])}")

    for key, name in [("group_id", "같은 원본 그룹"), ("object", "같은 물건")]:
        spread = split.groupby(key)["split"].nunique()
        leaked = list(spread[spread > 1].index)
        if leaked:
            errors.append(f"{name}이 여러 split에 걸침 {len(leaked)}건: {leaked[:5]}")
    return errors


def summary(split: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(split["split"], split["fake_type"], margins=True, margins_name="total")
    order = [s for s in SPLITS if s in table.index] + ["total"]
    cols = [c for c in ("real",) + FAKE_TYPES if c in table.columns] + ["total"]
    return table.loc[order, cols]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=paths.ROOT / "configs" / "preprocess.yaml")
    ap.add_argument("--real-metadata", type=Path, default=paths.REAL_METADATA_CSV)
    ap.add_argument("--fake-mapping", type=Path, default=paths.FAKE_MAPPING_CSV)
    ap.add_argument("--split-csv", type=Path, default=paths.SPLIT_CSV)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_build = sub.add_parser("build", help="split.csv 처음 만들기")
    p_build.add_argument("--force", action="store_true", help="이미 행이 있는 split.csv를 덮어쓰기")
    sub.add_parser("append", help="새 행을 train에만 추가")
    sub.add_parser("check", help="누수·형식 점검")
    args = ap.parse_args(argv)

    if args.cmd in ("build", "append"):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)["split"]
        rows = build_rows(_read_csv(args.real_metadata), _read_csv(args.fake_mapping))

    if args.cmd == "build":
        if args.split_csv.exists() and len(_read_csv(args.split_csv)) and not args.force:
            sys.exit(f"{args.split_csv}에 이미 행이 있습니다. 확정 후에는 append를 쓰고, 정말 다시 만들려면 --force.")
        result, skipped = build(rows, cfg["ratios"], cfg["seed"], cfg.get("test_photographers") or [])
    elif args.cmd == "append":
        result, skipped = append(_read_csv(args.split_csv), rows)
    else:
        result, skipped = _read_csv(args.split_csv), []

    if len(skipped):
        print(f"건너뛴 행 {len(skipped)}건 (train에 넣으면 누수):")
        print(skipped.to_string(index=False))

    errors = check(result)
    if errors:
        print("\n".join(["점검 실패:"] + [f"  - {e}" for e in errors]))
        sys.exit(1)
    if args.cmd != "check":
        args.split_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.split_csv, index=False)
        print(f"저장: {args.split_csv} ({len(result)}행)")
    print(summary(result).to_string())


if __name__ == "__main__":
    main()
