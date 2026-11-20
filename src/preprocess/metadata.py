"""③ 진짜 사진 메타데이터 자동 생성.

    python -m src.preprocess.metadata [--data-root <data 폴더>]

입력:  data/real/*.jpg, metadata/objects.csv
       파일명 {닉네임}_{물건}_{번호3자리}.jpg (메모 없는 사진은 {닉네임}_{물건}_nomemo_{번호3자리}.jpg)
출력:  metadata/real_metadata.csv

촬영자·물건·메모 유무는 파일명에서, 기종·해상도·촬영 시각은 사진의 EXIF에서 읽는다.
사람이 따로 적는 칸은 없다 (촬영 조건은 기록하지 않기로 함, 이슈 #5).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

from src.common import paths
from src.common.schema import REAL_METADATA_COLUMNS, REAL_NAME
from src.preprocess.postprocess import check_real_files

_ORIENTATION, _MODEL, _DATETIME, _EXIF_IFD, _DATETIME_ORIGINAL = 0x0112, 0x0110, 0x0132, 0x8769, 0x9003


def read_exif(path: Path) -> dict:
    """기종, 회전을 반영한 해상도, 촬영 시각을 읽는다. 없는 값은 빈 문자열."""
    with Image.open(path) as im:
        width, height = im.size
        exif = im.getexif()
    if exif.get(_ORIENTATION) in (5, 6, 7, 8):  # 90도 회전으로 저장된 사진
        width, height = height, width
    taken = exif.get_ifd(_EXIF_IFD).get(_DATETIME_ORIGINAL) or exif.get(_DATETIME) or ""
    if taken:  # "2026:11:17 14:03:21" → "2026-11-17 14:03:21"
        taken = taken.strip("\x00 ").replace(":", "-", 2)
    return {
        "phone_model": str(exif.get(_MODEL, "")).strip("\x00 "),
        "width": width,
        "height": height,
        "taken_at": taken,
    }


def build(data_root: Path, objects=()) -> tuple[pd.DataFrame, list[str], list[str]]:
    """(메타데이터, 오류, 경고)를 돌려준다. 오류가 있으면 메타데이터를 저장하면 안 된다.

    오류: JPG가 아님, 파일명 규칙 위반, objects.csv에 없는 물건, stem 중복
    경고: EXIF가 없음 (카톡 전송·스크린샷·편집본이면 EXIF가 지워져 원본이 아닐 수 있음)
    """
    errors = check_real_files(data_root)
    bad_files = {m.split(":", 1)[0] for m in errors}
    rows, warnings, stems = [], [], {}
    for p in sorted((data_root / "real").glob("*")):
        if p.name.startswith(".") or not p.is_file() or p.name in bad_files:
            continue
        m = REAL_NAME.match(p.name)
        if not m:
            errors.append(
                f"{p.name}: 파일명 규칙 위반 ({{닉네임}}_{{물건}}_{{번호3자리}}.jpg 또는 "
                f"{{닉네임}}_{{물건}}_nomemo_{{번호3자리}}.jpg, 영문 소문자·숫자만)"
            )
            continue
        if objects and m["object"] not in objects:
            errors.append(f"{p.name}: '{m['object']}'은(는) objects.csv에 없는 물건")
            continue
        if p.stem in stems:
            errors.append(f"{p.name}: {stems[p.stem]}와 이름이 겹침 (확장자만 다름)")
            continue
        stems[p.stem] = p.name
        info = read_exif(p)
        if not info["phone_model"] and not info["taken_at"]:
            warnings.append(f"{p.name}: EXIF 없음 (카톡 전송본·스크린샷·편집본인지 확인)")
        rows.append({
            "filename": p.name, "photographer": m["photographer"], "object": m["object"],
            "memo": 0 if m["nomemo"] else 1, **info,
        })
    return pd.DataFrame(rows, columns=REAL_METADATA_COLUMNS), errors, warnings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=paths.DATA)
    ap.add_argument("--objects", type=Path, default=paths.OBJECTS_CSV)
    ap.add_argument("--out", type=Path, default=paths.REAL_METADATA_CSV)
    args = ap.parse_args(argv)

    objects = set(pd.read_csv(args.objects, dtype=str)["object"]) if args.objects.exists() else set()
    meta, errors, warnings = build(args.data_root, objects)
    for w in warnings:
        print(f"경고: {w}")
    if errors:
        print(f"오류 {len(errors)}건 (고친 뒤 다시 실행하세요):")
        print("\n".join(f"  - {e}" for e in errors))
        sys.exit(1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(args.out, index=False)
    print(f"저장: {args.out} ({len(meta)}장)")
    if len(meta):  # 물건마다 메모 있음 25장 + 메모 없음 5장인지 확인용
        memo = meta["memo"].map({1: "메모 있음", 0: "메모 없음"})
        print(pd.crosstab(meta["object"], memo, margins=True, margins_name="합계").to_string())
        print(pd.crosstab(meta["photographer"], memo, margins=True, margins_name="합계").to_string())


if __name__ == "__main__":
    main()
