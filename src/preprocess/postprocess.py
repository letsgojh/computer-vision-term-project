"""③ 후처리 실행.

    python -m src.preprocess.postprocess run [--variants original kakao resize] [--overwrite]
    python -m src.preprocess.postprocess recapture <재촬영 사진 폴더>
    python -m src.preprocess.postprocess inspect <이미지> [...]

입력:  data/real/*.jpg, data/fake/{A,B,C}/*.png
출력:  data/processed/{original,kakao,resize,recapture}/{원본 stem}.jpg
"""

import argparse
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

from src.common import paths
from src.preprocess import transforms as T

AUTO_VARIANTS = ("original", "kakao", "resize")  # recapture는 사람이 찍어서 따로 넣는다
REAL_EXTS = {".jpg", ".jpeg"}
FAKE_EXTS = {".png"}


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def find_sources(data_root: Path) -> dict[str, Path]:
    """stem → 원본 경로. 진짜와 가짜 A·B·C를 모두 모은다 (마스크 제외)."""
    sources: dict[str, Path] = {}
    folders = [(data_root / "real", REAL_EXTS)] + [(data_root / "fake" / t, FAKE_EXTS) for t in "ABC"]
    for folder, exts in folders:
        for p in sorted(folder.glob("*")):
            if p.suffix.lower() not in exts:
                continue
            if p.stem in sources:
                raise ValueError(f"stem 중복: {p} / {sources[p.stem]}")
            sources[p.stem] = p
    return sources


def apply_variant(img: Image.Image, variant: str, cfg: dict) -> tuple[Image.Image, int]:
    """(저장할 이미지, 저장 품질)을 돌려준다."""
    if variant == "original":
        return img, cfg["jpeg_quality"]
    if variant == "kakao":
        return T.resize_long_side(img, cfg["kakao"]["long_side"]), cfg["kakao"]["quality"]
    if variant == "resize":
        return T.resize_short_side(img, cfg["resize"]["short_side"]), cfg["jpeg_quality"]
    raise ValueError(f"알 수 없는 후처리: {variant}")


def run(data_root: Path, cfg: dict, variants, overwrite: bool = False) -> dict[str, int]:
    sources = find_sources(data_root)
    written = {v: 0 for v in variants}
    for stem, src in tqdm(sources.items(), desc="postprocess"):
        targets = {v: data_root / "processed" / v / f"{stem}.jpg" for v in variants}
        todo = {v: p for v, p in targets.items() if overwrite or not p.exists()}
        if not todo:
            continue
        img = T.load_image(src)
        for v, out in todo.items():
            out_img, quality = apply_variant(img, v, cfg)
            T.save_jpeg(out_img, out, quality)
            written[v] += 1
    return written


def ingest_recapture(data_root: Path, src_dir: Path, cfg: dict) -> tuple[int, list[str]]:
    """화면 재촬영 사진을 processed/recapture로 넣는다. 파일명은 원본 stem과 같아야 한다."""
    known = find_sources(data_root)
    unknown = []
    count = 0
    for p in sorted(src_dir.glob("*")):
        if p.suffix.lower() not in REAL_EXTS | FAKE_EXTS:
            continue
        if p.stem not in known:
            unknown.append(p.name)
            continue
        T.save_jpeg(T.load_image(p), data_root / "processed" / "recapture" / f"{p.stem}.jpg", cfg["jpeg_quality"])
        count += 1
    return count, unknown


def inspect(files) -> None:
    for f in files:
        with Image.open(f) as im:
            size, fmt, has_exif = im.size, im.format, bool(im.getexif())
        print(f"{Path(f).name}\t{fmt}\t{size[0]}x{size[1]}\tquality≈{T.estimate_jpeg_quality(f)}\texif={has_exif}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=paths.DATA)
    ap.add_argument("--config", type=Path, default=paths.ROOT / "configs" / "preprocess.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="original·kakao·resize 일괄 생성")
    p_run.add_argument("--variants", nargs="+", choices=AUTO_VARIANTS, default=list(AUTO_VARIANTS))
    p_run.add_argument("--overwrite", action="store_true")

    p_rec = sub.add_parser("recapture", help="화면 재촬영 사진 넣기")
    p_rec.add_argument("src_dir", type=Path)

    p_ins = sub.add_parser("inspect", help="해상도·JPEG 품질·EXIF 확인 (카톡 설정값 측정용)")
    p_ins.add_argument("files", nargs="+", type=Path)

    args = ap.parse_args(argv)
    if args.cmd == "inspect":
        inspect(args.files)
        return
    cfg = load_config(args.config)
    if args.cmd == "run":
        for v, n in run(args.data_root, cfg, args.variants, args.overwrite).items():
            print(f"{v}: {n}장 생성")
    elif args.cmd == "recapture":
        n, unknown = ingest_recapture(args.data_root, args.src_dir, cfg)
        print(f"recapture: {n}장 추가")
        if unknown:
            print(f"원본 stem과 맞지 않아 건너뜀 {len(unknown)}장: {', '.join(unknown[:10])}")


if __name__ == "__main__":
    main()
