"""파이프라인 점검용 합성 데이터 생성기 (진짜/A/B/C + 후처리 3종)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.models.model2_memo.boxes import BoxRecord, write_boxes

COLUMNS = ["image_path", "label", "fake_type", "source_id", "split", "domain",
           "item_id", "photographer", "postprocess"]
PAPER = (240, 238, 225)


def _memo_text(img: Image.Image, rect: tuple[int, int, int, int], text: str) -> None:
    """메모 종이를 칠하고 글씨를 종이 크기에 맞춰 선명하게 넣는다."""
    x0, y0, x1, y1 = rect
    ImageDraw.Draw(img).rectangle([x0, y0, x1 - 1, y1 - 1], fill=PAPER)
    font = ImageFont.load_default()
    l, t, r, b = font.getbbox(text)
    mask = Image.new("L", (r - l, b - t), 0)
    ImageDraw.Draw(mask).text((-l, -t), text, fill=255, font=font)
    iw, ih = max(2, (x1 - x0) * 8 // 10), max(2, (y1 - y0) * 4 // 10)
    mask = mask.resize((iw, ih), Image.NEAREST)
    img.paste((20, 20, 20), (x0 + (x1 - x0 - iw) // 2, y0 + (y1 - y0 - ih) // 2), mask)


def _layout(rng, w: int, h: int):
    """메모와 제품 사각형을 서로 다른 좌/우 영역에 겹치지 않게 놓는다."""
    left = bool(rng.integers(2))
    mw = int(w * rng.uniform(0.30, 0.45))
    mh = int(h * rng.uniform(0.30, 0.40))
    pw = int(w * rng.uniform(0.25, 0.38))
    ph = int(h * rng.uniform(0.35, 0.60))

    def place(lo, hi, size, total):
        return lo + int(rng.integers(0, max(1, min(hi, total) - lo - size + 1)))

    mx_lo, px_lo = (int(w * 0.03), int(w * 0.55)) if left else (int(w * 0.52), int(w * 0.03))
    mx = place(mx_lo, mx_lo + int(w * 0.45), mw, w)
    px = place(px_lo, px_lo + int(w * 0.42), pw, w)
    my = place(int(h * 0.05), h - int(h * 0.05), mh, h)
    py = place(int(h * 0.05), h - int(h * 0.05), ph, h)
    return (mx, my, mx + mw, my + mh), (px, py, px + pw, py + ph)


def _scene(rng, w: int, h: int, text: str):
    """노이즈 배경 + 제품 + 메모 장면을 그린다."""
    noise = rng.integers(60, 130, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(noise, "RGB")
    memo, prod = _layout(rng, w, h)
    color = tuple(int(c) for c in rng.integers(40, 200, 3))
    ImageDraw.Draw(img).rectangle([prod[0], prod[1], prod[2] - 1, prod[3] - 1], fill=color)
    _memo_text(img, memo, text)
    return img, memo, prod


def _fake_a(real: Image.Image, memo, i: int, k: int) -> Image.Image:
    img = real.copy()
    _memo_text(img, memo, f"2026.12.{i % 20 + 1:02d} q{(k + 1) % 3}")
    region = img.crop(memo).filter(ImageFilter.GaussianBlur(1.5))
    arr = np.asarray(region, dtype=float) * np.array([1.0, 0.85, 0.65])
    img.paste(Image.fromarray(arr.clip(0, 255).astype(np.uint8)), memo[:2])
    return img


def _fake_b(real: Image.Image, prod) -> Image.Image:
    img = real.copy()
    region = img.crop(prod)
    arr = 255 - np.asarray(region, dtype=np.uint8)
    region = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(2.0))
    img.paste(region, prod[:2])
    return img


def _variants(img: Image.Image) -> dict[str, tuple[Image.Image, int]]:
    w, h = img.size
    s = 0.75
    return {
        "original": (img, 95),
        "kakao": (img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR), 40),
        "resize": (img.resize((max(1, int(w * 0.5)), max(1, int(h * 0.5))), Image.BILINEAR), 95),
    }


def make_smoke_data(out: Path, items: int = 6, per_item: int = 4, c_per_item: int = 2,
                    size: tuple[int, int] = (320, 240), seed: int = 0) -> Path:
    """합성 데이터를 만들고 model1_metadata.csv 경로를 돌려준다."""
    if items < 3:
        raise ValueError("items는 3 이상이어야 한다 (train/val/test 물건 단위 분할)")
    out = Path(out)
    w, h = size
    for post in ("original", "kakao", "resize"):
        (out / "data" / "processed" / post).mkdir(parents=True, exist_ok=True)
    rows, boxes = [], []

    def emit(img, stem, memo, label, ftype, source, split, item, photo):
        nbox = (memo[0] / w, memo[1] / h, memo[2] / w, memo[3] / h)
        for post, (v, q) in _variants(img).items():
            rel = f"data/processed/{post}/{stem}.jpg"
            v.save(out / rel, "JPEG", quality=q)
            rows.append([rel, label, ftype, source, split, "smoke", item, photo, post])
            boxes.append(BoxRecord(image_path=rel, box=nbox, score=1.0,
                                   source="manual", detector="smoke-gt"))

    for i in range(items):
        split = "test" if i == items - 1 else "val" if i == items - 2 else "train"
        k, item = i % 3, f"item{i}"
        for n in range(1, per_item + 1):
            s = seed * 100003 + i * 1000 + n
            rng = np.random.default_rng(s)
            real, memo, prod = _scene(rng, w, h, f"2026.11.04 p{k}")
            stem = f"p{k}_item{i}_{n:03d}"
            emit(real, stem, memo, 0, "real", stem, split, item, f"p{k}")
            emit(_fake_a(real, memo, i, k), f"fakeA_{stem}_s{s}", memo, 1, "A", stem, split, item, f"p{k}")
            emit(_fake_b(real, prod), f"fakeB_{stem}_s{s}", memo, 1, "B", stem, split, item, f"p{k}")
        for n in range(1, c_per_item + 1):
            s = seed * 100003 + i * 1000 + 500 + n
            rng = np.random.default_rng(s)
            img, memo, _ = _scene(rng, w, h, f"2026.11.04 p{k}")
            img = img.filter(ImageFilter.GaussianBlur(1.0))
            stem = f"fakeC_item{i}_{n:03d}_s{s}"
            emit(img, stem, memo, 1, "C", stem, split, item, "")

    csv = out / "model1_metadata.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(csv, index=False)
    write_boxes(boxes, out / "memo_boxes.csv")
    return csv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--items", type=int, default=6)
    p.add_argument("--per-item", type=int, default=4)
    p.add_argument("--c-per-item", type=int, default=2)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    print(make_smoke_data(a.out, a.items, a.per_item, a.c_per_item, (a.width, a.height), a.seed))


if __name__ == "__main__":
    main()
