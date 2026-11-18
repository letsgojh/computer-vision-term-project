"""규약에 맞는 더미 데이터를 만든다. 실제 데이터가 오기 전에 후처리·split 파이프라인을 시험하는 용도.

    python -m src.preprocess.make_dummy <출력 폴더>

<출력 폴더>/data 와 <출력 폴더>/metadata 가 생기며, 다른 명령에 --data-root, --real-metadata 등으로 넘긴다.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src.common.schema import FAKE_MAPPING_COLUMNS, fake_ab_name, fake_c_name, mask_name
from src.preprocess import metadata

# 물건 이름은 metadata/objects.csv의 20종에서 고른다
PHOTOGRAPHERS = {
    "jaehwan": ["mouse", "tumbler", "wallet"],
    "taei": ["keyboard", "earbuds", "smartwatch"],
    "dongsu": ["book", "backpack", "laptop"],
    "minji": ["sneakers", "headphones", "cosmetics"],
}


def _noise_image(rng: np.random.Generator, size) -> Image.Image:
    w, h = size
    base = rng.integers(0, 256, size=3)
    arr = np.clip(base + rng.normal(0, 25, size=(h, w, 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def make(root: Path, per_object: int = 4, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    data, meta = root / "data", root / "metadata"
    for d in ["real", "fake/A", "fake/B", "fake/C", "fake/masks"]:
        (data / d).mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)

    fake_rows = []
    for person, objects in PHOTOGRAPHERS.items():
        for obj in objects:
            for i in range(1, per_object + 1):
                name = f"{person}_{obj}_{i:03d}.jpg"
                size = (480, 640) if i % 2 else (640, 480)
                img = _noise_image(rng, size)
                exif = Image.Exif()
                exif[0x0110] = "DummyPhone"  # Model
                exif.get_ifd(0x8769)[0x9003] = f"2026:11:17 10:{i:02d}:00"  # DateTimeOriginal
                img.save(data / "real" / name, "JPEG", quality=90, exif=exif)
                # 원본마다 A·B 가짜 하나씩 (마지막 원본의 B는 3주차 보강분으로 표시)
                for t in "AB":
                    fake = fake_ab_name(t, name, seed=i)
                    _noise_image(rng, (512, 512)).save(data / "fake" / t / fake)
                    Image.new("L", (512, 512), 255).save(data / "fake" / "masks" / mask_name(fake))
                    fake_rows.append({
                        "fake_filename": fake, "fake_type": t, "source_filename": name,
                        "mask_filename": mask_name(fake), "object": obj, "prompt": "dummy",
                        "negative_prompt": "", "seed": i, "model": "dummy", "created_by": "dummy",
                        "is_reinforcement": int(t == "B" and i == per_object),
                    })
            for i in range(1, 3):
                fake = fake_c_name(obj, i, seed=i)
                _noise_image(rng, (512, 512)).save(data / "fake" / "C" / fake)
                fake_rows.append({
                    "fake_filename": fake, "fake_type": "C", "source_filename": "", "mask_filename": "",
                    "object": obj, "prompt": "dummy", "negative_prompt": "", "seed": i,
                    "model": "dummy", "created_by": "dummy", "is_reinforcement": 0,
                })

    # 진짜 메타데이터는 실제와 같은 방식(파일명·EXIF 자동 생성)으로 만든다
    real_meta, errors, _ = metadata.build(data)
    assert not errors, errors
    real_meta.to_csv(meta / "real_metadata.csv", index=False)
    with open(meta / "fake_mapping.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FAKE_MAPPING_COLUMNS)
        w.writeheader()
        w.writerows(fake_rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--per-object", type=int, default=4)
    args = ap.parse_args(argv)
    make(args.root, args.per_object)
    print(f"더미 데이터 생성: {args.root}")


if __name__ == "__main__":
    main()
