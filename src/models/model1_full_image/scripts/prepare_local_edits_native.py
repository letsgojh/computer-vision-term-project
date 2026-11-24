"""v6 data: local-edit pairs kept at the photo's *native* resolution.

v5 failed because its pairs were cut at demo scale (short side 512), where the edit trace is gone:
a real Qwen edit's region has ~63% of the surrounding camera noise at native resolution, but the
difference vanishes at short side <= 1024 (DEVELOPMENT_LOG, v5). Same edit simulation as
prepare_local_edits.make_edit, but nothing is resized afterwards:
  train/val: one 320x320 native-resolution crop per edit, centred inside the edited box, and the
             same crop from the original (re-encoded with the same JPEG quality).
  test:      whole native-resolution photos (edited + re-encoded original) for full-image scanning.
Small photos (< 2 MP, e.g. COCO) get larger boxes so a crop is mostly edited.
"""

import argparse
import hashlib
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from src.models.model1_full_image.scripts.prepare_local_edits import TEST_VAES, TRAIN_VAES, load_vaes, make_edit

EDITS = {"team_real": 4, "fraudbench_real": 1, "coco": 1}
TEST_EDITS = {"team_real": 2, "fraudbench_real": 1, "coco": 1}
LIMIT = {"coco": {"train": 800, "val": 200, "test": 200}, "fraudbench_real": {"train": 1399, "val": 196, "test": 200}}
CROP = 320


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", required=True, help="mixed_v3/metadata.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = pd.read_csv(args.metadata, keep_default_na=False)
    reals = meta[meta.source.isin(EDITS)]
    parts = []
    for (source, split), g in reals.groupby(["source", "split"]):
        n = LIMIT.get(source, {}).get(split, len(g))
        parts.append(g.sample(min(n, len(g)), random_state=args.seed))
    reals = pd.concat(parts)
    vaes = load_vaes(TEST_VAES, device)
    out = Path(args.output)
    rows = []
    for item in tqdm(list(reals.itertuples()), desc="native edits", mininterval=30):
        with Image.open(item.image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        w, h = image.size
        area = (0.01, 0.08) if w * h >= 2e6 else (0.15, 0.35)
        test = item.split == "test"
        names = TEST_VAES if test else TRAIN_VAES
        for k in range((TEST_EDITS if test else EDITS)[item.source]):
            name = names[rng.randrange(len(names))]
            edited, box = make_edit(image, vaes[name], name, rng, device, area)
            stem = f"{Path(item.image_path).stem}_{hashlib.md5(item.image_path.encode()).hexdigest()[:8]}_{k}"
            folder = out / item.split / item.source
            folder.mkdir(parents=True, exist_ok=True)
            if test:
                pictures, offset = {0: image, 1: edited}, (0, 0)
            else:
                size = min(CROP, w, h)
                cx, cy = rng.randint(box[0], box[2]), rng.randint(box[1], box[3])
                left, top = min(max(0, cx - size // 2), w - size), min(max(0, cy - size // 2), h - size)
                pictures, offset = {lbl: p.crop((left, top, left + size, top + size)) for lbl, p in [(0, image), (1, edited)]}, (left, top)
            quality = rng.randint(85, 95)
            for label, picture in pictures.items():
                path = folder / f"{stem}_{'edit' if label else 'orig'}.jpg"
                picture.save(path, quality=quality)
                rows.append({"label": label, "split": item.split, "group_id": item.group_id,
                             "domain": f"native_edit:{name}" if label else f"native_orig:{item.source}",
                             "source": f"native_edit_{item.source}" if label else f"native_orig_{item.source}",
                             "generator_group": ("holdout" if name == "movq" else "seen") if label else "real",
                             "image_path": str(path.resolve()), "fake_type": "unknown" if label else "real",
                             "source_id": f"{item.source_id}:{k}", "edit_vae": name if label else "",
                             "edit_box": ",".join(str(v - o) for v, o in zip(box, offset * 2)) if label else ""})
    frame = pd.DataFrame(rows)
    train = frame.split == "train"
    counts = frame[train].groupby("source").size()
    # Each photo source gets the same share inside each class.
    per_class = frame[train].groupby("label").source.nunique()
    frame["sample_weight"] = [1 / per_class[l] / counts[s] if t else 0.0 for s, l, t in zip(frame.source, frame.label, train)]
    frame.to_csv(out / "metadata.csv", index=False)
    print(pd.crosstab([frame.split, frame.domain], frame.label))


if __name__ == "__main__":
    main()
