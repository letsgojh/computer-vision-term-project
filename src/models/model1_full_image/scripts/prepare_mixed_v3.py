"""Build the v3 mixed dataset: Defactify + FraudBench + team phone photos/fakes.

Copies source images under --output (originals are never modified) and writes metadata.csv.
- PNG fakes are re-encoded to JPEG (quality 85-95, seeded) so file format cannot predict the label;
  JPEG inputs are copied byte-for-byte.
- group_id keeps related images in one split: Defactify caption, FraudBench category/review
  (a real photo and the fakes made from it), team photographer+object.
- Held out to test only: team photographers --test-photographers (with the fake-C images of
  their objects), FraudBench generators --holdout-generators, Defactify DALL-E 3 / Midjourney.
- sample_weight balances sources inside each class for the weighted training sampler.
"""

import argparse
import hashlib
import io
import random
import re
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

TEAM_OBJECTS = {"changjun": ["tumbler", "hoodie", "chair", "wallet"], "giwon": ["smartphone", "laptop", "earbuds", "smartwatch"],
                "KimGuenYoung": ["microphone", "muffler", "gloves", "perfume"], "LeeSangYoun": ["gamecontroller", "backpack", "book", "cosmetics"],
                "taei": ["tripod", "desk", "fridge", "lunchbox"], "YooJaehwan": ["headphone", "keyboard", "mouse", "shose"]}
FAKE_OBJECT_ALIASES = {"headphones": "headphone", "sneakers": "shose"}
VAL_OBJECTS = {"wallet", "smartwatch", "perfume", "cosmetics"}
# Target share of each source inside its class for the weighted sampler.
SOURCE_SHARE = {"coco": 0.40, "fraudbench_real": 0.35, "team_real": 0.25,
                "defactify_fake": 0.40, "fraudbench_fake": 0.45, "team_fake": 0.15}


def copy_image(src, dst, rng):
    """Copy JPEG bytes as-is; re-encode anything else as JPEG with a random high quality."""
    dst = dst.with_suffix(".jpg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        if src.suffix.lower() in {".jpg", ".jpeg"}:
            shutil.copyfile(src, dst)
        else:
            with Image.open(src) as image:
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, format="JPEG", quality=rng.randint(85, 95))
            dst.write_bytes(buffer.getvalue())
    return dst


def team_rows(team_dir, extra_dirs, out, test_people, rng):
    rows, seen = [], set()
    folders = {p.name.split("(")[0]: p for p in (team_dir / "real").iterdir() if p.is_dir()}
    sources = [(person, f) for person, folder in folders.items() for f in sorted(folder.rglob("*")) if f.suffix.lower() in {".jpg", ".jpeg"}]
    sources += [("giwon", f) for d in extra_dirs for f in sorted(Path(d).glob("*")) if f.suffix.lower() in {".jpg", ".jpeg"}]
    for person, src in sources:
        digest = hashlib.md5(src.read_bytes()).hexdigest()
        if digest in seen:  # same photo in two folders
            continue
        seen.add(digest)
        obj = next((o for o in TEAM_OBJECTS[person] if re.search(rf"_{o}_", src.name, re.I)), None)
        if obj is None:
            raise ValueError(f"Cannot find object name in {src}")
        split = "test" if person in test_people else ("val" if obj in VAL_OBJECTS else "train")
        dst = copy_image(src, out / "team_real" / person / f"{digest[:10]}_{src.stem}.jpg", rng)
        rows.append({"path": dst, "label": 0, "split": split, "group_id": f"team:{person}:{obj}", "domain": f"team_real:{person}",
                     "source": "team_real", "generator_group": "real"})
    owner = {o: p for p, objs in TEAM_OBJECTS.items() for o in objs}
    for src in sorted((team_dir / "fake" / "C").glob("*.png")):
        obj = src.name.split("_")[1]
        person = owner.get(FAKE_OBJECT_ALIASES.get(obj, obj))
        split = "test" if person in test_people else ("val" if FAKE_OBJECT_ALIASES.get(obj, obj) in VAL_OBJECTS else "train")
        dst = copy_image(src, out / "team_fake" / src.name, rng)
        rows.append({"path": dst, "label": 1, "split": split, "group_id": f"teamfake:{obj}", "domain": "team_fakeC",
                     "source": "team_fake", "generator_group": "holdout" if split == "test" else "seen"})
    return rows


def fraudbench_rows(fb_dir, out, holdout_generators, rng):
    files = [f for f in sorted(fb_dir.rglob("*")) if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    groups = sorted({(f.relative_to(fb_dir).parts[0], next(p for p in f.parts if p.startswith("Review_"))) for f in files})
    shuffled = groups[:]
    random.Random(0).shuffle(shuffled)
    n = len(shuffled)
    split_of = {g: ("test" if i < n * 0.2 else "val" if i < n * 0.3 else "train") for i, g in enumerate(shuffled)}
    rows = []
    for f in files:
        rel = f.relative_to(fb_dir)
        category, kind = rel.parts[0], rel.parts[1]
        review = next(p for p in rel.parts if p.startswith("Review_"))
        split = split_of[(category, review)]
        generator = rel.parts[2] if kind == "DeepFake" else "real"
        if generator in holdout_generators and split != "test":
            continue  # held-out generators appear only in test groups
        dst = copy_image(f, out / "fraudbench" / rel, rng)
        rows.append({"path": dst, "label": int(kind == "DeepFake"), "split": split,
                     "group_id": f"fb:{category}:{review}", "domain": f"fraudbench:{generator}",
                     "source": "fraudbench_fake" if kind == "DeepFake" else "fraudbench_real",
                     "generator_group": "real" if kind != "DeepFake" else ("holdout" if generator in holdout_generators else "seen")})
    return rows


def defactify_rows(defactify_metadata):
    meta = pd.read_csv(defactify_metadata, keep_default_na=False)
    root = Path(defactify_metadata).parent
    return [{"path": (root / r.image_path).resolve(), "label": int(r.label), "split": r.split, "group_id": f"defactify:{r.caption_id}",
             "domain": f"defactify:{r.domain}", "source": "coco" if r.label == 0 else "defactify_fake",
             "generator_group": r.generator_group} for r in meta.itertuples()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team", required=True, help="Folder with real/<person>(...)/ and fake/C/")
    parser.add_argument("--team-extra-giwon", nargs="*", default=[], help="Extra folders of giwon photos (deduplicated by content)")
    parser.add_argument("--fraudbench", required=True)
    parser.add_argument("--defactify-metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-photographers", nargs="*", default=["taei", "YooJaehwan"])
    parser.add_argument("--holdout-generators", nargs="*", default=["gpt-image-2", "nano-banana-2"])
    args = parser.parse_args()
    out = Path(args.output)
    rng = random.Random(42)
    rows = (defactify_rows(args.defactify_metadata) + fraudbench_rows(Path(args.fraudbench), out, set(args.holdout_generators), rng)
            + team_rows(Path(args.team), args.team_extra_giwon, out, set(args.test_photographers), rng))
    df = pd.DataFrame(rows)
    df["image_path"] = [str(Path(p).resolve()) for p in df.path]
    df["fake_type"] = df.label.map({0: "real", 1: "unknown"})
    df["source_id"] = df.group_id
    train = df.split == "train"
    counts = df[train].source.value_counts()
    df["sample_weight"] = [SOURCE_SHARE[s] / counts[s] if t else 0.0 for s, t in zip(df.source, train)]
    df.drop(columns=["path"]).to_csv(out / "metadata.csv", index=False)
    print(pd.crosstab([df.split, df.label], df.source))


if __name__ == "__main__":
    main()
