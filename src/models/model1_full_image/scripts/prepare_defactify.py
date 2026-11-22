"""Export Defactify parquet shards to JPEG files plus a model-1 metadata CSV.

Label_B: 0=real (MS COCO), 1=SD2.1, 2=SDXL, 3=SD3, 4=DALL-E 3, 5=Midjourney v6.
The mapping follows the dataset card order and was spot-checked by eye on same-caption rows.

Generators in --holdout are excluded from train/val and appear only in test, so the test
set can be scored separately for seen and unseen generators. Image bytes are written
unchanged (no re-encoding) so the source JPEG artifacts are preserved.
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

GENERATORS = {0: "coco", 1: "sd21", 2: "sdxl", 3: "sd3", 4: "dalle3", 5: "midjourney"}
SPLIT_NAMES = {"train": "train", "validation": "val", "test": "test"}


def load_split(source, split):
    """Read one Defactify split, keeping image bytes."""
    files = sorted(Path(source, "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards for split {split} in {source}")
    frame = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    frame["generator"] = frame.Label_B.map(GENERATORS)
    if frame.generator.isna().any():
        raise ValueError(f"Unknown Label_B values: {sorted(frame.Label_B[frame.generator.isna()].unique())}")
    if not ((frame.Label_A == 0) == (frame.Label_B == 0)).all():
        raise ValueError("Label_A and Label_B disagree on which images are real")
    return frame


def sample(frame, per_generator, seed):
    """Take up to per_generator rows from each generator (and from the real class)."""
    parts = [g.sample(min(len(g), per_generator[name]), random_state=seed)
             for name, g in frame.groupby("generator") if per_generator.get(name, 0) > 0]
    return pd.concat(parts, ignore_index=True)


def export(source, output, root, holdout, counts, seed):
    """Write images and metadata; returns the metadata frame."""
    output, root = Path(output), Path(root)
    seen = [g for g in GENERATORS.values() if g != "coco" and g not in holdout]
    rows = []
    for split, name in SPLIT_NAMES.items():
        frame = load_split(source, split)
        real_n, fake_n = counts[name]
        allowed = seen + (list(holdout) if name == "test" else [])
        plan = {"coco": real_n, **{g: fake_n for g in allowed}}
        frame = sample(frame, plan, seed)
        for index, row in frame.iterrows():
            folder = output / "images" / name / row.generator
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{name}_{row.generator}_{index:06d}.jpg"
            if not path.exists():
                path.write_bytes(row.Image["bytes"])
            caption_id = hashlib.sha1(row.Caption.strip().lower().encode("utf-8")).hexdigest()[:16]
            rows.append({"image_path": path.resolve().relative_to(root.resolve()).as_posix(),
                         "label": int(row.Label_A), "fake_type": "real" if row.Label_A == 0 else "unknown",
                         "source_id": f"{name}_{caption_id}", "split": name, "domain": row.generator,
                         "caption_id": caption_id,
                         "generator_group": "real" if row.Label_A == 0 else ("holdout" if row.generator in holdout else "seen")})
        print(f"{name}: {frame.generator.value_counts().to_dict()}")
    metadata = pd.DataFrame(rows)
    leaked = metadata.groupby("caption_id").split.nunique()
    leaked = leaked[leaked > 1].index
    if len(leaked):
        print(f"Removing {metadata.caption_id.isin(leaked).sum()} rows whose caption appears in more than one split")
        metadata = metadata[~metadata.caption_id.isin(leaked)]
    metadata.to_csv(output / "metadata.csv", index=False)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Folder holding data/*.parquet from the HF download")
    parser.add_argument("--output", required=True, help="Folder for images/ and metadata.csv")
    parser.add_argument("--root", default=".", help="image_path values are written relative to this folder")
    parser.add_argument("--holdout", nargs="*", default=["dalle3", "midjourney"])
    parser.add_argument("--train", nargs=2, type=int, default=[7000, 2334], metavar=("REAL", "PER_FAKE_GEN"))
    parser.add_argument("--val", nargs=2, type=int, default=[1500, 500], metavar=("REAL", "PER_FAKE_GEN"))
    parser.add_argument("--test", nargs=2, type=int, default=[3000, 1500], metavar=("REAL", "PER_FAKE_GEN"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    unknown = set(args.holdout) - set(GENERATORS.values()) | ({"coco"} & set(args.holdout))
    if unknown:
        parser.error(f"Invalid holdout generators: {sorted(unknown)}")
    counts = {"train": args.train, "val": args.val, "test": args.test}
    metadata = export(args.source, args.output, args.root, set(args.holdout), counts, args.seed)
    print(pd.crosstab(metadata.split, metadata.domain))


if __name__ == "__main__":
    main()
