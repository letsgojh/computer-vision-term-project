"""Add aligned real/fake pairs to a metadata file (v4 data).

For selected real train/val images: cut native-resolution crops, pass each crop through a
Stable Diffusion VAE (encode -> decode), and save the original crop (label 0) and its
reconstruction (label 1) with the same JPEG quality. Each pair shares content, camera and
compression, so the only difference is the generator's decoding trace; the detector cannot
rely on scene, device or file properties to separate them (dataset-alignment idea).
Pairs keep the source image's group_id and split, so no pair crosses splits.
"""

import argparse
import hashlib
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_pil_image, to_tensor
from tqdm import tqdm

# Target share of each source inside its class for the weighted sampler (v4).
SOURCE_SHARE = {"coco": 0.30, "fraudbench_real": 0.25, "team_real": 0.20, "aligned_real": 0.25,
                "defactify_fake": 0.30, "fraudbench_fake": 0.30, "team_fake": 0.10, "aligned_fake": 0.30}
CROPS_PER_SOURCE = {"team_real": 4, "fraudbench_real": 2, "coco": 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", required=True, help="v3 metadata.csv")
    parser.add_argument("--vae", required=True, help="Diffusers model folder containing vae/")
    parser.add_argument("--output", required=True, help="Folder for crops and the new metadata.csv")
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--coco-limit", type=int, default=2500, help="COCO reals to pair (train+val)")
    parser.add_argument("--quality", type=int, default=92)
    args = parser.parse_args()
    from diffusers import AutoencoderKL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae").to(device).eval()
    meta = pd.read_csv(args.metadata, keep_default_na=False)
    reals = meta[(meta.label == 0) & meta.split.isin(["train", "val"]) & meta.source.isin(CROPS_PER_SOURCE)]
    coco = reals[reals.source == "coco"]
    reals = pd.concat([reals[reals.source != "coco"], coco.sample(min(len(coco), args.coco_limit), random_state=0)])
    out = Path(args.output)
    rng = random.Random(0)
    rows = []
    for item in tqdm(reals.itertuples(), total=len(reals), desc="pairs"):
        with Image.open(item.image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        size = min(args.crop, *image.size) // 8 * 8
        for k in range(CROPS_PER_SOURCE[item.source]):
            left, top = rng.randint(0, image.width - size), rng.randint(0, image.height - size)
            crop = image.crop((left, top, left + size, top + size))
            with torch.no_grad():
                x = to_tensor(crop).unsqueeze(0).to(device) * 2 - 1
                recon = vae.decode(vae.encode(x).latent_dist.mean).sample.clamp(-1, 1)
            stem = f"{Path(item.image_path).stem}_{hashlib.md5(item.image_path.encode()).hexdigest()[:8]}_{k}"
            for label, picture, kind in [(0, crop, "aligned_real"), (1, to_pil_image((recon[0].cpu() + 1) / 2), "aligned_fake")]:
                path = out / kind / f"{stem}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                picture.save(path, quality=args.quality)
                rows.append({"image_path": str(path.resolve()), "label": label, "split": item.split, "group_id": item.group_id,
                             "domain": f"{kind}:{item.source}", "source": kind, "generator_group": "real" if label == 0 else "seen",
                             "fake_type": "real" if label == 0 else "unknown", "source_id": item.source_id})
    merged = pd.concat([meta, pd.DataFrame(rows)], ignore_index=True)
    train = merged.split == "train"
    counts = merged[train].source.value_counts()
    merged["sample_weight"] = [SOURCE_SHARE[s] / counts[s] if t else 0.0 for s, t in zip(merged.source, train)]
    merged.to_csv(out / "metadata.csv", index=False)
    print(pd.crosstab([merged.split, merged.label], merged.source))


if __name__ == "__main__":
    main()
