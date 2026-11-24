"""모델 2가 A 가짜를 '손글씨'로 잡는지, 다른 지름길로 잡는지 확인하는 대조 실험.

A 가짜는 마스크(바꾼 메모 영역) 밖이 원본과 같다. 이 성질을 이용해 test split의 A마다 네 가지를 비교한다.

  fake       가짜 그대로
  source     그 가짜의 원본(진짜)
  restore    가짜에서 마스크 안을 원본 픽셀로 되돌린 것      → 진짜처럼 나와야 정상
  transplant 원본에 가짜의 마스크 안만 옮겨 붙인 것          → 가짜처럼 나와야 정상
  control    원본을 생성과 똑같이 잘라 줄였다 키워 붙이되,
             글씨는 바꾸지 않은 것 (AI 편집 없음)             → 진짜처럼 나와야 정상

control이 가짜로 나오면 모델은 손글씨가 아니라 '잘라 붙인 흔적(리샘플링·경계)'을 보는 것이다.
모든 변형은 후처리 original과 같게 JPEG q95로 다시 인코딩한 뒤 예측한다.

사용:
  python -m src.models.model2_memo.scripts.shortcut_tests --config outputs/model2_realAC/config.yaml \
      --checkpoint outputs/model2_realAC/checkpoints/best_model.pth --data-root ~/cvtp_data
"""
from __future__ import annotations

import argparse
import importlib.util
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageOps

from ..boxes import read_boxes
from ..dataset import build_transform
from ..model import build_model
from ..utils import load_config, select_device

GEN_SCRIPT = Path(__file__).resolve().parents[1] / "notebooks" / "modal_fakeA_qwen.py"


def _jpeg(img: Image.Image, q: int = 95) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _soft(mask: Image.Image, radius: float) -> Image.Image:
    return mask.filter(ImageFilter.GaussianBlur(radius)) if radius > 0 else mask


def _gen_module():
    spec = importlib.util.spec_from_file_location("gen", GEN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(config: str, checkpoint: str, data_root: str, out_dir: str | None = None, device: str = "auto"):
    root = Path(data_root).expanduser()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = load_config(config)
    dev = select_device(device)
    model = build_model(ckpt["config"]["model"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(dev).eval()
    tf = build_transform(ckpt["config"], train=False)
    thr = float(ckpt["threshold"])
    boxes = read_boxes(cfg["locate"]["boxes_csv"])
    meta = pd.read_csv(cfg["data"]["metadata"], dtype=str)
    gen = _gen_module()

    @torch.inference_mode()
    def prob(img: Image.Image, box) -> float:
        return float(torch.sigmoid(model(tf(img, box).unsqueeze(0).to(dev))[0]))

    test = meta[(meta.split == "test") & (meta.fake_type == "A") & (meta.postprocess == "original")]
    rows = []
    for r in test.itertuples():
        fstem = Path(r.image_path).stem
        spath = f"data/processed/original/{r.source_id}.jpg"
        mpath = root / "data" / "fake" / "masks" / f"{fstem}_mask.png"
        if r.image_path not in boxes or spath not in boxes or not mpath.exists():
            continue
        fake = ImageOps.exif_transpose(Image.open(root / r.image_path)).convert("RGB")
        src = ImageOps.exif_transpose(Image.open(root / spath)).convert("RGB")
        if fake.size != src.size:
            continue
        mask = Image.open(mpath).convert("L")
        soft = _soft(mask, 0.01 * min(fake.size))
        fbox, sbox = boxes[r.image_path]["box"], boxes[spath]["box"]
        restore = _jpeg(Image.composite(src, fake, soft))
        transplant = _jpeg(Image.composite(fake, src, soft))
        # control: 생성 파이프라인 그대로(크롭→1024로 축소→복원→색 맞춤→페더 붙이기), 편집만 없음
        real_path = root / "data" / "real" / f"{r.source_id}.jpg"
        gen_box = None
        if real_path.exists():
            # 생성에 쓴 박스(원본 기준 정규화 좌표)를 그대로 쓴다
            gen_box = tuple(sbox)
            img, c, crop, small_png = gen.prepare(real_path, gen_box)
            control, _, _ = gen.paste(img, c, crop, small_png, gen_box)
            control = _jpeg(control)
        rows.append({
            "fake": r.image_path, "source": spath,
            "p_fake": prob(fake, fbox), "p_source": prob(src, sbox),
            "p_restore": prob(restore, fbox), "p_transplant": prob(transplant, sbox),
            "p_control": prob(control, sbox) if gen_box else np.nan,
        })
    df = pd.DataFrame(rows)
    out = Path(out_dir or Path(cfg["training"]["output_dir"]) / "shortcut_tests")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "shortcut_tests.csv", index=False)

    def rate(col):
        return float((df[col] >= thr).mean()) if len(df) else float("nan")

    lines = [f"test A 쌍 {len(df)}개, 임계값 {thr:.2f}", "",
             "변형        가짜로 판정한 비율   평균 P(fake)   기대",
             f"fake        {rate('p_fake'):6.1%}            {df.p_fake.mean():.3f}         높아야 함",
             f"source      {rate('p_source'):6.1%}            {df.p_source.mean():.3f}         낮아야 함",
             f"restore     {rate('p_restore'):6.1%}            {df.p_restore.mean():.3f}         낮아야 함 (높으면 메모 밖을 봄)",
             f"transplant  {rate('p_transplant'):6.1%}            {df.p_transplant.mean():.3f}         높아야 함 (낮으면 메모 밖을 봄)",
             f"control     {rate('p_control'):6.1%}            {df.p_control.mean():.3f}         낮아야 함 (높으면 손글씨가 아닌 붙이기 흔적을 봄)",
             "", f"쌍별로 가짜 점수가 원본보다 높은 비율: {(df.p_fake > df.p_source).mean():.1%}"]
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="모델 2 지름길 대조 실험 (A 타입)")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    run(a.config, a.checkpoint, a.data_root, a.output_dir, a.device)


if __name__ == "__main__":
    main()
