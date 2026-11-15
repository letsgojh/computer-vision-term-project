# -*- coding: utf-8 -*-
"""
가짜 생성 C - SDXL Text-to-Image 파이프라인 (팀 데이터 규약 100% 준수 버전)
실행 환경: Google Colab (T4 GPU)
출력 이미지 규칙: fakeC_{object}_{idx:03d}_s{seed}.png
출력 메타데이터: metadata/fake_mapping.csv
"""

import os
import csv
import random
import shutil
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

# =====================================================================
# 📌 [1번 셀] 필수 라이브러리 및 SDXL 모델 불러오기
# =====================================================================
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")


# =====================================================================
# 📌 [2번 셀] 팀 규약 매핑 및 generate() 함수 정의
# =====================================================================
OUT_DIR = "fakeC"
IMG_DIR = os.path.join(OUT_DIR, "C")
CSV_PATH = os.path.join(OUT_DIR, "fake_mapping.csv")

# 2주차 대량 생성 시 구글 드라이브 마운트 경로로 변경 가능:
# from google.colab import drive
# drive.mount('/content/drive')
# OUT_DIR = "/content/drive/MyDrive/CV_2조_fakeC"

if OUT_DIR == "fakeC" and os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(IMG_DIR, exist_ok=True)

# 20종 품목: {파일명/CSV용 짧은 영문 키: SDXL 프롬프트용 상세 영어 표현}
OBJ_MAP = {
    "smartphone": "smartphone",
    "laptop": "laptop",
    "smartwatch": "smartwatch",
    "earbuds": "wireless earbuds",
    "headphones": "headphones",
    "keyboard": "mechanical keyboard",
    "mouse": "wireless mouse",
    "sneakers": "sneakers",
    "wallet": "leather wallet",
    "hoodie": "folded hoodie",
    "chair": "desk chair",
    "book": "book",
    "bicycle": "bicycle",
    "backpack": "backpack",
    "tumbler": "tumbler",
    "cosmetics": "cosmetics bottle",
    "fridge": "mini fridge",
    "tripod": "camera tripod",
    "mattress": "mattress",
    "desk": "wooden study desk",
}

LARGE_OBJS = {"sneakers", "chair", "bicycle", "fridge", "tripod", "mattress", "desk"}

TEMPLATE = (
    'a casual snapshot of a {obj_prompt} placed on a {bg}, '
    'with only one small yellow sticky note beside it reading "{date} {nick}", '
    '{light}, realistic, amateur smartphone photo, clean background'
)

NEGATIVE = (
    "multiple sticky notes, two notes, cluttered, lots of paper, "
    "cartoon, illustration, 3d render, blurry text, gibberish, deformed, watermark, "
    "logo on object, text on object, wired, cable, "
    "large sheet of paper, blank paper, pen, pencil, marker, close-up, cropped"
)

FAKE_MAPPING_COLUMNS = [
    "fake_filename", "fake_type", "source_filename", "mask_filename", "object",
    "prompt", "negative_prompt", "seed", "model", "created_by", "is_reinforcement",
]


def fake_c_name(obj_key: str, idx: int, seed: int) -> str:
    return f"fakeC_{obj_key}_{idx:03d}_s{seed}.png"


def generate(
    obj_key: str,
    bg: str,
    light: str,
    date: str,
    nick: str = "changjun",
    seed: int = None,
    idx: int = 1,
    steps: int = 30,
    guidance: float = 7.0,
    is_reinforcement: int = 0,
    record_log: bool = True,
):
    seed = seed if seed is not None else random.randint(1, 99999)
    obj_prompt = OBJ_MAP.get(obj_key, obj_key)
    prompt = TEMPLATE.format(obj_prompt=obj_prompt, bg=bg, light=light, date=date, nick=nick)
    g = torch.Generator("cuda").manual_seed(seed)

    img = pipe(
        prompt,
        negative_prompt=NEGATIVE,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=g,
    ).images[0]

    fname = fake_c_name(obj_key, idx, seed)
    img.save(os.path.join(IMG_DIR, fname))

    if record_log:
        new_file = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(FAKE_MAPPING_COLUMNS)
            w.writerow([
                fname, "C", "", "", obj_key,
                prompt, NEGATIVE, seed, "sdxl-base-1.0", nick, is_reinforcement
            ])

    return img


# =====================================================================
# 📌 [3번 셀] 단건 샘플 테스트 (fakeC_mouse_001_s777.png)
# =====================================================================
if __name__ == "__main__":
    obj_counts = {}

    def next_idx(key: str) -> int:
        obj_counts[key] = obj_counts.get(key, 0) + 1
        return obj_counts[key]

    sample_img = generate(
        obj_key="mouse",
        bg="wooden desk",
        light="natural window light",
        date="2026.11.13",
        nick="changjun",
        seed=777,
        idx=next_idx("mouse"),
        steps=40,
    )

    # =====================================================================
    # 📌 [4번 셀] 팀 확정 20종 품목 각 1장씩 생성
    # =====================================================================
    bgs_small = ["wooden desk", "white table", "bed sheet", "floor"]
    bgs_large = ["wooden floor", "living room floor", "bedroom floor"]
    lights = ["bright fluorescent light", "natural window light", "dim warm lamp light"]

    for i, obj_key in enumerate(OBJ_MAP.keys(), start=1):
        bg = random.choice(bgs_large if obj_key in LARGE_OBJS else bgs_small)
        light = random.choice(lights)
        idx = next_idx(obj_key)
        print(f"[{i}/20] 생성 중: {obj_key} (idx={idx:03d}, 배경={bg})...")
        generate(
            obj_key=obj_key,
            bg=bg,
            light=light,
            date="2026.11.13",
            nick="changjun",
            seed=random.randint(1, 99999),
            idx=idx,
        )
