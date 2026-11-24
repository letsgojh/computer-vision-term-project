# ==============================================================================
# 🎨 [컴비 2조] 가짜 생성 B (제품 영역 인페인팅) - Colab 자동화 스타터
# ------------------------------------------------------------------------------
# 팀 규약 (docs/data_contract.md) 100% 준수:
#   - 가짜 이미지: fakeB_{원본stem}_s{seed}.png  -> data/fake/B/
#   - 마스크 파일: fakeB_{원본stem}_s{seed}_mask.png -> data/fake/masks/
#   - 대응표 CSV : fake_mapping_B.csv (source_filename, mask_filename 자동 기록)
#   - 구글 드라이브 실시간 저장 + 끊겨도 자동 이어받기(Resume) 지원
# ==============================================================================

# ==============================================================================
# 📌 [1번 셀] 라이브러리 설치 & 모델 로드 (CLIPSeg 자동 마스크 + SDXL Inpainting)
# ==============================================================================
# !pip install -q diffusers transformers accelerate safetensors pillow

import os, csv, glob, random, shutil
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from diffusers import AutoPipelineForInpainting, AutoencoderKL
from google.colab import drive

drive.mount("/content/drive")

INPAINT_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
INPAINT_MODEL_NAME = "sdxl-inpainting-0.1"

# 1) 자동 마스크 모델 (CLIPSeg - 가벼워서 T4에서 SDXL과 같이 돌아감)
clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to("cuda")

# 2) SDXL Inpainting 모델
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)
pipe = AutoPipelineForInpainting.from_pretrained(
    INPAINT_MODEL_ID,
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

print("✅ CLIPSeg + SDXL Inpainting 로드 완료!")


# ==============================================================================
# 📌 [2번 셀] 경로 설정 & 자동 마스크 + 인페인팅 함수 정의
# ==============================================================================
# 👉 내일 근영님과 합의 후, 드라이브의 실제 원본 폴더 경로만 여기 맞춰주세요!
REAL_DIR = "/content/drive/MyDrive/fakeB_input_real"     # 원본 진짜 사진(.jpg/.png) 폴더
OUT_DIR  = "/content/drive/MyDrive/fakeB_output"         # 결과 저장 폴더 (실시간 저장)

FAKE_B_DIR = os.path.join(OUT_DIR, "B")
MASK_DIR   = os.path.join(OUT_DIR, "masks")
CSV_PATH   = os.path.join(OUT_DIR, "fake_mapping_B.csv")

os.makedirs(FAKE_B_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

FAKE_MAPPING_COLUMNS = [
    "fake_filename", "fake_type", "source_filename", "mask_filename", "object",
    "prompt", "negative_prompt", "seed", "model", "created_by", "is_reinforcement",
]

# 물건별 영어 마스크 타깃 & 인페인팅 교체 프롬프트 (24종)
OBJ_PROMPTS = {
    "smartphone":     ("smartphone", "a realistic black smartphone lying on the surface, natural lighting, matching shadows, authentic photo"),
    "laptop":         ("laptop", "a realistic silver laptop with screen off, natural lighting, matching shadows, authentic photo"),
    "smartwatch":     ("smartwatch", "a realistic black smartwatch with silicone strap, natural lighting, matching shadows, authentic photo"),
    "earbuds":        ("wireless earbuds case", "a realistic wireless earbuds charging case, natural lighting, matching shadows, authentic photo"),
    "headphones":     ("headphones", "a realistic pair of black over-ear headphones, natural lighting, matching shadows, authentic photo"),
    "keyboard":       ("keyboard", "a realistic mechanical keyboard, natural lighting, matching shadows, authentic photo"),
    "mouse":          ("computer mouse", "a realistic wireless computer mouse, natural lighting, matching shadows, authentic photo"),
    "sneakers":       ("sneakers", "a realistic pair of casual sneakers, natural lighting, matching shadows, authentic photo"),
    "tumbler":        ("tumbler cup", "a realistic stainless steel tumbler, natural lighting, matching shadows, authentic photo"),
    "hoodie":         ("hoodie", "a realistic folded plain gray hoodie, natural lighting, matching shadows, authentic photo"),
    "chair":          ("chair", "a realistic black office desk chair, natural lighting, matching shadows, authentic photo"),
    "wallet":         ("leather wallet", "a realistic brown leather bifold wallet, natural lighting, matching shadows, authentic photo"),
    "gamecontroller": ("game controller", "a realistic black wireless game controller, natural lighting, matching shadows, authentic photo"),
    "backpack":       ("backpack", "a realistic everyday fabric backpack, natural lighting, matching shadows, authentic photo"),
    "book":           ("book", "a realistic closed hardcover book, natural lighting, matching shadows, authentic photo"),
    "cosmetics":      ("cosmetics bottle", "a realistic white skincare lotion pump bottle, natural lighting, matching shadows, authentic photo"),
    "fridge":         ("mini fridge", "a realistic silver mini refrigerator, natural lighting, matching shadows, authentic photo"),
    "tripod":         ("camera tripod", "a realistic black aluminum tripod, natural lighting, matching shadows, authentic photo"),
    "lunchbox":       ("lunchbox container", "a realistic plastic bento lunchbox container, natural lighting, matching shadows, authentic photo"),
    "desk":           ("wooden desk", "a realistic wooden study desk, natural lighting, matching shadows, authentic photo"),
    "microphone":     ("microphone", "a realistic desktop microphone on a stand, natural lighting, matching shadows, authentic photo"),
    "muffler":        ("knitted scarf", "a realistic folded knitted winter scarf, natural lighting, matching shadows, authentic photo"),
    "gloves":         ("winter gloves", "a realistic pair of winter gloves, natural lighting, matching shadows, authentic photo"),
    "perfume":        ("perfume bottle", "a realistic glass perfume bottle, natural lighting, matching shadows, authentic photo"),
}

NEG_PROMPT_B = (
    "cartoon, 3d render, deformed, blurry, watermark, logo, text, sticky note, "
    "memo paper, handwriting, artifacts, unnatural seam"
)

def make_auto_mask(pil_img: Image.Image, target_text: str, threshold=0.35, dilate_px=15) -> Image.Image:
    """CLIPSeg로 원본 사진에서 제품(target_text) 영역만 흰색(255), 배경/메모는 검정(0) 마스크 생성"""
    inputs = clipseg_processor(
        text=[target_text], images=[pil_img], padding="max_length", return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        preds = clipseg_model(**inputs).logits
    prob = torch.sigmoid(preds).squeeze().cpu().numpy()
    mask_bin = (prob > threshold).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_bin).resize(pil_img.size, Image.BILINEAR)
    # 경계 자연스럽게 약간 확장 + 블러
    if dilate_px > 0:
        k = dilate_px if dilate_px % 2 == 1 else dilate_px + 1
        mask_img = mask_img.filter(ImageFilter.MaxFilter(k))
    return mask_img.convert("L")

def generate_fake_b(
    src_path: str,
    obj_key: str,
    seed: int = None,
    strength: float = 0.85,
    guidance: float = 7.5,
    steps: int = 35,
    mask_threshold: float = 0.35,
    created_by: str = "changjun",
    is_reinforcement: int = 0,
    record_log: bool = True,
):
    seed = seed if seed is not None else random.randint(1, 99999)
    src_fname = os.path.basename(src_path)
    src_stem = os.path.splitext(src_fname)[0]

    # 원본 로드 (EXIF 회전 보정 + 1024 정사각/비율 유지 리사이즈)
    orig = ImageOps.exif_transpose(Image.open(src_path)).convert("RGB")
    orig_w, orig_h = orig.size

    # SDXL 권장 1024 기준 리사이즈 (8의 배수 맞춤)
    scale = 1024 / max(orig_w, orig_h)
    rw = max(512, int(round(orig_w * scale / 8) * 8))
    rh = max(512, int(round(orig_h * scale / 8) * 8))
    img_in = orig.resize((rw, rh), Image.LANCZOS)

    mask_target, prompt = OBJ_PROMPTS.get(
        obj_key, (obj_key, f"a realistic {obj_key}, natural lighting, matching shadows, authentic photo")
    )
    mask_in = make_auto_mask(img_in, mask_target, threshold=mask_threshold)

    g = torch.Generator("cuda").manual_seed(seed)
    out_img = pipe(
        prompt=prompt,
        negative_prompt=NEG_PROMPT_B,
        image=img_in,
        mask_image=mask_in,
        strength=strength,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=g,
    ).images[0]

    # 규약 파일명: fakeB_{원본stem}_s{seed}.png / {가짜stem}_mask.png
    fake_stem = f"fakeB_{src_stem}_s{seed}"
    fake_fname = f"{fake_stem}.png"
    mask_fname = f"{fake_stem}_mask.png"

    out_img.save(os.path.join(FAKE_B_DIR, fake_fname))
    mask_in.save(os.path.join(MASK_DIR, mask_fname))

    if record_log:
        new_file = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(FAKE_MAPPING_COLUMNS)
            w.writerow([
                fake_fname, "B", src_fname, mask_fname, obj_key,
                prompt, NEG_PROMPT_B, seed, INPAINT_MODEL_NAME, created_by, is_reinforcement,
            ])

    return img_in, mask_in, out_img, fake_fname


# ==============================================================================
# 📌 [3번 셀] 1장 테스트 (원본 / 자동마스크 / 인페인팅 결과 눈으로 확인)
# ==============================================================================
# sample_paths = sorted(glob.glob(os.path.join(REAL_DIR, "*.*")))
# orig_im, mask_im, fake_im, fn = generate_fake_b(sample_paths[0], obj_key="mouse", record_log=False)
# display(orig_im.resize((384, 384)), mask_im.resize((384, 384)), fake_im.resize((384, 384)))


# ==============================================================================
# 📌 [4번 셀] 폴더 전체 일괄 배치 생성 (자동 이어받기 지원)
# ==============================================================================
# 파일명 규약: {닉네임}_{물건}_{번호3자리}.jpg -> parts[1]이 자동으로 obj_key!
def run_batch(target_objects=None):
    all_files = sorted(
        p for p in glob.glob(os.path.join(REAL_DIR, "*.*"))
        if p.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"📂 원본 사진 {len(all_files)}장 발견")

    for i, src_p in enumerate(all_files, 1):
        src_fname = os.path.basename(src_p)
        src_stem = os.path.splitext(src_fname)[0]
        parts = src_stem.split("_")
        if len(parts) < 2:
            print(f"⚠️ 파일명 규약 불일치 스킵: {src_fname}")
            continue
        obj_key = parts[1].lower()
        if target_objects and obj_key not in target_objects:
            continue

        # 이미 생성된 원본이면 스킵 (이어받기)
        if glob.glob(os.path.join(FAKE_B_DIR, f"fakeB_{src_stem}_s*.png")):
            print(f"[{i}/{len(all_files)}] ⏭️ 이미 완료 (스킵): {src_fname}")
            continue

        _, _, _, fake_fn = generate_fake_b(src_p, obj_key=obj_key)
        print(f"[{i}/{len(all_files)}] ✅ {src_fname} -> {fake_fn}")

    zip_path = shutil.make_archive("/content/drive/MyDrive/fakeB_output", "zip", OUT_DIR)
    print(f"🎉 가짜 B 배치 완료! 압축 저장됨: {zip_path}")

# 실행 예시 (내일 근영님과 맡을 물건 합의 후 주석 해제해서 실행):
# run_batch(target_objects={"mouse", "keyboard", "smartphone"})  # 특정 물건만
# run_batch()  # REAL_DIR 전체
