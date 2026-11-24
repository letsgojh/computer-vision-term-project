"""가짜 A 만들기 (Modal 판): 진짜 사진의 메모 날짜만 Qwen-Image-Edit-2511로 바꾼다.

GPU 일은 Modal(A100-80GB)이 하고, 크롭·붙이기·저장은 내 컴퓨터에서 한다.
메모 위치는 모델 2 locate 결과(memo_boxes.csv, processed/original 기준 정규화 좌표)를 그대로 쓴다.

실행 (저장소 루트에서):
  modal run src/models/model2_memo/notebooks/modal_fakeA_qwen.py::download        # 모델을 Modal 볼륨에 한 번 받기
  modal run src/models/model2_memo/notebooks/modal_fakeA_qwen.py --n 5 --out ~/cvtp_data/fakeA_trial --compare
  modal run src/models/model2_memo/notebooks/modal_fakeA_qwen.py --n 0 --out ~/cvtp_data   # 전체 (data/fake/A, masks, metadata)
"""
import csv
import os
import hashlib
import io
import math
import random
import time
from datetime import date, timedelta
from pathlib import Path

import modal

EDIT_MODEL = "Qwen/Qwen-Image-Edit-2511"
LIGHTNING_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LIGHTNING_FILE = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
DIFFUSERS_COMMIT = "750b3be2c0198e194c87e931a6904066de167a5f"
MODEL_TAG = "Qwen-Image-Edit-2511"
# A100-80GB면 원래 크기(bf16), 그보다 작은 GPU(L4 24GB 등)면 4비트로 줄여서 올린다. 카드 등록 전에는 A100을 못 쓴다.
GPU = os.environ.get("CVTP_GPU", "A100-80GB")
MAX_GPUS = int(os.environ.get("CVTP_MAX_GPUS", "1"))  # GPU를 여러 대 띄우면 로컬 준비 동안 놀면서 과금된다
QUANT_4BIT = GPU not in ("A100-80GB", "H100", "H200", "B200")

BASE_SEED = 20261117
DATE_START, DATE_END = "2026-10-01", "2026-12-31"
CONTEXT, PASTE_PAD, FEATHER, EDIT_LONG_SIDE = 0.12, 0.04, 0.015, 1024
NEG = "blurry, distorted text, extra text, duplicated text, repeated date, words, letters, year, day, colon, printed font, clean vector strokes, smooth digital lines, hatching, sketchy outline letters, watermark, change of paper color, change of background"

app = modal.App("cvtp-fakeA-qwen")
vol = modal.Volume.from_name("cvtp-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", f"git+https://github.com/huggingface/diffusers@{DIFFUSERS_COMMIT}",
                 "transformers", "accelerate", "peft", "bitsandbytes", "safetensors", "pillow", "numpy", "huggingface_hub")
    .env({"HF_HOME": "/cache"})
)


@app.function(image=image, volumes={"/cache": vol}, timeout=3600, cpu=4, memory=8192)
def download():
    from huggingface_hub import hf_hub_download, snapshot_download
    snapshot_download(EDIT_MODEL)
    hf_hub_download(LIGHTNING_REPO, LIGHTNING_FILE)
    vol.commit()
    print("downloaded")


@app.cls(image=image, gpu=GPU, volumes={"/cache": vol}, timeout=3600, scaledown_window=60, memory=32768, max_containers=MAX_GPUS)
class Editor:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
        t = time.time()
        kw = dict(torch_dtype=torch.bfloat16)
        if QUANT_4BIT:
            from diffusers.quantizers import PipelineQuantizationConfig
            kw["quantization_config"] = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
                components_to_quantize=["transformer", "text_encoder"])
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(EDIT_MODEL, **kw).to("cuda")
        self.base_sched = self.pipe.scheduler
        # Qwen-Image-Lightning README의 스케줄러 설정
        self.light_sched = FlowMatchEulerDiscreteScheduler.from_config({
            "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False,
            "max_image_seq_len": 8192, "max_shift": math.log(3), "num_train_timesteps": 1000,
            "shift": 1.0, "shift_terminal": None, "stochastic_sampling": False,
            "time_shift_type": "exponential", "use_beta_sigmas": False, "use_dynamic_shifting": True,
            "use_exponential_sigmas": False, "use_karras_sigmas": False})
        self.pipe.load_lora_weights(LIGHTNING_REPO, weight_name=LIGHTNING_FILE, adapter_name="lightning")
        self.pipe.set_progress_bar_config(disable=True)
        print(f"loaded on {GPU} (4bit={QUANT_4BIT}) in {time.time() - t:.0f}s")

    @modal.method()
    def edit(self, png: bytes, prompt: str, seed: int, lightning: bool = True) -> bytes:
        import torch
        from PIL import Image
        t0 = time.time()
        small = Image.open(io.BytesIO(png)).convert("RGB")
        if lightning:
            self.pipe.enable_lora(); self.pipe.scheduler = self.light_sched
            kw = dict(num_inference_steps=4, true_cfg_scale=1.0)
        else:
            self.pipe.disable_lora(); self.pipe.scheduler = self.base_sched
            kw = dict(num_inference_steps=40, true_cfg_scale=4.0, negative_prompt=NEG)
        with torch.inference_mode():
            out = self.pipe(image=[small], prompt=prompt, generator=torch.Generator("cpu").manual_seed(seed),
                            num_images_per_prompt=1, **kw).images[0]
        print(f"edit {time.time() - t0:.1f}s", flush=True)
        buf = io.BytesIO(); out.save(buf, "PNG")
        return buf.getvalue()


# ── 내 컴퓨터 쪽 ─────────────────────────────────────────────────────────

def seed_for(stem):
    return int(hashlib.sha1(f"{BASE_SEED}:{stem}".encode()).hexdigest()[:8], 16) % 100000


def new_date(seed):
    d0, d1 = date.fromisoformat(DATE_START), date.fromisoformat(DATE_END)
    return (d0 + timedelta(days=random.Random(seed).randrange((d1 - d0).days + 1))).strftime("%Y.%m.%d")


def make_prompt(new):
    # 예전 "reads year Y, month M, day D"는 'year'·'day' 단어가 그대로 써지는 문제가 있어 숫자만 준다.
    y, m, d = new.split(".")
    # "2026 12 04"처럼 공백으로 주면 점을 빼먹어서(v7 시험) 점으로 준다. 원본이 하이픈이면 하이픈 유지.
    return (f"Change only the handwritten date on the paper note so that it reads {y}.{m}.{d} "
            "(use hyphens instead of dots only if the original uses hyphens). "
            "Only digits change: never write any words or letters, and never add colons. "
            "Keep the original separators (dots or hyphens), spacing, line breaks, orientation and rotation exactly as they are; "
            "if the note is upside down or sideways, the new date stays upside down or sideways in the same direction. "
            "Rewrite only the digits that differ, and do not add, repeat or duplicate any text. "
            "Write by hand with exactly the same writing tool as the original (pencil, ballpoint pen or marker), with the same ink color, "
            "stroke thickness, pressure, size, slant and handwriting style. If the original is written with a marker, use solid, fully filled "
            "marker strokes. The strokes should look like real handwriting photographed on paper, slightly uneven, not clean or digital. "
            "Keep the paper, its color, lighting, shadows, background and everything else exactly the same.")


def upright_rotation(png: bytes) -> int:
    """거꾸로 놓인 메모를 Qwen이 바로 세워 다시 쓰는 문제(2026-11-24 시험) 때문에, 편집 전에 180° 돌릴지 정한다.
    손글씨는 OCR(easyocr)이 못 읽어서, 작은 VLM(Qwen2-VL-2B)에 '거꾸로냐'를 그대로/180° 돌려 두 번 묻고
    그대로 둔 쪽이 더 '거꾸로'라고 답하면 180°. 시험 13장에서 거꾸로 3장·똑바로 8장 모두 맞음.
    옆으로 누운 메모는 어느 쪽이 나와도 Qwen이 방향을 유지하므로 상관없다. 내 컴퓨터 GPU에서만 실행."""
    global _VLM
    import torch
    from PIL import Image
    if _VLM is None:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        mid = "Qwen/Qwen2-VL-2B-Instruct"
        proc = AutoProcessor.from_pretrained(mid)
        model = Qwen2VLForConditionalGeneration.from_pretrained(mid, dtype=torch.float16, device_map="cuda").eval()
        tok = proc.tokenizer
        _VLM = (proc, model, tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0])
    proc, model, yes, no = _VLM
    q = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text":
         "Look at the handwritten date on this paper note. Is the handwriting upside down (rotated 180 degrees)? Answer Yes or No."}]}]
    text = proc.apply_chat_template(q, add_generation_prompt=True)

    def p_upside(im):
        im = im.copy(); im.thumbnail((640, 640))
        inp = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
        with torch.no_grad():
            lg = model(**inp).logits[0, -1]
        return float(torch.softmax(lg[[yes, no]].float(), 0)[0])

    im = Image.open(io.BytesIO(png)).convert("RGB")
    # 400장 검토(2026-11-24): 차이 0으로 자르면 똑바른 메모도 많이 뒤집혀서 0.028 여유를 둔다.
    # 그래도 틀리는 몇 장은 _crops/{stem}.rot을 직접 고친다(한 번 정해지면 다시 계산하지 않음).
    return 180 if p_upside(im) - p_upside(im.rotate(180)) > 0.028 else 0


def rotate_png(png: bytes, k: int) -> bytes:
    if not k:
        return png
    from PIL import Image
    buf = io.BytesIO(); Image.open(io.BytesIO(png)).rotate(k).save(buf, "PNG")
    return buf.getvalue()


_VLM = None


def expand(b, frac, w, h):
    x0, y0, x1, y1 = b[0] * w, b[1] * h, b[2] * w, b[3] * h
    dx, dy = (x1 - x0) * frac, (y1 - y0) * frac
    return (int(max(0, x0 - dx)), int(max(0, y0 - dy)), int(min(w, x1 + dx)), int(min(h, y1 + dy)))


def match_color(src, ref):
    import numpy as np
    from PIL import Image
    s, r = np.asarray(src, np.float32), np.asarray(ref, np.float32)
    s = (s - s.mean((0, 1))) / (s.std((0, 1)) + 1e-6) * r.std((0, 1)) + r.mean((0, 1))
    return Image.fromarray(np.clip(s, 0, 255).astype(np.uint8))


def prepare(path, box):
    """원본을 EXIF 회전대로 세운 뒤(postprocess·locate와 같은 기준) 편집할 크롭을 만든다."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    w, h = img.size
    c = expand(box, CONTEXT, w, h)
    crop = img.crop(c)
    s = EDIT_LONG_SIDE / max(crop.size)
    small = crop.resize((max(16, round(crop.width * s / 16) * 16), max(16, round(crop.height * s / 16) * 16)), Image.LANCZOS)
    buf = io.BytesIO(); small.save(buf, "PNG")
    return img, c, crop, buf.getvalue()


def paste(img, c, crop, out_png, box):
    import numpy as np
    from PIL import Image, ImageFilter
    w, h = img.size
    cx0, cy0 = c[0], c[1]
    out = match_color(Image.open(io.BytesIO(out_png)).convert("RGB").resize(crop.size, Image.LANCZOS), crop)
    px0, py0, px1, py1 = expand(box, PASTE_PAD, w, h)
    m = Image.new("L", crop.size, 0); m.paste(255, (px0 - cx0, py0 - cy0, px1 - cx0, py1 - cy0))
    soft = m.filter(ImageFilter.GaussianBlur(max(1, FEATHER * min(crop.size))))
    fake = img.copy(); fake.paste(Image.composite(out, crop, soft), (cx0, cy0))
    full = Image.new("L", (w, h), 0); full.paste(255, (px0, py0, px1, py1))
    diff = float(np.abs(np.asarray(out, np.float32) - np.asarray(crop, np.float32))[py0 - cy0:py1 - cy0, px0 - cx0:px1 - cx0].mean())
    return fake, full, diff


@app.local_entrypoint()
def main(n: int = 5, out: str = "~/cvtp_data/fakeA_trial", compare: bool = False,
         boxes: str = "outputs/model2_realC/memo_boxes.csv", data_root: str = "~/cvtp_data",
         created_by: str = "jaehwan", lightning: bool = True, only: str = "", limit: int = 0):
    from PIL import Image
    root, out = Path(data_root).expanduser(), Path(out).expanduser()
    bx = {}
    for r in csv.DictReader(open(boxes)):
        name = r["image_path"].split("/")[-1]
        if "/original/" in r["image_path"] and not name.startswith("fake"):
            bx[name] = (float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"]))
    names = sorted(bx)
    if only:  # 쉼표로 구분한 원본 파일 이름만
        names = [x for x in names if Path(x).stem in only.split(",")]; n = 0
    if n:
        random.Random(BASE_SEED).shuffle(names); names = sorted(names[:n])
    (out / "data" / "fake" / "A").mkdir(parents=True, exist_ok=True)
    (out / "data" / "fake" / "masks").mkdir(parents=True, exist_ok=True)
    (out / "metadata").mkdir(parents=True, exist_ok=True)
    map_csv = out / "metadata" / "fake_mapping_A.csv"
    cols = ["fake_filename", "fake_type", "source_filename", "mask_filename", "object", "prompt", "negative_prompt",
            "seed", "model", "created_by", "is_reinforcement", "new_text", "memo_diff", "edit_rotation"]
    done = {r["fake_filename"]: r for r in csv.DictReader(open(map_csv))} if map_csv.exists() else {}

    jobs = []
    for name in names:
        stem = Path(name).stem; seed = seed_for(stem)
        fname = f"fakeA_{stem}_s{seed}.png"
        if fname in done and (out / "data" / "fake" / "A" / fname).exists():
            continue
        jobs.append((name, stem, seed, fname))
    if limit and len(jobs) > limit:  # 예산 때문에 일부만: 사람·물건이 고르게 섞이도록 무작위로 고른다
        random.Random(BASE_SEED + len(done)).shuffle(jobs); jobs = sorted(jobs[:limit])
    print(f"할 일 {len(jobs)}장 (전체 후보 {len(names)})")
    if not jobs:
        return

    ed = Editor()
    t0 = time.time()
    # 1단계(무료, GPU 없음): 모든 크롭을 미리 디스크에 만들어 둔다
    cache = out / "_crops"; cache.mkdir(exist_ok=True)
    for i, (name, stem, seed, fname) in enumerate(jobs, 1):
        cp = cache / f"{stem}.png"
        if not cp.exists():
            cp.write_bytes(prepare(root / "data" / "real" / name, bx[name])[3])
        rp = cache / f"{stem}.rot"
        if not rp.exists():
            rp.write_text(str(upright_rotation(cp.read_bytes())))
        if i % 50 == 0: print(f"크롭 준비 {i}/{len(jobs)}", flush=True)
    # 2단계(GPU): 입력을 한꺼번에 큐에 넣어 GPU가 쉬지 않게 하고, 결과가 오는 대로 붙인다
    rot = {j[1]: int((cache / f"{j[1]}.rot").read_text()) for j in jobs}
    print("180° 돌려서 편집:", [k for k, v in rot.items() if v] or "없음", flush=True)
    args = ([rotate_png((cache / f"{j[1]}.png").read_bytes(), rot[j[1]]) for j in jobs],
            [make_prompt(new_date(j[2])) for j in jobs], [j[2] for j in jobs], [lightning] * len(jobs))
    first2 = [(j, (cache / f"{j[1]}.png").read_bytes()) for j in jobs[:2]] if compare else []
    for k, ((name, stem, seed, fname), res) in enumerate(zip(jobs, ed.edit.map(*args)), 1):
        img, c, crop, _ = prepare(root / "data" / "real" / name, bx[name])
        fake, mask, diff = paste(img, c, crop, rotate_png(res, rot[stem]), bx[name])  # 다시 원래 방향으로
        mname = fname[:-4] + "_mask.png"
        fake.save(out / "data" / "fake" / "A" / fname)
        mask.save(out / "data" / "fake" / "masks" / mname)
        new = new_date(seed)
        done[fname] = {"fake_filename": fname, "fake_type": "A", "source_filename": name, "mask_filename": mname,
                       "object": stem.split("_")[1], "prompt": make_prompt(new), "negative_prompt": "" if lightning else NEG,
                       "seed": seed, "model": MODEL_TAG + ("+lightning4" if lightning else "+base40") + ("-nf4" if QUANT_4BIT else ""),
                       "created_by": created_by, "is_reinforcement": 0, "new_text": new, "memo_diff": round(diff, 2), "edit_rotation": rot[stem]}
        with open(map_csv, "w", newline="") as f:
            wr = csv.DictWriter(f, cols); wr.writeheader(); wr.writerows(done.values())
        if compare:
            Image.open(io.BytesIO(res)).save(out / f"raw_{'lightning' if lightning else 'base40'}_{stem}.png")
        print(f"[{k}/{len(jobs)}] {fname} 날짜 {new} 변화량 {diff:.1f} ({time.time() - t0:.0f}s 누적)", flush=True)
    if compare:  # 앞의 2장은 Lightning 없이(40단계)도 만들어 품질 비교
        for (name, stem, seed, _), png in first2:
            res = ed.edit.remote(png, make_prompt(new_date(seed)), seed, False)
            Image.open(io.BytesIO(res)).save(out / f"raw_base40_{stem}.png")
            print("base40 saved", stem)
    print(f"끝 ({time.time() - t0:.0f}s) → {out}")
