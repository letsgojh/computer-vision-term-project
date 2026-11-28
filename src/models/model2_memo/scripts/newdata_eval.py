"""팀원이 새로 만든 B·A(시험 물건분)를 모델 2 실험 4와 모델 1 CLIP으로 채점한다.
입력: ~/cvtp_data/incoming/{B,A_qwen,A_qwen2}/  출력: ~/cvtp_work/newtest/"""
import re, sys, json, gc
from pathlib import Path
import pandas as pd, torch, yaml
from PIL import Image, ImageOps
import os
REPO = Path(__file__).resolve().parents[4]; W = Path(os.environ.get("CVTP_WORK", Path.home()/"cvtp_work")); D = Path(os.environ.get("CVTP_DATA", Path.home()/"cvtp_data"))
sys.path.insert(0, str(W/"pp"))
from src.preprocess import transforms as T
from src.preprocess.postprocess import apply_variant
PPCFG = yaml.safe_load(open(W/"pp/configs/preprocess.yaml"))
sys.path.remove(str(W/"pp")); [sys.modules.pop(k) for k in list(sys.modules) if k == "src" or k.startswith("src.")]
sys.path.insert(0, str(REPO))
from src.models.model2_memo.locate import HFZeroShotDetector, choose_box
from src.models.model2_memo.boxes import FULL_BOX
from src.models.model2_memo.dataset import build_transform
from src.models.model2_memo.model import build_model

OUT = W/"newtest"; (OUT/"img").mkdir(parents=True, exist_ok=True)
EXP = REPO/"outputs/model2_realAC4_aug"
split = pd.read_csv(D/"data/splits/split.csv")
test_src = set(split[(split.split == "test") & (split.fake_type == "real")].filename.str.replace(".jpg", "", regex=False))
sdxl_b = set(pd.read_csv(D/"incoming/B/fake_mapping_B.csv").fake_filename) if (D/"incoming/B/fake_mapping_B.csv").exists() else set()
qwen_b = set(pd.read_csv(W/"fake_mapping_B_qwen.csv").fake_filename)

def norm_src(stem):
    s = re.sub(r"^fake[AB]_", "", stem); s = re.sub(r"_s\d+$", "", s)
    s = re.sub(r"^yoojaehwan_", "jaehwan_", s, flags=re.I)
    return s[0].lower() + s[1:] if s else s

rows = []
for sub, tag in [("B", "B"), ("A_qwen", "A_qwen"), ("A_qwen2", "A_qwen2")]:
    for p in sorted((D/"incoming"/sub).glob("fake*")):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"): continue
        src = norm_src(p.stem)
        if tag == "B": gen = "B_sdxl" if p.name in sdxl_b else ("B_qwen" if p.name in qwen_b else "B_unknown")
        else: gen = tag
        rows.append(dict(file=str(p), set=gen, source_id=src, in_test=src in test_src, name=f"{gen}__{p.stem}"))
df = pd.DataFrame(rows); print(df.groupby(["set", "in_test"]).size())
df = df[df.in_test].reset_index(drop=True)

# 1) 후처리 (팀 규약과 같은 함수·설정)
recs = []
for r in df.itertuples():
    img = T.load_image(Path(r.file))
    for v in ("original", "kakao", "resize"):
        out = OUT/"img"/v/f"{r.name}.jpg"
        if not out.exists():
            o, q = apply_variant(img, v, PPCFG); T.save_jpeg(o, out, q)
        recs.append(dict(name=r.name, set=r.set, source_id=r.source_id, postprocess=v, path=str(out)))
R = pd.DataFrame(recs)

# 2) 메모 위치 (실험 4 설정 그대로)
cfg = yaml.safe_load(open(EXP/"config.yaml")); loc = cfg["locate"]
bx_csv = OUT/"boxes.csv"; cache = pd.read_csv(bx_csv).set_index("path").to_dict("index") if bx_csv.exists() else {}
det = None; out = []
for r in R.itertuples():
    if r.path in cache: out.append(cache[r.path]); continue
    det = det or HFZeroShotDetector(loc["model_id"], loc["queries"], loc["box_threshold"], loc["text_threshold"])
    with Image.open(r.path) as im: ch = choose_box(det.detect(im), loc["min_area_frac"], loc["max_area_frac"])
    out.append(dict(x0=(ch or (FULL_BOX, 0))[0][0], y0=(ch or (FULL_BOX, 0))[0][1], x1=(ch or (FULL_BOX, 0))[0][2], y1=(ch or (FULL_BOX, 0))[0][3],
                    score=(ch[1] if ch else 0.0), source="auto" if ch else "fallback"))
B = pd.DataFrame(out); R = pd.concat([R, B[["x0", "y0", "x1", "y1", "score", "source"]]], axis=1)
R[["path", "x0", "y0", "x1", "y1", "score", "source"]].to_csv(bx_csv, index=False)
del det; gc.collect(); torch.cuda.empty_cache()

# 3) 모델 2 실험 4
ck = torch.load(EXP/"checkpoints/best_model.pth", map_location="cpu", weights_only=False)
m = build_model(ck["config"]["model"], pretrained=False); m.load_state_dict(ck["model_state"]); m.cuda().eval()
tf = build_transform(ck["config"], train=False); thr = float(ck["threshold"]); p2 = []
with torch.inference_mode():
    for r in R.itertuples():
        with Image.open(r.path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            p2.append(float(torch.sigmoid(m(tf(im, (r.x0, r.y0, r.x1, r.y1)).unsqueeze(0).cuda()).float()[0])))
R["p_m2"] = p2; R["pred_m2"] = (R.p_m2 >= thr).astype(int)
del m; gc.collect(); torch.cuda.empty_cache()

# 4) 모델 1 CLIP (3크롭)
M1 = W/"model1"; sys.path.insert(0, str(M1))
for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]: sys.modules.pop(k)
sys.path.remove(str(REPO))
from src.models.model1_full_image.clip_detector import load_clip_detector, analyze_clip
cd = load_clip_detector(M1/"outputs/model1/clip_probe_v3data_3crop/clip_head.pt")
pc = []
for r in R.itertuples():
    with Image.open(r.path) as im: im.load(); pc.append(analyze_clip(cd, im, with_heatmap=False, views=3)["fake_probability"])
R["p_clip"] = pc
R.to_csv(OUT/"scores.csv", index=False); print("done", len(R), "thr", thr)
