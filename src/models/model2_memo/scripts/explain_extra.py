"""Grad-CAM 보강: (1) 가리기 실험(occlusion) (3) 조작 마스크와 겹침.
test A(original)마다 Grad-CAM과 가리기 지도를 만들고, 실제로 고친 영역(마스크)에 열이 얼마나 들어가는지 잰다.
사용: python -m src.models.model2_memo.scripts.explain_extra outputs/model2_realAC2 [--split test] [--figs 12]"""
import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image, ImageDraw, ImageOps

REPO = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(REPO))
from src.models.model2_memo.dataset import build_transform, crop_region, read_metadata, select_rows
from src.models.model2_memo.boxes import expand_box, to_pixels
from src.models.model2_memo.explain import gradcam, crop_cam, overlay
from src.models.model2_memo.boxes import read_boxes
from src.models.model2_memo.model import build_model
from src.models.model2_memo.utils import load_config

ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--split", default="test")
ap.add_argument("--figs", type=int, default=12); ap.add_argument("--patch", type=int, default=48); ap.add_argument("--stride", type=int, default=24)
a = ap.parse_args()
E = REPO / a.exp; OUT = E / "explain_extra"; (OUT / "figs").mkdir(parents=True, exist_ok=True)
D = Path(os.environ.get("CVTP_DATA", Path.home() / "cvtp_data"))
ck = torch.load(E / "checkpoints/best_model.pth", map_location="cpu", weights_only=False)
cfg = ck["config"]; src = load_config(E / "config.yaml"); thr = float(ck["threshold"])
model = build_model(cfg["model"], pretrained=False); model.load_state_dict(ck["model_state"]); model.cuda().eval()
tf = build_transform(cfg, train=False); S = int(cfg["model"]["input_size"]); margin = cfg["crop"]["margin"]
boxes = read_boxes(E / "memo_boxes.csv")
df = read_metadata(D / "data/splits/model1_eval.csv", src["data"]["root"], src["data"].get("leakage_columns", ("item_id",)), False)
rows = select_rows(df, a.split)
rows = rows[(rows.fake_type == "A") & (rows.postprocess == "original")]
print("대상 A:", len(rows), flush=True)


def mask_crop(stem, im_size, box):
    m = Image.open(D / "data/fake/masks" / f"{stem}_mask.png").convert("L")
    if m.size != im_size: m = m.resize(im_size, Image.NEAREST)
    return np.asarray(m.crop(to_pixels(expand_box(box, margin), *im_size)), dtype=np.float32) / 255.0


@torch.inference_mode()
def occlusion(x):
    """patch×patch 회색(정규화 후 0) 칸을 stride마다 덮고 가짜 확률 하락을 모은다 → S×S 지도."""
    l0 = model(x).float()[0]; p0 = float(torch.sigmoid(l0))  # 확률은 1.00에서 포화되므로 로짓 하락으로 잰다
    pos = [(i, j) for i in range(0, S - a.patch + 1, a.stride) for j in range(0, S - a.patch + 1, a.stride)]
    drop = torch.zeros(S, S, device="cuda"); cnt = torch.zeros(S, S, device="cuda")
    for k in range(0, len(pos), 64):
        chunk = pos[k:k + 64]; b = x.repeat(len(chunk), 1, 1, 1)
        for n, (i, j) in enumerate(chunk): b[n, :, i:i + a.patch, j:j + a.patch] = 0
        p = model(b).float()
        for n, (i, j) in enumerate(chunk):
            drop[i:i + a.patch, j:j + a.patch] += (l0 - p[n]); cnt[i:i + a.patch, j:j + a.patch] += 1
    m = (drop / cnt.clamp(min=1)).clamp(min=0).cpu().numpy()
    return p0, m, float(m.max())


def diff_crop(fake_im, source_id, box):
    """실제로 바뀐 픽셀: 가짜와 원본(같은 후처리)의 차이 > 30, 5px 팽창."""
    from PIL import ImageFilter
    src = ImageOps.exif_transpose(Image.open(D / "data/processed/original" / f"{source_id}.jpg")).convert("RGB")
    if src.size != fake_im.size: src = src.resize(fake_im.size)
    px = to_pixels(expand_box(box, margin), *fake_im.size)
    d = np.abs(np.asarray(fake_im.crop(px), dtype=np.int16) - np.asarray(src.crop(px), dtype=np.int16)).max(2)
    m = Image.fromarray(((d > 30) * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.MaxFilter(11))
    return np.asarray(m, dtype=np.float32) / 255.0


def stats(heat, mask):
    inside = mask > 0.5
    if inside.sum() == 0 or heat.sum() <= 0: return np.nan, np.nan, inside.mean()
    energy = float((heat * inside).sum() / heat.sum())
    peak = np.unravel_index(np.argmax(heat), heat.shape)
    return energy, float(inside[peak]), float(inside.mean())


recs = []; panels = []
for r in rows.to_dict("records"):
    rec = boxes[r["image_path"]]; stem = Path(r["image_path"]).stem
    with Image.open(r["_resolved_path"]) as im:
        im.load(); im = ImageOps.exif_transpose(im).convert("RGB")
        x = tf(im, rec["box"]).unsqueeze(0).cuda()
        crop = crop_region(im, rec["box"], margin)
        cam, prob = gradcam(model, x)
        p0, occ, occ_max = occlusion(x)
        g = crop_cam(cam, crop.size, S)
        o = crop_cam(occ / occ_max if occ_max > 0 else occ, crop.size, S)
        mk = mask_crop(stem, im.size, rec["box"])
        if mk.shape != g.shape: mk = np.asarray(Image.fromarray(mk).resize(crop.size, Image.NEAREST))
        dk = diff_crop(im, r["source_id"], rec["box"])
    ge, gp, area = stats(g, mk); oe, op, _ = stats(o, mk)
    gd, gdp, darea = stats(g, dk); od, odp, _ = stats(o, dk)
    recs.append(dict(image=stem, p_fake=prob, pred=int(prob >= thr), mask_area=area, cam_in_mask=ge, cam_peak_in_mask=gp,
                     occ_in_mask=oe, occ_peak_in_mask=op, occ_max_drop=occ_max,
                     diff_area=darea, cam_in_diff=gd, cam_peak_in_diff=gdp, occ_in_diff=od, occ_peak_in_diff=odp))
    panels.append((stem, prob, crop, g, o, dk))
    print(f"{len(recs)}/{len(rows)} {stem} p={prob:.2f} cam_in={ge:.2f} occ_in={oe:.2f} area={area:.2f}", flush=True)

res = pd.DataFrame(recs); res.to_csv(OUT / "mask_overlap.csv", index=False)


def outline(img, mk):
    img = img.copy(); e = Image.fromarray((mk > 0.5).astype(np.uint8) * 255)
    edge = np.asarray(e.filter(__import__("PIL.ImageFilter", fromlist=["x"]).FIND_EDGES)) > 0
    arr = np.asarray(img).copy(); arr[edge] = (0, 255, 0); return Image.fromarray(arr)


# 그림: 맞힌 것과 놓친 것을 섞어 figs장
res_i = res.reset_index()
pick = list(res_i[res_i.pred == 1].sort_values("p_fake", ascending=False).index[: a.figs * 2 // 3]) + \
       list(res_i[res_i.pred == 0].sort_values("p_fake").index[: a.figs - a.figs * 2 // 3])
H = 220
for k, i in enumerate(pick):
    stem, prob, crop, g, o, mk = panels[i]
    tiles = [outline(crop, mk), overlay(crop, g), overlay(crop, o)]
    tiles = [t.resize((max(1, round(t.width * H / t.height)), H)) for t in tiles]
    W = sum(t.width for t in tiles) + 20; sh = Image.new("RGB", (W, H + 22), "white"); d = ImageDraw.Draw(sh); xx = 0
    for t, name in zip(tiles, ["crop + changed pixels (green)", "Grad-CAM", "occlusion (logit drop)"]):
        sh.paste(t, (xx, 22)); d.text((xx + 2, 4), name, fill="black"); xx += t.width + 10
    d.text((W - 260, 4), f"{stem[:30]} P={prob:.2f}", fill="red")
    sh.save(OUT / "figs" / f"{k:02d}_{stem}.png")

def summ(s):
    return (f"[바뀐 픽셀 기준] Grad-CAM {s.cam_in_diff.mean():.1%} (최고점 안 {s.cam_peak_in_diff.mean():.1%}) | 가리기 {s.occ_in_diff.mean():.1%} "
            f"(최고점 안 {s.occ_peak_in_diff.mean():.1%}) | 무작위 기준 {s.diff_area.mean():.1%}\n    [편집 마스크 기준] "
            f"Grad-CAM 열 중 고친 영역 안 비율 평균 {s.cam_in_mask.mean():.1%} (최고점이 영역 안: {s.cam_peak_in_mask.mean():.1%}) | "
            f"가리기 실험 {s.occ_in_mask.mean():.1%} (최고점 안: {s.occ_peak_in_mask.mean():.1%}) | 무작위 기준(영역 넓이) {s.mask_area.mean():.1%}")
txt = [f"[{a.exp} / {a.split} A original {len(res)}장]", "전체: " + summ(res),
       "맞힌 것: " + summ(res[res.pred == 1]), "놓친 것: " + summ(res[res.pred == 0])]
(OUT / "summary.txt").write_text("\n".join(txt) + "\n", encoding="utf-8"); print("\n".join(txt))
