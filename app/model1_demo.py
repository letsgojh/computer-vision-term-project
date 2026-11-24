"""Gradio demo for Model 1: is this marketplace photo AI-generated?

Run from the repository root:
  python -m app.model1_demo --checkpoint outputs/model1/defactify_native_crop/checkpoints/best_model.pth
  python -m app.model1_demo --checkpoint outputs/model1/clip_probe_v3data/clip_head.pt   (final model, CLIP)
  add --local-checkpoint outputs/model1/local_native_v6/checkpoints/best_model.pth for partial edits

Shows the AI probability, a three-level verdict (pass / suspicious / AI) and, for AI
verdicts, a Grad-CAM heatmap. Both thresholds are sliders; "save" writes them back to
thresholds.json so the team default changes for everyone using that file.
With --local-checkpoint, the photo is also scanned at native resolution for partial AI edits
(v6, thresholds from that run's thresholds.json); the final verdict is the more severe of the two,
and suspicious tiles are shown on the photo.
"""

import argparse
import json
from pathlib import Path

import gradio as gr

from src.models.model1_full_image.inference import DISCLAIMER, analyze, load_detector
from src.models.model1_full_image.thresholds import LABELS_KO, load_thresholds, save_thresholds, verdict

HEATMAP_NOTE = "히트맵은 모델이 판단에 주목한 영역입니다. 조작·생성된 위치를 증명하지 않습니다."
LOCAL_NOTE = "표시된 칸은 카메라 노이즈가 없어 AI로 다시 그려졌을 가능성이 높은 영역입니다. 촬영한 화면(모니터)도 높게 나올 수 있습니다."
SEVERITY = {"pass": 0, "suspicious": 1, "ai": 2}


def is_clip(checkpoint_path):
    return Path(checkpoint_path).name == "clip_head.pt"


def build_app(checkpoint_path, thresholds_path, device="auto", version=None, local_checkpoint=None):
    local = None
    if local_checkpoint:
        from src.models.model1_full_image.local_detector import scan, tile_heatmap
        local_model = load_detector(local_checkpoint, device)
        # Logit thresholds (not 0-1 probabilities), so not load_thresholds()
        local_th = json.loads((Path(local_checkpoint).resolve().parent.parent / "thresholds.json").read_text(encoding="utf-8"))
        local = (local_model, local_th, scan, tile_heatmap)
    if is_clip(checkpoint_path):
        from src.models.model1_full_image.clip_detector import analyze_clip, load_clip_detector
        detector = load_clip_detector(checkpoint_path, device)
        score_image = lambda image: analyze_clip(detector, image)
        version = version or Path(checkpoint_path).resolve().parent.name
        model_note = "CLIP ViT-L/14 고정 + 선형 분류기 · 사진 가운데 정사각형을 봅니다"
        fallback = {"suspicious": 0.5, "ai": 0.8}
    else:
        model, checkpoint, device = load_detector(checkpoint_path, device)
        score_image = lambda image: analyze(model, checkpoint, device, image, with_heatmap=True)
        version = version or Path(checkpoint_path).resolve().parent.parent.name
        model_note = f"학습 epoch {checkpoint['epoch']}"
        fallback = {"suspicious": 0.5, "ai": checkpoint["threshold"]}
    thresholds_path = Path(thresholds_path)
    stored = load_thresholds(thresholds_path) if thresholds_path.exists() else fallback

    def run(image, suspicious, ai):
        if image is None:
            return "이미지를 올려 주세요.", None, ""
        if suspicious > ai:
            return "의심 임계값은 AI 임계값보다 클 수 없습니다.", None, ""
        result = score_image(image)
        score = result["fake_probability"]
        level = verdict(score, suspicious, ai)
        crops = result["crop_probabilities"]
        text = f"## {LABELS_KO[level]}\n\n**AI 생성 확률: {score:.1%}**  (의심 ≥ {suspicious:.2f}, AI ≥ {ai:.2f})"
        if len(crops) > 1:
            text += f"\n\n이미지 조각 {len(crops)}개 평균 · 최저 {min(crops):.1%} · 최고 {max(crops):.1%}"
        if local is not None:
            (model_l, ckpt_l, dev_l), th, scan, tile_heatmap = local
            logits, corners, scanned = scan(model_l, ckpt_l, dev_l, image)
            local_level = "ai" if logits.max() >= th["ai"] else "suspicious" if logits.max() >= th["suspicious"] else "pass"
            overall = max(level, local_level, key=SEVERITY.get)
            text = (f"## {LABELS_KO[overall]}\n\n**전체 생성 판별 (CLIP): {score:.1%}** → {LABELS_KO[level]}  (의심 ≥ {suspicious:.2f}, AI ≥ {ai:.2f})\n\n"
                    f"**부분 편집 판별 (원본 해상도 {scanned.width}×{scanned.height}, 조각 {len(logits)}개): 최고 점수 {logits.max():.1f}** → "
                    f"{LABELS_KO[local_level]}  (의심 ≥ {th['suspicious']:.1f}, AI ≥ {th['ai']:.1f})")
            if local_level != "pass":
                return text, tile_heatmap(scanned, logits, corners, th["suspicious"], th["ai"]), f"{LOCAL_NOTE}\n\n<small>{DISCLAIMER}</small>"
        if level == "ai":
            return text, result["heatmap"], f"{HEATMAP_NOTE}\n\n<small>{DISCLAIMER}</small>"
        return text, None, ""

    def save(suspicious, ai):
        if suspicious > ai:
            return "저장 실패: 의심 임계값이 AI 임계값보다 큽니다."
        values = load_thresholds(thresholds_path) if thresholds_path.exists() else {}
        values.update({"suspicious": float(suspicious), "ai": float(ai), "edited_in_demo": True})
        save_thresholds(thresholds_path, values)
        return f"저장했습니다: {thresholds_path}"

    with gr.Blocks(title="AI 생성 이미지 판별 (모델 1)") as demo:
        gr.Markdown("# 중고거래 사진 AI 생성 판별\n사진을 올리면 AI로 생성됐을 확률과 판정을 보여줍니다.\n\n"
                    f"**모델 버전:** {version} · {model_note} · 개발 기록: `src/models/model1_full_image/DEVELOPMENT_LOG.md`")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="사진")
                suspicious = gr.Slider(0, 1, value=stored["suspicious"], step=0.001, label="의심 임계값 (이 이상이면 '의심')")
                ai = gr.Slider(0, 1, value=stored["ai"], step=0.001, label="AI 임계값 (이 이상이면 'AI 생성')")
                with gr.Row():
                    button = gr.Button("판별", variant="primary")
                    save_button = gr.Button("임계값 저장")
                saved = gr.Markdown()
            with gr.Column():
                verdict_text = gr.Markdown()
                heatmap = gr.Image(label="히트맵 (AI 판정 또는 부분 편집 의심 시)", interactive=False)
                note = gr.Markdown()
        button.click(run, [image, suspicious, ai], [verdict_text, heatmap, note])
        save_button.click(save, [suspicious, ai], saved)
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds", help="Default: thresholds.json next to the checkpoints folder")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--version", help='Shown in the header, e.g. "v1 — Defactify, 노이즈 지름길 있음". Default: run folder name')
    parser.add_argument("--local-checkpoint", help="v6 native-resolution partial-edit detector (optional)")
    args = parser.parse_args()
    run_dir = Path(args.checkpoint).resolve().parent  # CLIP head sits in the run folder; CNN checkpoints in run/checkpoints
    thresholds = args.thresholds or (run_dir if is_clip(args.checkpoint) else run_dir.parent) / "thresholds.json"
    build_app(args.checkpoint, thresholds, args.device, args.version, args.local_checkpoint).launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
