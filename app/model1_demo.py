"""Gradio demo for Model 1: is this marketplace photo AI-generated?

Run from the repository root:
  python -m app.model1_demo --checkpoint outputs/model1/defactify_native_crop/checkpoints/best_model.pth

Shows the AI probability, a three-level verdict (pass / suspicious / AI) and, for AI
verdicts, a Grad-CAM heatmap. Both thresholds are sliders; "save" writes them back to
thresholds.json so the team default changes for everyone using that file.
"""

import argparse
from pathlib import Path

import gradio as gr

from src.models.model1_full_image.inference import DISCLAIMER, analyze, load_detector
from src.models.model1_full_image.thresholds import LABELS_KO, load_thresholds, save_thresholds, verdict

HEATMAP_NOTE = "히트맵은 모델이 판단에 주목한 영역입니다. 조작·생성된 위치를 증명하지 않습니다."


def build_app(checkpoint_path, thresholds_path, device="auto", version=None):
    model, checkpoint, device = load_detector(checkpoint_path, device)
    version = version or Path(checkpoint_path).resolve().parent.parent.name
    thresholds_path = Path(thresholds_path)
    stored = load_thresholds(thresholds_path) if thresholds_path.exists() else {"suspicious": 0.5, "ai": checkpoint["threshold"]}

    def run(image, suspicious, ai):
        if image is None:
            return "이미지를 올려 주세요.", None, ""
        if suspicious > ai:
            return "의심 임계값은 AI 임계값보다 클 수 없습니다.", None, ""
        result = analyze(model, checkpoint, device, image, with_heatmap=True)
        score = result["fake_probability"]
        level = verdict(score, suspicious, ai)
        crops = result["crop_probabilities"]
        text = (f"## {LABELS_KO[level]}\n\n"
                f"**AI 생성 확률: {score:.1%}**  (의심 ≥ {suspicious:.2f}, AI ≥ {ai:.2f})\n\n"
                f"이미지 조각 {len(crops)}개 평균 · 최저 {min(crops):.1%} · 최고 {max(crops):.1%}")
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
                    f"**모델 버전:** {version} · 학습 epoch {checkpoint['epoch']} · 개발 기록: `src/models/model1_full_image/DEVELOPMENT_LOG.md`")
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
                heatmap = gr.Image(label="Grad-CAM 히트맵 (AI 판정 시)", interactive=False)
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
    args = parser.parse_args()
    thresholds = args.thresholds or Path(args.checkpoint).resolve().parent.parent / "thresholds.json"
    build_app(args.checkpoint, thresholds, args.device, args.version).launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
