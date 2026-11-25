"""중고 사진이 AI 생성인지 실제 사진인지 점수화한다.

사용 예:
  python detect.py calibrate --real data/real
  python detect.py predict --input data/query
  python detect.py evaluate --real data/real --fake data/fake
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings("ignore", message="Arguments other than a weight enum")
warnings.filterwarnings("ignore", message="The `local_dir_use_symlinks` argument is deprecated")
warnings.filterwarnings("ignore", message="You are sending unauthenticated requests")

from detector.images import list_images
from detector.pipeline import (
    Config,
    attach_reference,
    configs_match,
    evaluate_auroc,
    load_reference,
    pick_device,
    print_distribution,
    print_record_line,
    print_summary,
    save_reference,
    score_files,
    write_csv,
)


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


def build_config(args: argparse.Namespace) -> Config:
    if args.preset == "paper":
        vae_names = ["sd14", "sd2"]
        dino_id = "facebook/dinov2-large"
        prep_size = 896
    else:
        vae_names = ["sd15"]
        dino_id = "facebook/dinov2-base"
        prep_size = 448
    if args.vae:
        vae_names = list(args.vae)
    if args.dino:
        dino_id = args.dino
    if args.prep_size:
        prep_size = args.prep_size
    return Config(
        preset=args.preset,
        vae_names=vae_names,
        lambda_r=args.lambda_r,
        kernel_size=args.kernel_size,
        blur_sigma=args.blur_sigma,
        crop_size=args.crop_size,
        use_latent=args.latent and not args.no_latent,
        use_warpad=not args.no_warpad,
        timestep=args.timestep,
        noise_samples=args.noise_samples,
        dino_id=dino_id,
        prep_size=prep_size,
        alpha=args.alpha,
        device=str(pick_device(args.device)),
    )


def _take(paths: list[Path], limit: int | None) -> list[Path]:
    if limit is None:
        return paths
    return paths[:limit]


def _require_images(paths: list[Path], folder: str, role: str) -> None:
    if paths:
        return
    raise SystemExit(
        f"{role} 사진을 찾지 못했습니다: {folder}\n"
        "jpg, jpeg, png, webp, bmp 파일을 그 폴더에 넣은 뒤 다시 실행하세요."
    )


def command_calibrate(args: argparse.Namespace) -> None:
    config = build_config(args)
    paths = _take(list_images(args.real), args.max_images)
    _require_images(paths, args.real, "기준이 될 실제")
    if len(paths) < 20:
        print(f"실제 사진이 {len(paths)}장입니다. 백분위가 안정되려면 같은 조건의 사진을 30장 이상 두는 편이 좋습니다.")
    _warn_cpu_latent(config)
    records = score_files(paths, config)
    save_reference(args.reference, config, records)
    print_distribution("실제 사진 점수 분포", records)
    print(f"기준 저장: {args.reference}")


def command_predict(args: argparse.Namespace) -> None:
    config = build_config(args)
    paths = _take(list_images(args.input), args.max_images)
    _require_images(paths, args.input, "검사할")
    _warn_cpu_latent(config)
    records = score_files(paths, config)
    reference_path = Path(args.reference)
    if reference_path.exists():
        reference = load_reference(reference_path)
        if not configs_match(reference.get("config", {}), config):
            print("저장해 둔 실제 사진 기준과 이번 설정이 다릅니다. 백분위는 같은 설정으로 calibrate한 분포와 비교할 때 의미가 있습니다.")
        records = attach_reference(
            records,
            reference.get("records", []),
            low=args.low_percentile,
            mid=args.mid_percentile,
        )
    else:
        print("실제 사진 기준 파일이 없습니다. 점수만 계산합니다. 판정을 보려면 먼저 calibrate를 실행하세요.")
        for record in records:
            record.setdefault("verdict", "기준 없음")
    for index, record in enumerate(records, start=1):
        print_record_line(index, len(records), record)
    write_csv(args.output, records)
    print_summary(records)
    print(f"결과 저장: {args.output}")


def command_evaluate(args: argparse.Namespace) -> None:
    config = build_config(args)
    real_paths = _take(list_images(args.real), args.max_images)
    fake_paths = _take(list_images(args.fake), args.max_images)
    _require_images(real_paths, args.real, "실제")
    _require_images(fake_paths, args.fake, "AI")
    _warn_cpu_latent(config)
    print(f"실제 {len(real_paths)}장, AI {len(fake_paths)}장")
    real_records = score_files(real_paths, config)
    fake_records = score_files(fake_paths, config)
    for record in real_records:
        record["label"] = "real"
    for record in fake_records:
        record["label"] = "fake"
    write_csv(args.output, real_records + fake_records)
    print("AUROC (1에 가까울수록 실제와 AI가 잘 갈립니다. 0.5는 구분하지 못한 경우입니다.)")
    found = evaluate_auroc(real_records, fake_records)
    if not found:
        print("점수가 모이지 않아 AUROC를 계산하지 못했습니다.")
    for key, value in found.items():
        print(f"  {key}: {value:.4f}")
    print(f"결과 저장: {args.output}")


def _warn_cpu_latent(config: Config) -> None:
    if config.use_latent and config.device.startswith("cpu"):
        print("CPU에서 잠재 점수는 이미지 한 장에 수 분 걸릴 수 있습니다. 재구성·WaRPAD만 보려면 --no-latent 로 실행하세요.")
    if config.device.startswith("cpu"):
        print("CPU로 계산합니다. 첫 실행은 모델 다운로드가 포함되고, 이후에는 장당 수십 초 정도를 보세요.")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--preset", choices=["fast", "paper"], default="fast")
    common.add_argument("--device", default=None, help="cpu 또는 cuda. 비우면 자동 선택")
    common.add_argument("--vae", nargs="+", default=None, help="sd15, sd14, sd2 또는 Hugging Face 저장소")
    common.add_argument("--latent", action="store_true", help="RDD 잠재 디바이어싱까지 계산")
    common.add_argument("--no-latent", action="store_true", help="잠재 점수를 끄기")
    common.add_argument("--no-warpad", action="store_true", help="WaRPAD를 끄고 재구성 점수만 사용")
    common.add_argument("--lambda-r", type=float, default=0.5)
    common.add_argument("--kernel-size", type=int, default=3)
    common.add_argument("--blur-sigma", type=float, default=0.8)
    common.add_argument("--crop-size", type=int, default=512)
    common.add_argument("--timestep", type=float, default=0.05)
    common.add_argument("--noise-samples", type=int, default=4)
    common.add_argument("--dino", default=None, help="base, large 또는 Hugging Face DINOv2 이름")
    common.add_argument("--prep-size", type=int, default=None)
    common.add_argument("--alpha", type=float, default=0.1)
    common.add_argument("--max-images", type=int, default=None)

    parser = argparse.ArgumentParser(
        description="중고 상품 사진이 AI 생성인지 실제 사진인지 학습 없이 검사합니다."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    calibrate = sub.add_parser("calibrate", parents=[common], help="실제 사진으로 점수 기준을 저장")
    calibrate.add_argument("--real", default="data/real")
    calibrate.add_argument("--reference", default="outputs/real_reference.json")
    calibrate.set_defaults(func=command_calibrate)

    predict = sub.add_parser("predict", parents=[common], help="검사할 사진에 판정을 붙임")
    predict.add_argument("--input", default="data/query")
    predict.add_argument("--reference", default="outputs/real_reference.json")
    predict.add_argument("--output", default="outputs/predictions.csv")
    predict.add_argument("--low-percentile", type=float, default=5.0)
    predict.add_argument("--mid-percentile", type=float, default=20.0)
    predict.set_defaults(func=command_predict)

    evaluate = sub.add_parser("evaluate", parents=[common], help="실제/AI 폴더가 모두 있을 때 AUROC")
    evaluate.add_argument("--real", default="data/real")
    evaluate.add_argument("--fake", default="data/fake")
    evaluate.add_argument("--output", default="outputs/evaluation.csv")
    evaluate.set_defaults(func=command_evaluate)
    return parser


def main() -> None:
    _configure_stdio()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
