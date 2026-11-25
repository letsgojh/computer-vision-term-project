"""검사 실행: 점수 계산, 실제 사진 기준 저장, 판정, AUROC."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from detector.images import open_rgb
from detector.rdd import ImageDebiaser, LatentDebiaser, LayerLPIPS, combine_image_scores, load_vae
from detector.scores import (
    REALNESS_KEYS,
    VERDICT_NO_REFERENCE,
    describe_distribution,
    fuse_percentiles,
    percentile_rank,
    reference_values,
    verdict_from_percentile,
)
from detector.warpad import WarpadScorer

FUSION_KEYS = ("s_image", "warpad", "s_latent")

CSV_COLUMNS = [
    ("file", "파일"),
    ("width", "원본너비"),
    ("height", "원본높이"),
    ("f_ae", "재구성오차_fAE"),
    ("s_image", "이미지디바이어싱_Simage"),
    ("s_image_percentile", "Simage_백분위"),
    ("s_latent", "잠재디바이어싱_Slatent"),
    ("s_latent_percentile", "Slatent_백분위"),
    ("s_rdd", "RDD점수"),
    ("warpad", "WaRPAD점수"),
    ("warpad_percentile", "WaRPAD_백분위"),
    ("fused_percentile", "종합_백분위"),
    ("verdict", "판정"),
    ("error", "오류"),
]


@dataclass
class Config:
    preset: str = "fast"
    vae_names: list[str] = field(default_factory=lambda: ["sd15"])
    lambda_r: float = 0.5
    kernel_size: int = 3
    blur_sigma: float = 0.8
    crop_size: int = 512
    use_latent: bool = False
    use_warpad: bool = True
    timestep: float = 0.05
    noise_samples: int = 4
    dino_id: str = "facebook/dinov2-base"
    prep_size: int = 448
    patch_size: int = 224
    alpha: float = 0.1
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def comparable(self) -> dict[str, Any]:
        data = self.to_dict()
        data.pop("preset", None)
        return data


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def score_files(paths: list[Path], config: Config) -> list[dict[str, Any]]:
    device = torch.device(config.device)
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        try:
            image = open_rgb(path)
            width, height = image.size
        except Exception as error:
            records[str(path)] = {"file": str(path), "error": f"이미지를 열 수 없습니다: {error}"}
            continue
        records[str(path)] = {"file": str(path), "width": width, "height": height, "_parts": []}

    usable = [path for path in paths if "_parts" in records[str(path)]]
    if not usable:
        return list(records.values())

    print(f"장치: {device}. 오토인코더 {len(config.vae_names)}개로 재구성 점수를 계산합니다.")
    distance = LayerLPIPS(device)
    for name in config.vae_names:
        print(f"VAE 불러오는 중: {name}")
        vae = load_vae(name, device)
        debiaser = ImageDebiaser(
            vae,
            distance,
            crop_size=config.crop_size,
            kernel_size=config.kernel_size,
            sigma=config.blur_sigma,
            lambda_r=config.lambda_r,
        )
        for path in tqdm(usable, desc=f"Simage {name}"):
            key = str(path)
            if "error" in records[key]:
                continue
            try:
                records[key]["_parts"].append(debiaser.score_path(path))
            except Exception as error:
                _note(records[key], f"{name} 재구성 실패: {error}")
        del vae, debiaser
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del distance
    if device.type == "cuda":
        torch.cuda.empty_cache()

    latent_values: dict[str, list[float]] = {str(path): [] for path in usable}
    if config.use_latent:
        print(
            "잠재 디바이어싱을 계산합니다. "
            f"이미지마다 모델당 UNet 순전파가 {config.noise_samples * 2}번입니다."
        )
        for name in config.vae_names:
            print(f"확산 모델 불러오는 중: {name}")
            latent_model = LatentDebiaser(
                name,
                device,
                timestep=config.timestep,
                noise_samples=config.noise_samples,
                crop_size=config.crop_size,
            )
            for path in tqdm(usable, desc=f"Slatent {name}"):
                key = str(path)
                if "error" in records[key]:
                    continue
                try:
                    latent_values[key].append(latent_model.score_path(path))
                except Exception as error:
                    _note(records[key], f"{name} 잠재 점수 실패: {error}")
            latent_model.close()

    warpad_values: dict[str, float] = {}
    if config.use_warpad:
        print(f"WaRPAD 불러오는 중: {config.dino_id}, 변 길이 {config.prep_size}")
        warpad = WarpadScorer(
            config.dino_id,
            device,
            prep_size=config.prep_size,
            patch_size=config.patch_size,
            alpha=config.alpha,
        )
        for path in tqdm(usable, desc="WaRPAD"):
            key = str(path)
            if "error" in records[key]:
                continue
            try:
                warpad_values[key] = warpad.score_path(path)
            except Exception as error:
                _note(records[key], f"WaRPAD 실패: {error}")
        warpad.close()

    finished = []
    for path in paths:
        record = records[str(path)]
        parts = record.pop("_parts", [])
        if parts:
            latent = None
            values = latent_values.get(str(path), [])
            if values:
                latent = float(sum(values) / len(values))
            record.update(combine_image_scores(parts, latent))
        if str(path) in warpad_values:
            record["warpad"] = warpad_values[str(path)]
        notes = record.pop("_notes", [])
        if notes:
            record["error"] = " | ".join(notes)
        if not any(key in record for key in ("s_image", "warpad", "s_latent")) and "error" not in record:
            record["error"] = record.get("error", "점수를 계산하지 못했습니다.")
        finished.append(record)
    return finished


def save_reference(path: str | Path, config: Config, records: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.to_dict(),
        "records": _public_records(records),
        "summary": {
            key: describe_distribution(reference_values(records, key))
            for key in REALNESS_KEYS
            if reference_values(records, key)
        },
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_reference(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _public_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for record in records:
        item = {}
        for key, value in record.items():
            if key.startswith("_"):
                continue
            if isinstance(value, float) and not math.isfinite(value):
                item[key] = None
            else:
                item[key] = value
        cleaned.append(item)
    return cleaned


def attach_reference(
    records: list[dict[str, Any]],
    reference_records: list[dict[str, Any]],
    low: float,
    mid: float,
) -> list[dict[str, Any]]:
    pools = {key: reference_values(reference_records, key) for key in FUSION_KEYS}
    judged = []
    for record in records:
        item = dict(record)
        parts = {}
        for key, pool in pools.items():
            if key in item and pool:
                rank = percentile_rank(item[key], pool)
                item[f"{key}_percentile"] = rank
                parts[key] = rank
        if parts:
            fused = fuse_percentiles(parts)
            item["fused_percentile"] = fused
            item["verdict"] = verdict_from_percentile(fused, low=low, mid=mid)
        elif "error" in item:
            item["verdict"] = "오류"
        else:
            item["verdict"] = VERDICT_NO_REFERENCE
        judged.append(item)
    return judged


def write_csv(path: str | Path, records: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    rename = {key: label for key, label in CSV_COLUMNS if key in frame.columns}
    ordered = [key for key, _label in CSV_COLUMNS if key in frame.columns]
    extra = [column for column in frame.columns if column not in ordered]
    frame = frame[ordered + extra].rename(columns=rename)
    frame.to_csv(destination, index=False, encoding="utf-8-sig")


def print_record_line(index: int, total: int, record: dict[str, Any]) -> None:
    name = Path(str(record.get("file", ""))).name
    verdict = record.get("verdict", "")
    if "fused_percentile" in record and _finite(record.get("fused_percentile")):
        print(f"[{index}/{total}] {name}  종합 백분위 {record['fused_percentile']:.1f}  {verdict}")
        return
    bits = []
    for key, label in (("s_image", "Simage"), ("warpad", "WaRPAD"), ("f_ae", "fAE")):
        if _finite(record.get(key)):
            bits.append(f"{label}={record[key]:.4f}")
    detail = "  ".join(bits) if bits else record.get("error", "")
    print(f"[{index}/{total}] {name}  {detail}  {verdict}".rstrip())


def print_summary(records: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for record in records:
        label = str(record.get("verdict", "점수만 계산"))
        counts[label] = counts.get(label, 0) + 1
    print(f"검사 {len(records)}장")
    for label, count in counts.items():
        print(f"  {label}: {count}장")


def print_distribution(title: str, records: list[dict[str, Any]]) -> None:
    print(title)
    for key, label in (
        ("f_ae", "재구성 오차 fAE"),
        ("s_image", "Simage"),
        ("s_latent", "Slatent"),
        ("warpad", "WaRPAD"),
    ):
        stats = describe_distribution(reference_values(records, key))
        if not stats:
            continue
        print(
            f"  {label}: n={stats['count']}  평균={stats['mean']:.4f}  "
            f"하위5%={stats['p05']:.4f}  중앙={stats['p50']:.4f}  상위5%={stats['p95']:.4f}"
        )


def configs_match(saved: dict[str, Any], config: Config) -> bool:
    left = dict(saved)
    left.pop("preset", None)
    right = config.comparable()
    return left == right


def evaluate_auroc(real_records: list[dict[str, Any]], fake_records: list[dict[str, Any]]) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    scores = {}
    for key in ("s_image", "s_latent", "s_rdd", "warpad", "f_ae"):
        real = reference_values(real_records, key)
        fake = reference_values(fake_records, key)
        if len(real) < 2 or len(fake) < 2:
            continue
        labels = np.array([1] * len(real) + [0] * len(fake))
        values = np.array(real + fake, dtype=np.float64)
        scores[key] = float(roc_auc_score(labels, values))
    return scores


def _note(record: dict[str, Any], message: str) -> None:
    record.setdefault("_notes", []).append(message)
    print(f"  건너뜀: {Path(str(record.get('file', ''))).name} — {message}")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))
