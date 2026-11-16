"""메모 위치 찾기: zero-shot 탐지, 수동 박스(train만), 전체 이미지 fallback, 유형별 리포트."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from .boxes import FULL_BOX, Box, BoxRecord, read_boxes, validate_box, write_boxes
from .dataset import POSTPROCESSES, crop_region, read_metadata
from .utils import load_config, select_device

Detection = tuple[Box, float]
_MANUAL_COLUMNS = ["stem", "postprocess", "x0", "y0", "x1", "y1"]
_SHARED_PP = ("original", "kakao", "resize")  # postprocess 없는 수동 박스를 쓸 수 있는 종류


class Detector(Protocol):
    name: str

    def detect(self, image: Image.Image) -> list[Detection]: ...


class HFZeroShotDetector:
    """Grounding DINO 계열 zero-shot 탐지기 (transformers). 박스는 0~1 정규화."""

    def __init__(self, model_id: str, queries: Sequence[str], box_threshold: float,
                 text_threshold: float, device: str = "auto"):
        import torch  # noqa: F401
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.name = model_id
        self.device = select_device(device)
        self.text = ". ".join(q.lower() for q in queries) + "."
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()

    def detect(self, image: Image.Image) -> list[Detection]:
        import torch

        img = ImageOps.exif_transpose(image).convert("RGB")
        w, h = img.size
        inputs = self.processor(images=img, text=self.text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.box_threshold,
            text_threshold=self.text_threshold, target_sizes=[(h, w)])[0]
        out: list[Detection] = []
        for box, score in zip(res["boxes"].tolist(), res["scores"].tolist()):
            x0, y0, x1, y1 = box
            norm = [min(1.0, max(0.0, v)) for v in (x0 / w, y0 / h, x1 / w, y1 / h)]
            try:
                out.append((validate_box(norm), float(score)))
            except ValueError:
                continue
        return out


def choose_box(detections: list[Detection], min_area_frac: float,
               max_area_frac: float) -> Detection | None:
    """면적이 범위 안인 탐지 중 점수가 가장 높은 것 (동점이면 앞의 것)."""
    best: Detection | None = None
    for box, score in detections:
        area = (box[2] - box[0]) * (box[3] - box[1])
        if min_area_frac <= area <= max_area_frac and (best is None or score > best[1]):
            best = (box, score)
    return best


def read_manual_boxes(path: str | Path) -> dict[tuple[str, str], Box]:
    """수동 박스 CSV를 {(stem, postprocess 또는 ""): box}로 읽는다. 파일이 없으면 {}."""
    p = Path(path)
    if not p.is_file():
        return {}
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    missing = [c for c in _MANUAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"수동 박스 CSV에 컬럼이 없습니다: {missing}")
    out: dict[tuple[str, str], Box] = {}
    for row in df.to_dict("records"):
        stem, pp = row["stem"].strip(), row["postprocess"].strip()
        if pp and pp not in POSTPROCESSES:
            raise ValueError(f"잘못된 postprocess 값: {pp!r} ({stem})")
        try:
            box = validate_box([row[c] for c in ("x0", "y0", "x1", "y1")])
        except ValueError as exc:
            raise ValueError(f"수동 박스 오류 ({stem}): {exc}") from exc
        if (stem, pp) in out:
            raise ValueError(f"수동 박스 키 중복: ({stem!r}, {pp!r})")
        out[(stem, pp)] = box
    return out


def manual_box_for(manual: dict[tuple[str, str], Box], image_path: str,
                   postprocess: str) -> Box | None:
    """(stem, postprocess)를 먼저, 없으면 original/kakao/resize에 한해 (stem, "")."""
    stem = Path(image_path).stem
    if (stem, postprocess) in manual:
        return manual[(stem, postprocess)]
    if postprocess in _SHARED_PP:
        return manual.get((stem, ""))
    return None


def detector_fingerprint(locate_cfg: dict) -> str:
    """탐지 설정(모델, 질의, 임계값, 면적 범위)의 지문. 캐시 재사용 판단에 쓴다."""
    keys = ("model_id", "queries", "box_threshold", "text_threshold",
            "min_area_frac", "max_area_frac")
    h = hashlib.sha1(json.dumps({k: locate_cfg[k] for k in keys}, sort_keys=True)
                     .encode("utf-8")).hexdigest()[:8]
    return f"{locate_cfg['model_id']}#{h}"


def locate_frame(df: pd.DataFrame, get_detector: Callable[[], Detector],
                 manual: dict[tuple[str, str], Box], cached: dict[str, BoxRecord],
                 manual_splits: Sequence[str], min_area_frac: float, max_area_frac: float,
                 fingerprint: str, overwrite: bool = False, save_every: int = 200,
                 on_checkpoint: Callable[[list[BoxRecord]], None] | None = None,
                 stats: dict | None = None) -> list[BoxRecord]:
    """행마다 박스 레코드 하나: 수동(허용 split) > 캐시 재사용(지문 일치) > 탐지 > fallback.

    새 탐지 save_every개마다(와 예외 시) on_checkpoint(지금까지의 레코드)를 부른다.
    stats가 있으면 reused/redetected(지문이 달라 다시 탐지한 수)를 채운다.
    """
    detector: Detector | None = None
    records: list[BoxRecord] = []
    reused = stale = new = flushed = 0
    try:
        for row in tqdm(df.to_dict("records"), desc="locate"):
            path, pp = row["image_path"], row["postprocess"]
            mbox = manual_box_for(manual, path, pp) if row["split"] in manual_splits else None
            if mbox is not None:
                records.append({"image_path": path, "box": mbox, "score": 1.0,
                                "source": "manual", "detector": "manual"})
                continue
            old = cached.get(path)
            if old is not None and old["source"] in ("auto", "fallback") and not overwrite:
                if old["detector"] == fingerprint:
                    records.append(old)
                    reused += 1
                    continue
                stale += 1
            if detector is None:
                detector = get_detector()
            with Image.open(row["_resolved_path"]) as im:
                chosen = choose_box(detector.detect(im), min_area_frac, max_area_frac)
            if chosen is not None:
                records.append({"image_path": path, "box": chosen[0], "score": chosen[1],
                                "source": "auto", "detector": fingerprint})
            else:
                records.append({"image_path": path, "box": FULL_BOX, "score": 0.0,
                                "source": "fallback", "detector": fingerprint})
            new += 1
            if on_checkpoint is not None and save_every > 0 and new % save_every == 0:
                on_checkpoint(list(records))
                flushed = new
    except BaseException:
        if on_checkpoint is not None and new > flushed:
            on_checkpoint(list(records))
        raise
    if stats is not None:
        stats.update(reused=reused, redetected=stale)
    return records


def _box_area(box: Box) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _crop_short_px(box: Box, size: tuple[int, int]) -> float:
    return min(size[0] * (box[2] - box[0]), size[1] * (box[3] - box[1]))


def locate_report(df: pd.DataFrame, boxes: dict[str, BoxRecord], threshold: float = 0.10,
                  rel_threshold: float = 0.25,
                  image_sizes: dict[str, tuple[int, int]] | None = None
                  ) -> tuple[pd.DataFrame, list[str]]:
    """(split, fake_type, postprocess)별 source 비율·박스 품질 표와 real 대비 차이 경고."""
    work = df[["image_path", "split", "fake_type", "postprocess"]].copy()
    work["source"] = work.image_path.map(lambda p: boxes[p]["source"])
    work["score"] = work.image_path.map(lambda p: boxes[p]["score"])
    work["area"] = work.image_path.map(lambda p: _box_area(boxes[p]["box"]))
    work["short_px"] = work.image_path.map(
        lambda p: _crop_short_px(boxes[p]["box"], image_sizes[p])
        if image_sizes is not None and p in image_sizes else float("nan"))
    rows = []
    for (split, ft, pp), g in work.groupby(["split", "fake_type", "postprocess"], sort=True):
        n = len(g)
        auto = g[g.source == "auto"]
        rows.append({"split": split, "fake_type": ft, "postprocess": pp, "n": n,
                     "auto_rate": (g.source == "auto").sum() / n,
                     "manual_rate": (g.source == "manual").sum() / n,
                     "fallback_rate": (g.source == "fallback").sum() / n,
                     "auto_mean_score": float(auto.score.mean()) if len(auto) else float("nan"),
                     "median_area_frac": float(g.area.median()),
                     "median_crop_short_px": float(g.short_px.median())})
    table = pd.DataFrame(rows, columns=["split", "fake_type", "postprocess", "n",
                                        "auto_rate", "manual_rate", "fallback_rate",
                                        "auto_mean_score", "median_area_frac",
                                        "median_crop_short_px"])
    warnings: list[str] = []
    for (split, pp), g in table.groupby(["split", "postprocess"], sort=True):
        real = g[g.fake_type == "real"]
        if real.empty:
            continue
        real = real.iloc[0]
        for _, r in g[g.fake_type != "real"].iterrows():
            for name in ("fallback", "manual"):
                a, b = r[f"{name}_rate"], real[f"{name}_rate"]
                if abs(a - b) > threshold:
                    warnings.append(
                        f"{split}/{pp}: {r.fake_type}의 {name} 비율 {a:.2f}이(가) "
                        f"real {b:.2f}과 {threshold:.2f} 넘게 다릅니다")
            a, b = r.auto_mean_score, real.auto_mean_score
            if not (np.isnan(a) or np.isnan(b)) and abs(a - b) > threshold:
                warnings.append(
                    f"{split}/{pp}: {r.fake_type}의 auto_mean_score {a:.3f}이(가) "
                    f"real {b:.3f}과 {threshold:.2f} 넘게 다릅니다")
            for name in ("median_area_frac", "median_crop_short_px"):
                a, b = r[name], real[name]
                if np.isnan(a) or np.isnan(b) or b == 0:
                    continue
                if abs(a - b) / b > rel_threshold:
                    warnings.append(
                        f"{split}/{pp}: {r.fake_type}의 {name} {a:.4g}이(가) "
                        f"real {b:.4g}과 {rel_threshold:.0%} 넘게 다릅니다")
    return table, warnings


def _image_sizes(df: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """헤더만 읽어 EXIF 회전을 반영한 (W, H)를 돌려준다."""
    out = {}
    for path, resolved in zip(df.image_path, df._resolved_path):
        with Image.open(resolved) as im:
            w, h = im.size
            if im.getexif().get(0x0112) in (5, 6, 7, 8):
                w, h = h, w
        out[path] = (w, h)
    return out


def dump_crop_samples(df: pd.DataFrame, boxes: dict[str, BoxRecord], out_dir: Path,
                      margin: float, per_group: int) -> int:
    """그룹(split, fake_type, postprocess)마다 앞 N개의 평가용 크롭을 저장한다. 저장 개수를 돌려준다."""
    saved = 0
    for (split, ft, pp), g in df.groupby(["split", "fake_type", "postprocess"], sort=True):
        folder = out_dir / f"{split}_{ft}_{pp}"
        folder.mkdir(parents=True, exist_ok=True)
        for row in g.head(per_group).to_dict("records"):
            with Image.open(row["_resolved_path"]) as im:
                crop = crop_region(im, boxes[row["image_path"]]["box"], margin)
            scale = 512 / max(crop.size)
            if scale < 1:
                crop = crop.resize((max(1, round(crop.width * scale)),
                                    max(1, round(crop.height * scale))), Image.LANCZOS)
            crop.save(folder / f"{Path(row['image_path']).stem}.jpg", quality=90)
            saved += 1
    return saved


def _warn_unmatched_manual(manual: dict[tuple[str, str], Box], df: pd.DataFrame) -> None:
    """CSV의 어떤 행과도 맞지 않는 수동 박스 키가 있으면 경고한다."""
    seen = {(Path(p).stem, pp) for p, pp in zip(df.image_path, df.postprocess)}
    shared = {stem for stem, pp in seen if pp in _SHARED_PP}
    unmatched = [k for k in manual if not (k[0] in shared if k[1] == "" else k in seen)]
    if unmatched:
        print(f"경고: 수동 박스 {len(unmatched)}개가 CSV의 어떤 이미지와도 맞지 않습니다 "
              f"(예: {', '.join(k[0] for k in unmatched[:3])})")


def run(config_path, metadata: str | None = None, overwrite: bool = False,
        device: str | None = None, detector: Detector | None = None,
        dump_crops: int = 0) -> pd.DataFrame:
    """전체 이미지의 메모 박스를 찾아 캐시 CSV와 리포트를 쓰고 리포트 표를 돌려준다."""
    cfg = load_config(config_path)
    data, loc = cfg["data"], cfg["locate"]
    meta_path = Path(metadata or data["metadata"])
    df = read_metadata(meta_path, data["root"],
                       tuple(data.get("leakage_columns", ("item_id",))),
                       verify_images=data.get("verify_images", True))
    boxes_csv = Path(loc["boxes_csv"])
    cache = read_boxes(boxes_csv)
    manual = read_manual_boxes(loc["manual_csv"])
    _warn_unmatched_manual(manual, df)

    def get_detector() -> Detector:
        if detector is not None:
            return detector
        return HFZeroShotDetector(loc["model_id"], loc["queries"], loc["box_threshold"],
                                  loc["text_threshold"], device or cfg.get("training", {}).get("device", "auto"))

    merged = dict(cache)

    def checkpoint(partial: list[BoxRecord]) -> None:
        merged.update({r["image_path"]: r for r in partial})
        write_boxes(merged.values(), boxes_csv)

    stats: dict = {}
    try:
        records = locate_frame(df, get_detector, manual, cache, loc.get("manual_splits", ["train"]),
                               loc["min_area_frac"], loc["max_area_frac"],
                               detector_fingerprint(loc), overwrite=overwrite,
                               on_checkpoint=checkpoint, stats=stats)
    finally:
        write_boxes(merged.values(), boxes_csv)  # 예외·Ctrl-C여도 지금까지의 결과를 남긴다
    checkpoint(records)
    print(f"캐시 재사용 {stats['reused']}개, 설정이 달라 다시 탐지 {stats['redetected']}개")

    current = {r["image_path"]: r for r in records}
    table, warnings = locate_report(
        df, current, threshold=loc.get("warn_rate_diff", 0.10),
        rel_threshold=loc.get("warn_rel_diff", 0.25), image_sizes=_image_sizes(df))
    table.to_csv(boxes_csv.parent / f"locate_report_{meta_path.stem}.csv", index=False,
                 encoding="utf-8")
    for w in warnings:
        print(f"경고: {w}")
    counts = pd.Series([r["source"] for r in records]).value_counts()
    print("source 집계: " + ", ".join(f"{s}={int(counts.get(s, 0))}" for s in ("auto", "manual", "fallback")))
    if dump_crops > 0:
        n = dump_crop_samples(df, current, boxes_csv.parent / "crop_samples",
                              cfg["crop"]["margin"], dump_crops)
        print(f"크롭 샘플 {n}장 저장: {boxes_csv.parent / 'crop_samples'}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="모델 2 메모 위치 찾기")
    parser.add_argument("--config", default="configs/model2_memo.yaml")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)
    parser.add_argument("--dump-crops", type=int, default=0,
                        help="그룹(split, fake_type, postprocess)마다 저장할 크롭 샘플 수")
    args = parser.parse_args()
    try:
        run(args.config, metadata=args.metadata, overwrite=args.overwrite, device=args.device,
            dump_crops=args.dump_crops)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
