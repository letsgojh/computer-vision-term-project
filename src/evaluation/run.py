"""평가 진입점. 예측 CSV와 split을 묶어 outputs/results 에 결과표를 쓴다."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.common.paths import FAKE_MAPPING_CSV, PREDICTIONS, PROCESSED, REAL_METADATA_CSV, RESULTS, SPLIT_CSV
from src.common.schema import SPLIT_COLUMNS
from src.evaluation.analyze import (
    diagnose,
    error_frames,
    feedback_markdown,
    macro_recall,
    metadata_slices,
    paired_comparison,
    representatives,
    robustness,
)
from src.evaluation.demo import build_demo
from src.evaluation.load import (
    attach_metadata,
    attach_split,
    collect_split_warnings,
    coverage_table,
    discover_predictions,
    load_prediction_files,
    load_table,
)
from src.evaluation.metrics import choose_threshold, metric_rows
from src.evaluation.render import (
    error_csv,
    export_cases,
    gallery_markdown,
    save_figures,
    write_report,
    write_support,
    write_tables,
)
from src.evaluation.vocab import MODEL_KO


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


def _prepare(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    for name in ("tables", "errors", "figures", "cases"):
        path = out_dir / name
        if path.exists():
            shutil.rmtree(path)


def _threshold_map(frame: pd.DataFrame, val_frame: pd.DataFrame) -> dict[tuple[str, str], float]:
    found = {}
    if frame.empty:
        return found
    for model, run in frame[["model", "run"]].drop_duplicates().itertuples(index=False):
        part = val_frame[(val_frame["model"] == model) & (val_frame["run"] == run)] if not val_frame.empty else part_empty()
        found[(model, run)] = choose_threshold(part["label"], part["prob_fake"]) if len(part) else None
    return found


def part_empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["model", "run", "label", "prob_fake"])


def run_evaluation(
    prediction_paths: list[Path],
    split_csv: Path,
    real_metadata_csv: Path | None,
    fake_mapping_csv: Path | None,
    processed_root: Path | None,
    out_dir: Path,
    eval_split: str = "test",
    threshold: float = 0.5,
    n_boot: int = 0,
    cases_per_cell: int = 1,
    demo: bool = False,
) -> dict:
    """예측을 읽어 결과표·오류 사례·약한 유형 피드백을 out_dir에 쓴다."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold는 0과 1 사이여야 합니다.")
    notes: list[str] = []
    pred, load_notes = load_prediction_files(prediction_paths)
    notes.extend(load_notes)
    split = load_table(split_csv, ["filename", "label", "fake_type", "split"], "split.csv")
    if split.empty:
        raise ValueError("split.csv에 평가할 행이 없습니다.")
    for col in SPLIT_COLUMNS:
        if col not in split.columns:
            split[col] = pd.NA

    real_meta = load_table(
        real_metadata_csv,
        ["filename"],
        "real_metadata.csv",
    )
    mapping = load_table(fake_mapping_csv, ["fake_filename"], "fake_mapping.csv")
    if real_meta.empty:
        notes.append("real_metadata.csv가 비어 있어 촬영 조건(펜·조명·폰)을 붙이지 못했습니다.")
    if mapping.empty:
        notes.append("fake_mapping.csv가 비어 있어 A·B 오류를 원본 사진 조건에 연결하지 못했습니다.")

    selected, split_table, join_notes = attach_split(pred, split, eval_split)
    notes.extend(join_notes)
    selected = attach_metadata(selected, real_meta, mapping)
    notes.extend(collect_split_warnings(split_table, selected, eval_split))

    val_selected, _, _ = attach_split(pred, split, "val")
    val_thresholds = _threshold_map(selected, val_selected)
    val_labels = {}
    for (model, run), value in val_thresholds.items():
        if value is None:
            continue
        label = MODEL_KO.get(model, model)
        if run not in {"", "default"}:
            same = selected[(selected["model"] == model)]["run"].nunique()
            if same > 1:
                label = f"{label} ({run})"
        val_labels[label] = value

    def fixed(_model, _run):
        return threshold

    metrics = metric_rows(selected, fixed, n_boot=n_boot)

    def from_val(model, run):
        value = val_thresholds.get((model, run))
        return threshold if value is None else value

    metrics_val = metric_rows(selected, from_val, n_boot=0) if any(v is not None for v in val_thresholds.values()) else pd.DataFrame()

    coverage = coverage_table(split_table, selected, eval_split)
    low = coverage[coverage["coverage"].notna() & (coverage["coverage"] < 0.95)] if not coverage.empty else coverage
    if not low.empty:
        notes.append("일부 후처리의 예측 커버리지가 95% 미만입니다. 그 칸은 test 전체 점수가 아닙니다.")

    fn, fp = error_frames(selected, fixed)
    slices = metadata_slices(selected, fixed)
    reps = representatives(fn, fp, cases_per_cell)
    paired = paired_comparison(selected, fixed)
    robust = robustness(metrics)
    macros = macro_recall(metrics)
    diagnoses = diagnose(metrics, slices, paired, robust)

    _prepare(out_dir)
    tables = out_dir / "tables"
    errors = out_dir / "errors"
    write_tables(metrics, tables)
    write_support(metrics, tables)
    if not metrics_val.empty:
        from src.evaluation.render import grid, md_table
        from src.evaluation.vocab import NOTION_SCOPES

        headers, rows = grid(metrics_val, NOTION_SCOPES, "f1")
        note = "기준값은 모델마다 val 전체에서 F1이 최대인 값입니다. 표의 숫자는 그 기준값으로 잰 test F1입니다.\n\n"
        (tables / "f1_val_threshold.md").write_text(note + md_table(headers, rows) + "\n", encoding="utf-8")
        pd.DataFrame(rows, columns=headers).to_csv(tables / "f1_val_threshold.csv", index=False, encoding="utf-8-sig")
    macros.to_csv(tables / "macro_recall.csv", index=False, encoding="utf-8-sig")
    robust.to_csv(out_dir / "robustness.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(out_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(out_dir / "coverage.csv", index=False, encoding="utf-8-sig")

    errors.mkdir(parents=True, exist_ok=True)
    error_csv(fn, errors / "fn.csv")
    error_csv(fp, errors / "fp.csv")
    slices.to_csv(errors / "metadata_slices.csv", index=False, encoding="utf-8-sig")
    cases = export_cases(reps, processed_root, out_dir)
    cases.to_csv(errors / "representative.csv", index=False, encoding="utf-8-sig")
    (errors / "gallery.md").write_text(gallery_markdown(cases), encoding="utf-8")

    feedback = feedback_markdown(metrics, slices, robust, eval_split, threshold, demo)
    (out_dir / "feedback_weak_types.md").write_text(feedback, encoding="utf-8")
    save_figures(selected, metrics, out_dir / "figures")
    write_report(
        out_dir / "report.md",
        metrics=metrics,
        notes=notes,
        diagnoses=diagnoses,
        coverage=coverage,
        val_thresholds=val_labels,
        eval_split=eval_split,
        threshold=threshold,
        demo=demo,
        paired=paired,
    )
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_split": eval_split,
        "threshold": threshold,
        "positive_class": "fake",
        "demo": demo,
        "prediction_files": [path.name for path in prediction_paths],
        "n_rows": int(len(selected)),
        "val_thresholds": {f"{model}:{run}": value for (model, run), value in val_thresholds.items()},
        "notes": notes,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"out_dir": out_dir, "metrics": metrics, "notes": notes, "feedback": feedback}


def _print_summary(result: dict) -> None:
    out = result["out_dir"]
    print(f"결과를 썼습니다: {out}")
    print(f"- 보고서: {out / 'report.md'}")
    print(f"- 결과표: {out / 'tables' / 'bundle.md'}")
    print(f"- 오류 사례: {out / 'errors' / 'gallery.md'}")
    print(f"- 생성 피드백: {out / 'feedback_weak_types.md'}")


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser(description="인증샷 진위 판별 평가 (가짜 = 양성)")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS, help="예측 CSV 파일 또는 폴더")
    parser.add_argument("--split", type=Path, default=SPLIT_CSV)
    parser.add_argument("--real-metadata", type=Path, default=REAL_METADATA_CSV)
    parser.add_argument("--fake-mapping", type=Path, default=FAKE_MAPPING_CSV)
    parser.add_argument("--processed", type=Path, default=PROCESSED, help="대표 사례 이미지를 찾을 data/processed")
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--eval-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=200, help="F1·AUC 구간 반복 횟수. 0이면 생략")
    parser.add_argument("--cases", type=int, default=1, help="칸마다 고를 대표 오류 수")
    parser.add_argument("--demo", action="store_true", help="형식 확인용 합성 데이터로 결과표를 만듭니다")
    args = parser.parse_args(argv)

    out = args.out
    predictions = args.predictions
    split_csv = args.split
    real_meta = args.real_metadata
    mapping = args.fake_mapping
    processed = args.processed
    demo = args.demo
    if demo:
        if args.out == RESULTS:
            out = RESULTS / "format_preview"
        demo_root = out / "_demo_inputs"
        built = build_demo(demo_root)
        predictions = built["predictions"]
        split_csv = built["split"]
        real_meta = built["real_metadata"]
        mapping = built["fake_mapping"]
        processed = None

    files = discover_predictions(predictions)
    if not files:
        print(
            "예측 CSV가 없습니다. 모델 담당 결과를 outputs/predictions/{model}_{run}.csv 에 두면 됩니다.\n"
            "열은 filename, postprocess, prob_fake 입니다.\n"
            "결과표 형식을 먼저 보려면: python -m src.evaluation.run --demo",
            file=sys.stderr,
        )
        return 1
    try:
        result = run_evaluation(
            files,
            split_csv,
            real_meta,
            mapping,
            processed,
            out,
            eval_split=args.eval_split,
            threshold=args.threshold,
            n_boot=args.bootstrap,
            cases_per_cell=max(1, args.cases),
            demo=demo,
        )
    except ValueError as exc:
        print(f"평가를 진행하지 못했습니다: {exc}", file=sys.stderr)
        return 1
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
