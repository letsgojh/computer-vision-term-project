"""결과표, 보고서, 그림, 대표 사례를 outputs/results 에 쓴다."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.common.schema import POSTPROCESSES
from src.evaluation.vocab import (
    ALL_SCOPES,
    METRIC_HELP,
    NOTION_SCOPES,
    POSTPROCESS_COLUMNS,
    POSTPROCESS_KO,
    SCOPE_KO,
    SCOPE_NOTE,
)

OFFICIAL = (
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("auc", "AUC"),
)
EXTRA = (
    ("specificity", "Specificity"),
    ("balanced_accuracy", "Balanced Accuracy"),
    ("average_precision", "AP"),
    ("brier", "Brier"),
    ("ece", "ECE"),
    ("prob_gap", "확률 간격"),
)


def fmt(value, digits=3) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except TypeError:
        pass
    return f"{float(value):.{digits}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell).replace("|", "/") for cell in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def identities(metrics: pd.DataFrame):
    if metrics.empty:
        return []
    preferred = {"model1": 0, "model2": 1}
    table = metrics[["model", "run", "model_label"]].drop_duplicates()

    def key(row):
        return (preferred.get(row.model, 9), str(row.model), str(row.run))

    return sorted(table.itertuples(index=False), key=key)


def _cell(metrics: pd.DataFrame, model, run, scope, postprocess, column: str) -> str:
    hit = metrics[
        (metrics["model"] == model)
        & (metrics["run"] == run)
        & (metrics["fake_type"] == scope)
        & (metrics["postprocess"] == postprocess)
    ]
    if hit.empty or int(hit.iloc[0]["n"]) == 0:
        return "—"
    return fmt(hit.iloc[0][column])


def _bundle(metrics: pd.DataFrame, model, run, scope, postprocess) -> str:
    hit = metrics[
        (metrics["model"] == model)
        & (metrics["run"] == run)
        & (metrics["fake_type"] == scope)
        & (metrics["postprocess"] == postprocess)
    ]
    if hit.empty or int(hit.iloc[0]["n"]) == 0:
        return "—"
    row = hit.iloc[0]
    return (
        f"Acc {fmt(row.accuracy)} · P {fmt(row.precision)} · R {fmt(row.recall)} · "
        f"F1 {fmt(row.f1)} · AUC {fmt(row.auc)} · n {int(row.n_pos)}/{int(row.n_neg)}"
    )


def grid(metrics: pd.DataFrame, scopes, column: str | None) -> tuple[list[str], list[list[str]]]:
    headers = ["모델 / 유형", *POSTPROCESS_COLUMNS]
    rows = []
    for ident in identities(metrics):
        for scope in scopes:
            label = f"{ident.model_label} – {SCOPE_KO[scope]}"
            cells = []
            for postprocess in POSTPROCESSES:
                if column is None:
                    cells.append(_bundle(metrics, ident.model, ident.run, scope, postprocess))
                else:
                    cells.append(_cell(metrics, ident.model, ident.run, scope, postprocess, column))
            rows.append([label, *cells])
    return headers, rows


def write_tables(metrics: pd.DataFrame, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(folder / "metrics_long.csv", index=False, encoding="utf-8-sig")
    specs = [(None, "bundle", NOTION_SCOPES), ("f1", "f1", NOTION_SCOPES)]
    specs += [(key, key, NOTION_SCOPES) for key, _ in OFFICIAL]
    specs += [(key, key, NOTION_SCOPES) for key, _ in EXTRA]
    specs += [(None, "bundle_with_overall", ALL_SCOPES), ("f1", "f1_with_overall", ALL_SCOPES)]
    for column, name, scopes in specs:
        headers, rows = grid(metrics, scopes, column)
        (folder / f"{name}.md").write_text(md_table(headers, rows) + "\n", encoding="utf-8")
        pd.DataFrame(rows, columns=headers).to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig")


def write_support(metrics: pd.DataFrame, folder: Path) -> None:
    if metrics.empty:
        return
    rows = []
    for ident in identities(metrics):
        for scope in ALL_SCOPES:
            cells = [f"{ident.model_label} – {SCOPE_KO[scope]}"]
            for postprocess in POSTPROCESSES:
                hit = metrics[
                    (metrics["model"] == ident.model)
                    & (metrics["run"] == ident.run)
                    & (metrics["fake_type"] == scope)
                    & (metrics["postprocess"] == postprocess)
                ]
                if hit.empty or int(hit.iloc[0]["n"]) == 0:
                    cells.append("—")
                else:
                    row = hit.iloc[0]
                    cells.append(f"{int(row.n_pos)}/{int(row.n_neg)}")
            rows.append(cells)
    headers = ["모델 / 유형", *POSTPROCESS_COLUMNS]
    text = "n은 가짜/진짜 장수입니다. 칸의 장수가 작으면 그 숫자의 순위를 단정하지 않습니다.\n\n"
    (folder / "support.md").write_text(text + md_table(headers, rows) + "\n", encoding="utf-8")


def error_csv(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        "model_label", "model", "run", "error", "filename", "postprocess", "fake_type",
        "object", "photographer", "prob_fake", "threshold", "pen", "paper", "lighting",
        "background", "phone_model", "source_filename", "memo_text", "image_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8-sig")
        return
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    out[columns].sort_values(["model", "postprocess", "prob_fake"]).to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def export_cases(reps: pd.DataFrame, processed_root: Path | None, out_dir: Path) -> pd.DataFrame:
    """대표 사례의 원본 경로를 적고, 파일이 있으면 cases/로 복사한다."""
    rows = []
    if reps.empty:
        return pd.DataFrame(rows)
    for record in reps.itertuples(index=False):
        filename = str(record.filename)
        postprocess = str(record.postprocess)
        expected = f"data/processed/{postprocess}/{filename}"
        src = None
        image_path = getattr(record, "image_path", "")
        if isinstance(image_path, str) and image_path and Path(image_path).is_file():
            src = Path(image_path)
        elif processed_root is not None:
            candidate = processed_root / postprocess / filename
            if candidate.is_file():
                src = candidate
        copied = ""
        if src is not None:
            dest_dir = out_dir / "cases" / str(record.error).lower() / str(record.model) / postprocess
            dest_dir.mkdir(parents=True, exist_ok=True)
            target = dest_dir / f"{record.fake_type}_{filename}"
            shutil.copy2(src, target)
            copied = target.relative_to(out_dir).as_posix()
        rows.append(
            {
                "error": record.error,
                "model_label": record.model_label,
                "fake_type": record.fake_type,
                "postprocess": postprocess,
                "postprocess_ko": POSTPROCESS_KO.get(postprocess, postprocess),
                "filename": filename,
                "prob_fake": record.prob_fake,
                "threshold": record.threshold,
                "object": getattr(record, "object", ""),
                "photographer": getattr(record, "photographer", ""),
                "pen": getattr(record, "pen", ""),
                "paper": getattr(record, "paper", ""),
                "lighting": getattr(record, "lighting", ""),
                "background": getattr(record, "background", ""),
                "phone_model": getattr(record, "phone_model", ""),
                "source_filename": getattr(record, "source_filename", ""),
                "memo_text": getattr(record, "memo_text", ""),
                "expected_path": expected,
                "copied_path": copied,
            }
        )
    return pd.DataFrame(rows)


def gallery_markdown(cases: pd.DataFrame) -> str:
    lines = [
        "# 대표 오류 사례",
        "",
        "발표에 쓸 사례입니다. FN은 그 칸에서 가짜 확률이 가장 낮은 것, FP는 가장 높은 것입니다.",
        "이미지 파일이 `data/processed/{후처리}/`에 있으면 `cases/`에 복사합니다.",
        "",
    ]
    if cases.empty:
        lines.append("대표 사례가 없습니다.")
        return "\n".join(lines) + "\n"
    ordered = cases.copy()
    ordered["_rank"] = np.where(ordered["error"] == "FN", ordered["prob_fake"], -ordered["prob_fake"])
    ordered = ordered.sort_values(["error", "_rank"])
    for row in ordered.itertuples(index=False):
        kind = "놓친 가짜 (FN)" if row.error == "FN" else "오탐한 진짜 (FP)"
        lines.append(f"## {kind} · {row.model_label} · {row.fake_type} · {row.postprocess_ko}")
        lines.append("")
        lines.append(f"- 파일: `{row.filename}`")
        lines.append(f"- 가짜 확률: {fmt(row.prob_fake)} (기준 {fmt(row.threshold, 2)})")
        bits = []
        for label, value in (
            ("물건", row.object),
            ("촬영자", row.photographer),
            ("펜", row.pen),
            ("종이", row.paper),
            ("조명", row.lighting),
            ("배경", row.background),
            ("폰", row.phone_model),
        ):
            text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
            if text and text.lower() != "nan":
                bits.append(f"{label} {text}")
        if bits:
            lines.append("- 조건: " + ", ".join(bits))
        source = "" if row.source_filename is None else str(row.source_filename)
        if source and source.lower() != "nan" and source != row.filename:
            lines.append(f"- 원본 진짜: `{source}`")
        memo = "" if row.memo_text is None or (isinstance(row.memo_text, float) and pd.isna(row.memo_text)) else str(row.memo_text)
        if memo and memo.lower() != "nan":
            lines.append(f"- 메모: {memo}")
        lines.append(f"- 이미지 위치: `{row.expected_path}`")
        if row.copied_path:
            lines.append(f"- 복사본: `{row.copied_path}`")
        lines.append("")
    return "\n".join(lines)


def _style():
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "맑은 고딕", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 11,
        }
    )


def save_figures(frame: pd.DataFrame, metrics: pd.DataFrame, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    _style()
    _plot_f1(metrics, folder / "f1_by_type.png")
    _plot_roc(frame, folder / "roc_original.png")
    _plot_hist(frame, folder / "score_hist_original.png")
    _plot_drop(metrics, folder / "recall_vs_original.png")


def _plot_f1(metrics: pd.DataFrame, path: Path) -> None:
    if metrics.empty:
        return
    people = identities(metrics)
    fig, axes = plt.subplots(1, len(people), figsize=(5.2 * len(people), 4.2), squeeze=False)
    colors = {"A": "#d55e00", "B": "#0072b2", "C": "#009e73"}
    x = np.arange(len(POSTPROCESSES))
    width = 0.24
    for ax, ident in zip(axes[0], people):
        for i, scope in enumerate(NOTION_SCOPES):
            vals = []
            for postprocess in POSTPROCESSES:
                text = _cell(metrics, ident.model, ident.run, scope, postprocess, "f1")
                vals.append(np.nan if text == "—" else float(text))
            ax.bar(x + (i - 1) * width, vals, width=width, label=f"type {scope}", color=colors[scope])
        ax.set_xticks(x, [POSTPROCESS_KO[p] for p in POSTPROCESSES])
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1")
        ax.set_title(ident.model_label)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_roc(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    part = frame[frame["postprocess"] == "original"]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    colors = {"model1": "#0072b2", "model2": "#d55e00"}
    drawn = False
    for ident in identities(frame.rename(columns={"model_label": "model_label"})):
        group = part[(part["model"] == ident.model) & (part["run"] == ident.run)]
        y = group["label"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, group["prob_fake"])
        ax.plot(fpr, tpr, color=colors.get(ident.model, "#444444"), label=ident.model_label)
        drawn = True
    if not drawn:
        plt.close(fig)
        return
    ax.plot([0, 1], [0, 1], color="#bbbbbb", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC · original · all fakes")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_hist(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    people = identities(frame)
    part = frame[frame["postprocess"] == "original"]
    fig, axes = plt.subplots(1, len(people), figsize=(5.0 * len(people), 3.8), squeeze=False)
    for ax, ident in zip(axes[0], people):
        group = part[(part["model"] == ident.model) & (part["run"] == ident.run)]
        ax.hist(group.loc[group["label"] == 0, "prob_fake"], bins=15, range=(0, 1), alpha=0.65, label="real", color="#0072b2")
        ax.hist(group.loc[group["label"] == 1, "prob_fake"], bins=15, range=(0, 1), alpha=0.65, label="fake", color="#d55e00")
        ax.axvline(0.5, color="#333333", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("prob_fake")
        ax.set_title(ident.model_label)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_drop(metrics: pd.DataFrame, path: Path) -> None:
    if metrics.empty:
        return
    people = identities(metrics)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(people))
    width = 0.22
    offsets = {"kakao": -width, "resize": 0, "recapture": width}
    colors = {"kakao": "#e69f00", "resize": "#0072b2", "recapture": "#cc79a7"}
    for postprocess, offset in offsets.items():
        vals = []
        for ident in people:
            base = _cell(metrics, ident.model, ident.run, "all", "original", "recall")
            cur = _cell(metrics, ident.model, ident.run, "all", postprocess, "recall")
            if base == "—" or cur == "—":
                vals.append(np.nan)
            else:
                vals.append(float(cur) - float(base))
        ax.bar(x + offset, vals, width=width, label=POSTPROCESS_KO[postprocess], color=colors[postprocess])
    ax.axhline(0, color="#888888", linewidth=1)
    ax.set_xticks(x, [ident.model_label for ident in people])
    ax.set_ylabel("Recall change from original")
    ax.set_title("Robustness · all fakes")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(
    path: Path,
    *,
    metrics: pd.DataFrame,
    notes: list[str],
    diagnoses: list[str],
    coverage: pd.DataFrame,
    val_thresholds: dict,
    eval_split: str,
    threshold: float,
    demo: bool,
    paired: pd.DataFrame,
) -> None:
    lines = ["# 평가 · 분석 결과", ""]
    if demo:
        lines.append("> **형식 확인용 합성 데이터입니다.** 모델 성능이 아닙니다. 실제 예측 CSV로 다시 실행하면 같은 형식으로 덮어씁니다.")
        lines.append("")
    lines.append(f"양성은 가짜(label 1)입니다. `{eval_split}` split만 집계했습니다. 판정은 `prob_fake >= {threshold:.2f}` 입니다.")
    lines.append(SCOPE_NOTE)
    lines.append("묶음 칸의 `n`은 가짜 장수/진짜 장수입니다.")
    lines.append("")
    lines.append("## 1. 결과표")
    lines.append("")
    lines.append("노션 양식과 같은 행입니다. 칸 안에 Accuracy, Precision, Recall, F1, AUC를 함께 적었습니다.")
    lines.append("")
    headers, rows = grid(metrics, NOTION_SCOPES, None)
    lines.append(md_table(headers, rows))
    lines.append("")
    lines.append("전체(모든 가짜)와 유형을 같이 보려면 `tables/bundle_with_overall.md`를 엽니다.")
    lines.append("지표를 하나씩 붙여 넣으려면 `tables/accuracy.md`, `precision.md`, `recall.md`, `f1.md`, `auc.md`를 씁니다.")
    lines.append("")
    lines.append("## 2. 지표가 말하는 것")
    lines.append("")
    lines.append(md_table(["지표", "의미"], [[name, meaning] for name, meaning in METRIC_HELP]))
    lines.append("")
    lines.append("Accuracy만 높고 Recall이 낮으면 가짜를 놓치고 있는 것입니다. AUC는 높은데 F1이 낮으면 점수 자체는 나누지만 기준값 0.5가 맞지 않는 것입니다. AUC까지 낮으면 그 유형을 구분하지 못하는 것입니다.")
    lines.append("")
    for sentence in diagnoses:
        lines.append(f"- {sentence}")
    lines.append("")
    lines.append("## 3. 추가 지표")
    lines.append("")
    lines.append("요청된 다섯 지표 옆에, 모델이 실제로 갈라져 있는지 확인하는 값을 같이 둡니다.")
    lines.append("")
    lines.append("- Specificity: 진짜를 진짜로 두는 비율. Precision과 별개로 오탐 규모를 봅니다.")
    lines.append("- Balanced Accuracy: 진짜와 가짜 장수가 다를 때 Accuracy가 많은 쪽을 따라가는지 확인합니다.")
    lines.append("- AP: 가짜가 적어도 높은 점수 순으로 가짜가 먼저 오는지 봅니다. ROC-AUC와 같이 기준값에 기대지 않습니다.")
    lines.append("- Brier, ECE: 0.9가 정말 90% 가짜인지. 확률을 발표에서 확신도처럼 말할 수 있는지 봅니다.")
    lines.append("- 확률 간격: 가짜 점수 평균과 진짜 점수 평균이 벌어져 있는지. 히스토그램(`figures/score_hist_original.png`)과 함께 봅니다.")
    lines.append("- 유형 평균 Recall: A·B·C를 같은 무게로 평균합니다. 전체 Recall과 다르면 장수가 많은 유형이 점수를 끌고 있습니다. `tables/macro_recall.csv`")
    lines.append("- F1·AUC의 구간은 표본을 다시 뽑아 잡은 95% 구간입니다. `tables/metrics_long.csv`의 `f1_ci_low`, `f1_ci_high`. 장수가 적으면 비워 둡니다.")
    lines.append("")
    if val_thresholds:
        shown = ", ".join(f"{label} {value:.3f}" for label, value in val_thresholds.items())
        lines.append(f"val에서 F1이 최대인 기준값: {shown}. 이 값으로 test를 다시 잰 표는 `tables/f1_val_threshold.md`입니다. 기준값을 고를 때 test는 보지 않았습니다.")
    else:
        lines.append("val 예측이 없어, 기준값을 val에서 다시 고른 표는 만들지 않았습니다. 본 결과표의 기준값은 0.5입니다.")
    lines.append("")
    lines.append("## 4. 후처리")
    lines.append("")
    lines.append("원본에서 잘 맞히고 카톡 압축·리사이즈·재촬영에서만 떨어지면, 모델이 그 열화에 약한 것입니다. 차이는 `robustness.csv`, 그림은 `figures/recall_vs_original.png`입니다.")
    lines.append("")
    lines.append("## 5. 모델 1과 모델 2")
    lines.append("")
    if paired.empty:
        lines.append("두 모델의 예측이 모두 있을 때, 같은 파일을 누가 맞혔는지 `comparison.csv`에 적습니다. 유형 A(메모 조작)를 우선 비교합니다.")
    else:
        lines.append("같은 파일에 대해 누구만 맞혔는지는 `comparison.csv`입니다. `only_model2`가 유형 A에서 크면 메모 영역을 보는 쪽이 그 칸을 더 맞힌 것입니다.")
        focus = paired[(paired["fake_type"] == "A") & (paired["postprocess"] == "original")]
        if not focus.empty:
            lines.append("")
            lines.append(md_table(
                ["후처리", "장수", "둘 다 맞음", "모델 1만", "모델 2만", "둘 다 틀림"],
                [
                    [
                        POSTPROCESS_KO.get(row.postprocess, row.postprocess),
                        str(int(row.n)),
                        str(int(row.both_correct)),
                        str(int(row.only_model1)),
                        str(int(row.only_model2)),
                        str(int(row.both_wrong)),
                    ]
                    for row in focus.itertuples(index=False)
                ],
            ))
    lines.append("")
    lines.append("## 6. 오류 분석")
    lines.append("")
    lines.append("- 놓친 가짜 전체: `errors/fn.csv`")
    lines.append("- 오탐한 진짜 전체: `errors/fp.csv`")
    lines.append("- 촬영 조건별 집중: `errors/metadata_slices.csv` (`concentrated`가 참인 행)")
    lines.append("- 발표용 대표 사례: `errors/gallery.md`, 이미지가 있으면 `cases/`")
    lines.append("")
    lines.append("FN은 가짜인데 기준값보다 낮은 점수입니다. FP는 진짜인데 기준값 이상인 점수입니다. A·B의 촬영 조건은 원본 진짜 메타데이터와 연결합니다.")
    lines.append("")
    lines.append("## 7. 생성 담당 피드백")
    lines.append("")
    lines.append("`feedback_weak_types.md`에 우선 보강 칸과 요청 형식을 적어 두었습니다. 보강분은 train에만 넣습니다.")
    lines.append("")
    lines.append("## 8. 데이터 점검")
    lines.append("")
    if coverage.empty:
        lines.append("커버리지를 계산하지 못했습니다.")
    else:
        lines.append("| 모델 | 후처리 | 평가 파일 수 | 예측이 있는 파일 | 비율 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in coverage.itertuples(index=False):
            lines.append(
                f"| {row.model_label} | {POSTPROCESS_KO.get(row.postprocess, row.postprocess)} | "
                f"{int(row.n_expected)} | {int(row.n_found)} | {fmt(row.coverage)} |"
            )
        lines.append("")
        lines.append("비율이 1보다 작으면 그 칸의 점수는 test 전체가 아닙니다.")
    lines.append("")
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- 입력 형식에서 고칠 항목은 없었습니다.")
    lines.append("")
    lines.append("## 9. 파일")
    lines.append("")
    lines.append("- `report.md` 이 문서")
    lines.append("- `tables/` 결과표")
    lines.append("- `errors/` 오류 목록과 대표 사례")
    lines.append("- `figures/` F1, ROC, 점수 분포, 후처리 하락")
    lines.append("- `feedback_weak_types.md`")
    lines.append("- `robustness.csv`, `comparison.csv`, `run_meta.json`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
