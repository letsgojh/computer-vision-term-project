"""오류가 어느 유형·촬영 조건에 몰리는지, 생성 담당에게 무엇을 요청할지 정리한다."""

from __future__ import annotations

import pandas as pd

from src.common.schema import POSTPROCESSES
from src.evaluation.vocab import (
    META_FIELDS,
    META_KO,
    NOTION_SCOPES,
    POSTPROCESS_HINT,
    POSTPROCESS_KO,
    TYPE_HINT,
)

MIN_SLICE_N = 8
MIN_LIFT = 1.5
MIN_DELTA = 0.20


def _lookup(metrics: pd.DataFrame, model, run, scope, postprocess):
    hit = metrics[
        (metrics["model"] == model)
        & (metrics["run"] == run)
        & (metrics["fake_type"] == scope)
        & (metrics["postprocess"] == postprocess)
    ]
    if hit.empty:
        return None
    return hit.iloc[0]


def error_frames(frame: pd.DataFrame, threshold_of) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        empty = frame.copy()
        return empty, empty
    scored = frame.copy()
    scored["threshold"] = [
        float(threshold_of(model, run)) for model, run in zip(scored["model"], scored["run"])
    ]
    scored["pred"] = (scored["prob_fake"] >= scored["threshold"]).astype(int)
    fn = scored[(scored["label"] == 1) & (scored["pred"] == 0)].copy()
    fp = scored[(scored["label"] == 0) & (scored["pred"] == 1)].copy()
    fn["error"] = "FN"
    fp["error"] = "FP"
    return fn, fp


def _slice_rows(part: pd.DataFrame, error_kind: str, mask: pd.Series, base_rate: float, extra: dict) -> list[dict]:
    rows = []
    for field in META_FIELDS:
        if field not in part.columns:
            continue
        series = part[field]
        for value, group in part.groupby(series.fillna("(비어 있음)").astype(str), dropna=False):
            if value in {"", "nan", "(비어 있음)"}:
                continue
            n = int(len(group))
            n_err = int(mask.loc[group.index].sum())
            rate = n_err / n if n else None
            lift = None if rate is None or base_rate <= 0 else rate / base_rate
            delta = None if rate is None else rate - base_rate
            concentrated = bool(
                n >= MIN_SLICE_N
                and rate is not None
                and delta is not None
                and delta >= MIN_DELTA
                and (base_rate <= 0 or (lift is not None and lift >= MIN_LIFT))
            )
            rows.append(
                {
                    **extra,
                    "error": error_kind,
                    "field": field,
                    "field_ko": META_KO.get(field, field),
                    "value": value,
                    "n": n,
                    "n_error": n_err,
                    "rate": rate,
                    "base_rate": base_rate,
                    "lift": lift,
                    "delta": delta,
                    "concentrated": concentrated,
                }
            )
    return rows


def metadata_slices(frame: pd.DataFrame, threshold_of) -> pd.DataFrame:
    """FP는 진짜 사진의 촬영 조건, FN은 가짜 유형과 원본(A·B) 촬영 조건으로 나눈다."""
    rows = []
    if frame.empty:
        return pd.DataFrame(rows)
    for (model, run, label), group in frame.groupby(["model", "run", "model_label"], sort=False):
        threshold = float(threshold_of(model, run))
        for postprocess in POSTPROCESSES:
            part = group[group["postprocess"] == postprocess]
            reals = part[part["label"] == 0]
            if len(reals):
                fp_mask = reals["prob_fake"] >= threshold
                base = float(fp_mask.mean())
                rows.extend(
                    _slice_rows(
                        reals,
                        "FP",
                        fp_mask,
                        base,
                        {
                            "model": model,
                            "run": run,
                            "model_label": label,
                            "postprocess": postprocess,
                            "fake_type": "real",
                        },
                    )
                )
            for scope in ("all", *NOTION_SCOPES):
                fakes = part[part["label"] == 1] if scope == "all" else part[part["fake_type"] == scope]
                if not len(fakes):
                    continue
                fn_mask = fakes["prob_fake"] < threshold
                base = float(fn_mask.mean())
                rows.extend(
                    _slice_rows(
                        fakes,
                        "FN",
                        fn_mask,
                        base,
                        {
                            "model": model,
                            "run": run,
                            "model_label": label,
                            "postprocess": postprocess,
                            "fake_type": scope,
                        },
                    )
                )
    return pd.DataFrame(rows)


def representatives(fn: pd.DataFrame, fp: pd.DataFrame, per_cell: int) -> pd.DataFrame:
    """칸마다 가장 확신한 오류를 고른다. FN은 확률이 낮은 것, FP는 확률이 높은 것."""
    picked = []
    specs = ((fn, True), (fp, False))
    for frame, lowest in specs:
        if frame.empty:
            continue
        keys = ["model", "run", "postprocess", "fake_type"]
        for _, group in frame.groupby(keys, sort=False):
            ordered = group.sort_values("prob_fake", ascending=lowest)
            picked.append(ordered.head(per_cell))
    if not picked:
        return pd.DataFrame()
    out = pd.concat(picked, ignore_index=True)
    out = out.drop_duplicates(["model", "run", "filename", "postprocess", "error"])
    return out


def paired_comparison(frame: pd.DataFrame, threshold_of) -> pd.DataFrame:
    """같은 파일을 모델 1·2가 어떻게 달리 맞히는지. 유형 A 비교가 핵심이다."""
    if frame.empty or not {"model1", "model2"} <= set(frame["model"]):
        return pd.DataFrame()
    runs1 = set(frame.loc[frame["model"] == "model1", "run"])
    runs2 = set(frame.loc[frame["model"] == "model2", "run"])
    shared = sorted(runs1 & runs2)
    if shared:
        pairs = [(run, run) for run in shared]
    elif len(runs1) == 1 and len(runs2) == 1:
        pairs = [(next(iter(runs1)), next(iter(runs2)))]
    else:
        return pd.DataFrame()

    rows = []
    postprocesses = list(dict.fromkeys([*POSTPROCESSES, *frame["postprocess"].unique()]))
    for run1, run2 in pairs:
        left_all = frame[(frame["model"] == "model1") & (frame["run"] == run1)]
        right_all = frame[(frame["model"] == "model2") & (frame["run"] == run2)]
        for scope in ("all", *NOTION_SCOPES):
            for postprocess in postprocesses:
                left = left_all[left_all["postprocess"] == postprocess]
                right = right_all[right_all["postprocess"] == postprocess]
                if scope != "all":
                    left = left[left["fake_type"].isin([scope, "real"])]
                    right = right[right["fake_type"].isin([scope, "real"])]
                merged = left[["filename", "label", "prob_fake"]].merge(
                    right[["filename", "prob_fake"]],
                    on="filename",
                    suffixes=("_m1", "_m2"),
                )
                if merged.empty:
                    continue
                y = merged["label"].to_numpy()
                c1 = (merged["prob_fake_m1"] >= float(threshold_of("model1", run1))).to_numpy()
                c2 = (merged["prob_fake_m2"] >= float(threshold_of("model2", run2))).to_numpy()
                ok1 = c1 == y
                ok2 = c2 == y
                rows.append(
                    {
                        "run_model1": run1,
                        "run_model2": run2,
                        "fake_type": scope,
                        "postprocess": postprocess,
                        "n": int(len(merged)),
                        "both_correct": int((ok1 & ok2).sum()),
                        "both_wrong": int((~ok1 & ~ok2).sum()),
                        "only_model1": int((ok1 & ~ok2).sum()),
                        "only_model2": int((ok2 & ~ok1).sum()),
                    }
                )
    return pd.DataFrame(rows)


def robustness(metrics: pd.DataFrame) -> pd.DataFrame:
    """원본 대비 후처리에서 Recall·AUC·F1이 얼마나 떨어지는지."""
    rows = []
    if metrics.empty:
        return pd.DataFrame(rows)
    identities = metrics[["model", "run", "model_label"]].drop_duplicates()
    for ident in identities.itertuples(index=False):
        for scope in ("all", *NOTION_SCOPES):
            base = _lookup(metrics, ident.model, ident.run, scope, "original")
            if base is None:
                continue
            for postprocess in POSTPROCESSES:
                if postprocess == "original":
                    continue
                current = _lookup(metrics, ident.model, ident.run, scope, postprocess)
                if current is None:
                    continue
                row = {
                    "model": ident.model,
                    "run": ident.run,
                    "model_label": ident.model_label,
                    "fake_type": scope,
                    "postprocess": postprocess,
                    "n_pos": current["n_pos"],
                }
                for key in ("recall", "f1", "auc", "accuracy"):
                    before, after = base[key], current[key]
                    row[f"{key}_original"] = before
                    row[key] = after
                    row[f"{key}_delta"] = None if before is None or after is None else float(after - before)
                rows.append(row)
    return pd.DataFrame(rows)


def macro_recall(metrics: pd.DataFrame) -> pd.DataFrame:
    """A·B·C Recall의 단순 평균. 장수가 많은 유형에 전체가 끌리는지 보기 위한 값."""
    rows = []
    if metrics.empty:
        return pd.DataFrame(rows)
    identities = metrics[["model", "run", "model_label"]].drop_duplicates()
    for ident in identities.itertuples(index=False):
        for postprocess in POSTPROCESSES:
            values = []
            for scope in NOTION_SCOPES:
                row = _lookup(metrics, ident.model, ident.run, scope, postprocess)
                if row is not None and row["recall"] is not None and int(row["n_pos"]) > 0:
                    values.append(float(row["recall"]))
            overall = _lookup(metrics, ident.model, ident.run, "all", postprocess)
            rows.append(
                {
                    "model": ident.model,
                    "run": ident.run,
                    "model_label": ident.model_label,
                    "postprocess": postprocess,
                    "macro_recall": (sum(values) / len(values)) if values else None,
                    "overall_recall": None if overall is None else overall["recall"],
                    "n_types": len(values),
                }
            )
    return pd.DataFrame(rows)


def weak_cells(metrics: pd.DataFrame, min_pos: int = 5) -> pd.DataFrame:
    """가짜를 놓치는 칸을 Recall이 낮은 순으로 정렬한다."""
    if metrics.empty:
        return pd.DataFrame()
    part = metrics[metrics["fake_type"].isin(NOTION_SCOPES) & (metrics["n_pos"] >= min_pos)].copy()
    if part.empty:
        return part
    part["_recall_sort"] = part["recall"].fillna(1.1)
    part["_f1_sort"] = part["f1"].fillna(1.1)
    return part.sort_values(
        ["_recall_sort", "_f1_sort", "auc"],
        ascending=[True, True, True],
    ).drop(columns=["_recall_sort", "_f1_sort"])


def _fmt(value, digits=3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except TypeError:
        pass
    return f"{float(value):.{digits}f}"


def diagnose(metrics: pd.DataFrame, slices: pd.DataFrame, paired: pd.DataFrame, robust: pd.DataFrame) -> list[str]:
    """숫자에서 바로 읽을 수 있는 문장만 만든다."""
    lines = []
    if metrics.empty:
        return ["계산할 예측이 없습니다."]

    for ident in metrics[["model", "run", "model_label"]].drop_duplicates().itertuples(index=False):
        overall = _lookup(metrics, ident.model, ident.run, "all", "original")
        if overall is None or int(overall["n"]) == 0:
            continue
        name = ident.model_label
        auc, f1 = overall["auc"], overall["f1"]
        if auc is not None and f1 is not None and auc >= 0.80 and f1 < 0.60:
            lines.append(
                f"{name}: 원본 AUC는 {_fmt(auc)}인데 F1은 {_fmt(f1)}입니다. "
                "점수는 진짜·가짜를 나누지만 기준값 0.5가 이 모델의 운영점과 어긋납니다. "
                "val에서 고른 기준값 표를 같이 보십시오."
            )
        elif auc is not None and auc < 0.65:
            lines.append(
                f"{name}: 원본 AUC가 {_fmt(auc)}입니다. 기준값을 바꿔도 진짜와 가짜가 잘 갈라지지 않습니다."
            )
        if overall["prob_gap"] is not None and overall["prob_gap"] < 0.15:
            lines.append(
                f"{name}: 원본에서 가짜 평균 확률과 진짜 평균 확률의 간격이 {_fmt(overall['prob_gap'])}뿐입니다."
            )
        if overall["uncertain_rate"] is not None and overall["uncertain_rate"] >= 0.25:
            lines.append(
                f"{name}: 원본 예측의 {_fmt(overall['uncertain_rate'] * 100, 1)}%가 기준값 ±0.05 안에 있습니다. "
                "확신이 낮은 경계 사례가 많습니다."
            )
        if overall["ece"] is not None and overall["ece"] >= 0.10:
            lines.append(
                f"{name}: 원본 ECE가 {_fmt(overall['ece'])}입니다. 확률값을 확신도로 그대로 말하기 어렵습니다."
            )
        recalls = []
        for scope in NOTION_SCOPES:
            row = _lookup(metrics, ident.model, ident.run, scope, "original")
            if row is not None and row["recall"] is not None and int(row["n_pos"]) >= 5:
                recalls.append((scope, float(row["recall"])))
        if len(recalls) >= 2 and max(v for _, v in recalls) - min(v for _, v in recalls) >= 0.20:
            shown = ", ".join(f"{scope} {_fmt(value)}" for scope, value in recalls)
            lines.append(f"{name}: 원본 Recall이 유형마다 벌어집니다 ({shown}). 전체 점수만으로는 약한 유형이 가려집니다.")

    if not robust.empty:
        dropped = robust[(robust["fake_type"] == "all") & (robust["recall_delta"].notna()) & (robust["recall_delta"] <= -0.15)]
        for row in dropped.itertuples(index=False):
            lines.append(
                f"{row.model_label}: 전체 Recall이 원본 {_fmt(row.recall_original)}에서 "
                f"{POSTPROCESS_KO.get(row.postprocess, row.postprocess)} {_fmt(row.recall)}로 "
                f"{_fmt(row.recall_delta)} 떨어집니다."
            )

    if not paired.empty:
        focus = paired[(paired["fake_type"] == "A") & (paired["postprocess"] == "original")]
        for row in focus.itertuples(index=False):
            if row.only_model2 >= row.only_model1 + 5:
                lines.append(
                    f"유형 A 원본에서 모델 2만 맞힌 경우가 {int(row.only_model2)}장, "
                    f"모델 1만 맞힌 경우가 {int(row.only_model1)}장입니다. 메모 영역을 보는 쪽이 이 칸에서 앞섭니다."
                )
            elif row.only_model1 >= row.only_model2 + 5:
                lines.append(
                    f"유형 A 원본에서 모델 1만 맞힌 경우가 {int(row.only_model1)}장, "
                    f"모델 2만 맞힌 경우가 {int(row.only_model2)}장입니다. 메모 모델이 메모 조작을 더 잘 잡고 있지 않습니다."
                )

    if not slices.empty:
        hot = _collapse_hot(slices[slices["concentrated"]])
        for row in hot.head(6).itertuples(index=False):
            kind = "오탐(FP)" if row.error == "FP" else "놓친 가짜(FN)"
            scope = _target_name(row.fake_type)
            lines.append(
                f"{row.model_label} · {POSTPROCESS_KO.get(row.postprocess, row.postprocess)} · {scope}: "
                f"{kind}이 {row.conditions}에 몰립니다 "
                f"({int(row.n_error)}/{int(row.n)}, 비율 {_fmt(row.rate)}, 전체 대비 차이 {_fmt(row.delta)})."
            )
    if not lines:
        lines.append("표본 안에서 한 유형·한 촬영 조건으로 크게 기운 신호는 기준을 넘지 않았습니다.")
    return lines


def feedback_markdown(
    metrics: pd.DataFrame,
    slices: pd.DataFrame,
    robust: pd.DataFrame,
    eval_split: str,
    threshold: float,
    demo: bool,
) -> str:
    """생성 담당에게 그대로 전달할 약한 유형 메모."""
    lines = ["# 약한 유형 피드백", ""]
    if demo:
        lines.append("> 형식 확인용 합성 데이터입니다. 실제 생성 요청이 아닙니다.")
        lines.append("")
    lines.append(f"평가 split: `{eval_split}`. 판정: `prob_fake >= {threshold:.2f}` 이면 가짜. 양성은 가짜입니다.")
    lines.append("")
    lines.append("보강 이미지는 `is_reinforcement=1`로 **train에만** 추가합니다. test와 val은 바꾸지 않습니다.")
    lines.append("")

    ranked = weak_cells(metrics)
    lines.append("## 우선 보강할 칸")
    lines.append("")
    if ranked.empty:
        lines.append("가짜가 5장 이상인 칸이 없어 순위를 만들지 않았습니다.")
    else:
        for rank, row in enumerate(ranked.head(5).itertuples(index=False), start=1):
            pp = POSTPROCESS_KO.get(row.postprocess, row.postprocess)
            lines.append(
                f"{rank}. **{row.model_label} – {row.fake_type} – {pp}**: "
                f"Recall {_fmt(row.recall)}, F1 {_fmt(row.f1)}, AUC {_fmt(row.auc)} "
                f"(가짜 {int(row.n_pos)}장, 진짜 {int(row.n_neg)}장)."
            )
            lines.append(f"   - {TYPE_HINT.get(row.fake_type, '')} {POSTPROCESS_HINT.get(row.postprocess, '')}")
        lines.append("")
        top = ranked.iloc[0]
        lines.append(
            f"1순위는 {top.model_label}의 유형 {top.fake_type}, "
            f"{POSTPROCESS_KO.get(top.postprocess, top.postprocess)} 입니다."
        )
    lines.append("")
    lines.append("## 원본 Recall")
    lines.append("")
    lines.append("| 모델 | A | B | C | 전체 | 유형 평균 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    if not metrics.empty:
        identities = metrics[["model", "run", "model_label"]].drop_duplicates()
        macros = macro_recall(metrics)
        for ident in identities.itertuples(index=False):
            cells = []
            for scope in (*NOTION_SCOPES, "all"):
                row = _lookup(metrics, ident.model, ident.run, scope, "original")
                cells.append("—" if row is None else _fmt(row.recall))
            macro_row = macros[
                (macros["model"] == ident.model)
                & (macros["run"] == ident.run)
                & (macros["postprocess"] == "original")
            ]
            macro = "—" if macro_row.empty else _fmt(macro_row.iloc[0]["macro_recall"])
            lines.append(f"| {ident.model_label} | " + " | ".join(cells) + f" | {macro} |")
    lines.append("")
    lines.append("유형 평균은 A·B·C Recall을 장수와 상관없이 같은 무게로 본 값입니다. 전체 Recall과 벌어지면 많은 유형이 점수를 끌고 있는 것입니다.")
    lines.append("")

    lines.append("## 후처리로 재현율이 떨어진 칸")
    lines.append("")
    if robust.empty:
        lines.append("비교할 원본 칸이 없습니다.")
    else:
        dropped = robust[robust["recall_delta"].notna() & (robust["recall_delta"] <= -0.10) & (robust["n_pos"] >= 5)]
        dropped = dropped.sort_values("recall_delta")
        if dropped.empty:
            lines.append("가짜 5장 이상인 칸에서 Recall이 0.10 이상 떨어진 후처리는 없습니다.")
        else:
            lines.append("| 모델 | 유형 | 후처리 | 원본 Recall | 해당 Recall | 차이 |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for row in dropped.itertuples(index=False):
                lines.append(
                    f"| {row.model_label} | {SCOPE_NAME[row.fake_type]} | {POSTPROCESS_KO.get(row.postprocess, row.postprocess)} "
                    f"| {_fmt(row.recall_original)} | {_fmt(row.recall)} | {_fmt(row.recall_delta)} |"
                )
    lines.append("")

    lines.append("## 놓친 가짜가 몰리는 촬영 조건")
    lines.append("")
    lines.append("A·B는 원본 진짜 사진의 펜·종이·조명·배경·폰을 따라갑니다. C는 원본 사진이 없어 물건·촬영자만 해당합니다.")
    lines.append("한 사진이 여러 조건에 동시에 속하면 같이 표시됩니다. 원인 하나로 단정하지 않습니다.")
    lines.append("")
    fn = _hot(slices, "FN")
    lines.extend(_hot_lines(fn) if not fn.empty else ["기준을 넘는 FN 집중은 없습니다.", ""])
    if fn.empty:
        lines.append("")

    lines.append("## 오탐한 진짜가 몰리는 촬영 조건")
    lines.append("")
    lines.append("이 조건은 생성 보강보다, 그 외형의 진짜를 더 확보하거나 모델이 그 단서를 가짜로 외우지 않게 하는 쪽에 해당합니다.")
    lines.append("")
    fp = _hot(slices, "FP")
    lines.extend(_hot_lines(fp) if not fp.empty else ["기준을 넘는 FP 집중은 없습니다."])
    lines.append("")

    lines.append("## 모델 담당에게")
    lines.append("")
    if metrics.empty:
        lines.append("비교할 모델이 없습니다.")
    else:
        lines.extend(_model_notes(metrics, paired_from_metrics(metrics)))
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


SCOPE_NAME = {"all": "전체", "A": "A", "B": "B", "C": "C", "real": "진짜"}


def _hot(slices: pd.DataFrame, kind: str) -> pd.DataFrame:
    if slices.empty:
        return slices
    hot = slices[(slices["error"] == kind) & (slices["concentrated"])].copy()
    if hot.empty:
        return hot
    return hot.sort_values(["delta", "n_error"], ascending=False)


def _collapse_hot(hot: pd.DataFrame) -> pd.DataFrame:
    """같은 장수·같은 오류 건수는 한 사진 묶음으로 보고 조건을 한 칸에 모은다."""
    if hot.empty:
        return hot
    work = hot.copy()
    work["_key"] = (
        work["model"].astype(str)
        + "|"
        + work["run"].astype(str)
        + "|"
        + work["postprocess"].astype(str)
        + "|"
        + work["error"].astype(str)
        + "|"
        + work["fake_type"].astype(str)
        + "|"
        + work["n"].astype(str)
        + "|"
        + work["n_error"].astype(str)
    )
    order = {name: index for index, name in enumerate(META_FIELDS)}
    rows = []
    for _, group in work.groupby("_key", sort=False):
        group = group.assign(_ord=group["field"].map(lambda value: order.get(value, 99))).sort_values("_ord")
        record = group.iloc[0].to_dict()
        record["conditions"] = ", ".join(f"{row.field_ko}={row.value}" for row in group.itertuples(index=False))
        rows.append(record)
    out = pd.DataFrame(rows)
    if "_key" in out.columns:
        out = out.drop(columns=["_key"])
    return out.sort_values(["delta", "n_error"], ascending=False)


def _target_name(fake_type: str) -> str:
    if fake_type == "real":
        return "진짜"
    if fake_type == "all":
        return "전체 가짜"
    return f"유형 {fake_type}"


def _hot_lines(hot: pd.DataFrame) -> list[str]:
    collapsed = _collapse_hot(hot).head(16)
    lines = ["| 모델 | 후처리 | 대상 | 조건 | 오류/장수 | 비율 | 전체 비율 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in collapsed.itertuples(index=False):
        target = _target_name(row.fake_type)
        lines.append(
            f"| {row.model_label} | {POSTPROCESS_KO.get(row.postprocess, row.postprocess)} | {target} "
            f"| {row.conditions} | {int(row.n_error)}/{int(row.n)} | {_fmt(row.rate)} | {_fmt(row.base_rate)} |"
        )
    lines.append("")
    return lines


def paired_from_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """피드백 문장용. 상세 짝비교는 paired_comparison이 따로 만든다."""
    return metrics


def _model_notes(metrics: pd.DataFrame, _metrics: pd.DataFrame) -> list[str]:
    lines = []
    m1 = metrics[metrics["model"] == "model1"]
    m2 = metrics[metrics["model"] == "model2"]
    if m1.empty or m2.empty:
        lines.append("모델 1과 모델 2의 예측이 모두 있어야 서로 비교합니다.")
        return lines
    for scope, comment_hi, comment_lo in (
        (
            "A",
            "모델 2의 유형 A Recall이 모델 1보다 높습니다. 메모 영역을 보는 구성이 메모 조작에 유리합니다.",
            "모델 2의 유형 A Recall이 모델 1보다 낮습니다. 메모를 따로 봐도 메모 조작을 더 잘 잡고 있지 않습니다.",
        ),
        (
            "B",
            "모델 1의 유형 B Recall이 모델 2보다 높습니다. 제품 조작은 전체 이미지를 볼 때 더 잘 드러납니다.",
            "모델 2가 유형 B에서도 모델 1보다 높습니다. 메모 밖 단서까지 보고 있는지 확인이 필요합니다.",
        ),
    ):
        a = _mean_recall(m1, scope, "original")
        b = _mean_recall(m2, scope, "original")
        if a is None or b is None:
            continue
        if scope == "A" and b >= a + 0.10:
            lines.append(comment_hi + f" (모델 1 {_fmt(a)}, 모델 2 {_fmt(b)})")
        elif scope == "A" and a >= b + 0.10:
            lines.append(comment_lo + f" (모델 1 {_fmt(a)}, 모델 2 {_fmt(b)})")
        elif scope == "B" and a >= b + 0.10:
            lines.append(comment_hi + f" (모델 1 {_fmt(a)}, 모델 2 {_fmt(b)})")
        elif scope == "B" and b >= a + 0.10:
            lines.append(comment_lo + f" (모델 1 {_fmt(a)}, 모델 2 {_fmt(b)})")
    if not lines:
        lines.append("원본 유형 A·B에서 두 모델의 Recall 차이가 0.10 미만입니다.")
    return lines


def _mean_recall(part: pd.DataFrame, scope: str, postprocess: str):
    hit = part[(part["fake_type"] == scope) & (part["postprocess"] == postprocess)]
    values = [float(v) for v in hit["recall"] if pd.notna(v)]
    if not values:
        return None
    return sum(values) / len(values)
