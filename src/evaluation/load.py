"""예측 CSV를 데이터 규약 형식으로 맞추고 split·메타데이터와 연결한다."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.schema import POSTPROCESSES
from src.evaluation.vocab import MODEL_KO

FILE_COLUMNS = ("filename", "file", "image", "image_path", "path", "filepath")
PROB_COLUMNS = ("prob_fake", "fake_probability", "probability", "prob", "score")
POST_COLUMNS = ("postprocess", "preprocess", "augmentation")

VALUE_ALIASES = {
    "original": "original",
    "orig": "original",
    "raw": "original",
    "원본": "original",
    "kakao": "kakao",
    "kakaotalk": "kakao",
    "katalk": "kakao",
    "kakaocompress": "kakao",
    "kakaocompressed": "kakao",
    "compressed": "kakao",
    "compress": "kakao",
    "카톡": "kakao",
    "카톡압축": "kakao",
    "카카오": "kakao",
    "카카오톡": "kakao",
    "resize": "resize",
    "resized": "resize",
    "리사이즈": "resize",
    "크기변경": "resize",
    "recapture": "recapture",
    "reshoot": "recapture",
    "rephoto": "recapture",
    "screen": "recapture",
    "재촬영": "recapture",
}


def read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame


def basename(value) -> str:
    text = str(value).strip().strip('"').strip("'")
    if not text or text.lower() == "nan":
        return ""
    return Path(text.replace("\\", "/")).name


def normalize_postprocess(value) -> str:
    raw = str(value).strip().lower()
    key = raw.replace(" ", "").replace("_", "").replace("-", "")
    if key in VALUE_ALIASES:
        return VALUE_ALIASES[key]
    if raw in POSTPROCESSES:
        return raw
    return raw


def normalize_fake_type(value) -> str:
    text = str(value).strip()
    if text.lower() in {"real", "genuine", "0", "nan", ""}:
        return "real" if text.lower() in {"real", "genuine", "0"} else ""
    if text.upper() in {"A", "B", "C"}:
        return text.upper()
    return text


def display_label(model: str, run: str, n_runs: int) -> str:
    base = MODEL_KO.get(model, model)
    if n_runs <= 1 or run in {"", "default"}:
        return base
    return f"{base} ({run})"


def infer_identity(path: Path) -> tuple[str, str]:
    """파일명·폴더명에서 model1/model2와 run 이름을 읽는다."""
    stem = path.stem
    blob = "/".join(path.parts).lower().replace("\\", "/")
    model = None
    for key in ("model1", "model2"):
        if key in blob:
            model = key
            break
    if model is None:
        model = stem
    lower = stem.lower()
    if model in {"model1", "model2"} and lower.startswith(model):
        run = stem[len(model):].lstrip("_-") or "default"
    elif lower in {"test_predictions", "predictions", "pred"}:
        run = "default"
    else:
        run = stem or "default"
    return model, run


def _pick(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _rescale(prob: pd.Series) -> tuple[pd.Series, str | None]:
    values = pd.to_numeric(prob, errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return values, None
    if float(finite.max()) > 1.5 and float(finite.max()) <= 100 and float(finite.min()) >= 0:
        return values / 100.0, "확률값이 0~100 범위라 100으로 나눠 0~1로 맞췄습니다."
    return values, None


def normalize_prediction(frame: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, list[str]]:
    """계약 CSV와 모델 브랜치 CSV(image_path, fake_probability)를 같은 열로 맞춘다."""
    notes = []
    source = path.name
    file_col = _pick(frame, FILE_COLUMNS)
    prob_col = _pick(frame, PROB_COLUMNS)
    if file_col is None or prob_col is None:
        raise ValueError(
            f"{source}에서 파일명·확률 열을 찾지 못했습니다. "
            "filename/prob_fake 또는 image_path/fake_probability 가 필요합니다."
        )
    out = pd.DataFrame()
    out["filename"] = frame[file_col].map(basename)
    out["prob_fake"], note = _rescale(frame[prob_col])
    if note:
        notes.append(f"{source}: {note}")
    if "image_path" in frame.columns:
        out["image_path"] = frame["image_path"].astype(str)
    elif file_col in {"image_path", "path", "filepath"}:
        out["image_path"] = frame[file_col].astype(str)

    if "postprocess" in frame.columns or "preprocess" in frame.columns or "augmentation" in frame.columns:
        col = _pick(frame, POST_COLUMNS)
        out["postprocess"] = frame[col].map(normalize_postprocess)
    elif "domain" in frame.columns:
        normalized = frame["domain"].map(normalize_postprocess)
        known = float(normalized.isin(POSTPROCESSES).mean()) if len(normalized) else 0.0
        if known >= 0.8:
            out["postprocess"] = normalized
            notes.append(f"{source}: domain 열을 후처리로 사용했습니다.")
        else:
            out["postprocess"] = "original"
            notes.append(
                f"{source}: domain 값이 후처리명(original/kakao/resize/recapture)이 아니라 "
                "후처리를 original로 두었습니다."
            )
    else:
        out["postprocess"] = "original"
        notes.append(f"{source}: 후처리 열이 없어 original로 두었습니다.")

    if "label" in frame.columns:
        out["pred_label"] = pd.to_numeric(frame["label"], errors="coerce")

    model, run = infer_identity(path)
    out["model"] = model
    out["run"] = run
    out["prediction_file"] = path.name
    out = out[out["filename"].astype(str).str.len() > 0]
    invalid = ~np.isfinite(out["prob_fake"]) | (out["prob_fake"] < -1e-6) | (out["prob_fake"] > 1 + 1e-6)
    n_invalid = int(invalid.sum())
    if n_invalid:
        notes.append(f"{source}: 확률이 0~1 밖이거나 비어 있는 행 {n_invalid}개를 제외했습니다.")
        out = out.loc[~invalid].copy()
    out["prob_fake"] = out["prob_fake"].clip(0, 1)
    return out.reset_index(drop=True), notes


def assign_labels(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        frame["model_label"] = []
        return frame
    counts = frame.groupby("model")["run"].nunique().to_dict()
    labeled = frame.copy()
    labeled["model_label"] = [
        display_label(model, run, int(counts.get(model, 1)))
        for model, run in zip(labeled["model"], labeled["run"])
    ]
    return labeled


def load_prediction_files(paths: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    notes = []
    for path in paths:
        raw = read_csv(path)
        frame, file_notes = normalize_prediction(raw, path)
        notes.extend(file_notes)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), notes
    merged = pd.concat(frames, ignore_index=True)
    key = ["model", "run", "filename", "postprocess"]
    dup = merged.duplicated(key, keep=False)
    if dup.any():
        n_dup = int(merged.loc[dup, key].drop_duplicates().shape[0])
        notes.append(f"같은 파일·후처리 예측이 {n_dup}건 겹쳐 확률을 평균냈습니다.")
        others = [col for col in merged.columns if col not in key + ["prob_fake"]]
        averaged = merged.groupby(key, as_index=False)["prob_fake"].mean()
        if others:
            side = merged[key + others].drop_duplicates(key, keep="first")
            averaged = averaged.merge(side, on=key, how="left")
        merged = averaged
    return assign_labels(merged), notes


def discover_predictions(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    return sorted(item for item in path.rglob("*.csv") if item.is_file())


def load_table(path: Path | None, required: list[str], kind: str) -> pd.DataFrame:
    if path is None or not path.is_file():
        return pd.DataFrame(columns=required)
    frame = read_csv(path)
    missing = [col for col in required if col not in frame.columns]
    if missing and len(frame.columns) > 0 and not frame.empty:
        raise ValueError(f"{kind}에 필요한 열이 없습니다: {', '.join(missing)}")
    for col in required:
        if col not in frame.columns:
            frame[col] = pd.NA
    return frame


def attach_split(pred: pd.DataFrame, split: pd.DataFrame, eval_split: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """split.csv의 정답을 붙인다. 예측 파일 안의 label은 점검용으로만 쓴다."""
    notes = []
    table = split.copy()
    table["filename"] = table["filename"].map(basename)
    table["label"] = pd.to_numeric(table["label"], errors="coerce")
    table["fake_type"] = table["fake_type"].map(normalize_fake_type)
    table["split"] = table["split"].astype(str).str.strip()
    for col in ("object", "group_id", "photographer"):
        if col not in table.columns:
            table[col] = ""
        table[col] = table[col].fillna("").astype(str)

    pred_label = None
    if "pred_label" in pred.columns:
        pred_label = pred[["model", "run", "filename", "postprocess", "pred_label"]].copy()

    keep = [col for col in pred.columns if col != "pred_label"]
    merged = pred[keep].merge(
        table[["filename", "label", "fake_type", "object", "group_id", "photographer", "split"]],
        on="filename",
        how="left",
    )
    if pred_label is not None:
        check = pred_label.merge(table[["filename", "label"]], on="filename", how="left")
        both = check["pred_label"].notna() & check["label"].notna()
        mismatch = both & (check["pred_label"].astype(float) != check["label"].astype(float))
        n_mis = int(mismatch.sum())
        if n_mis:
            notes.append(
                f"예측 파일의 label과 split.csv가 다른 행이 {n_mis}개입니다. 지표는 split.csv를 기준으로 계산했습니다."
            )

    known = merged["split"].notna()
    off_split = merged[known & (merged["split"] != eval_split)]
    unknown = merged[~known]
    if len(unknown):
        notes.append(f"split.csv에 없는 예측 {len(unknown)}행은 지표에서 빼었습니다.")
    if len(off_split):
        notes.append(f"{eval_split}가 아닌 예측 {len(off_split)}행은 이 결과표에서 빼었습니다.")

    selected = merged[merged["split"] == eval_split].copy()
    selected = selected[selected["label"].isin([0, 1])]
    bad_type = ~selected["fake_type"].isin(["real", "A", "B", "C"])
    if bad_type.any():
        notes.append(f"fake_type이 real/A/B/C가 아닌 행 {int(bad_type.sum())}개는 지표에서 빼었습니다.")
        selected = selected.loc[~bad_type].copy()
    label_clash = (
        ((selected["fake_type"] == "real") & (selected["label"] != 0))
        | (selected["fake_type"].isin(["A", "B", "C"]) & (selected["label"] != 1))
    )
    if label_clash.any():
        notes.append(f"label과 fake_type이 어긋난 행이 {int(label_clash.sum())}개입니다. label을 정답으로 사용했습니다.")
    return selected.reset_index(drop=True), table, notes


def attach_metadata(frame: pd.DataFrame, real_meta: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """진짜는 자기 메타데이터, A·B는 원본 진짜의 촬영 조건을 붙인다. C는 원본이 없다."""
    out = frame.copy().reset_index(drop=True)
    out["is_reinforcement"] = 0
    source = out["filename"].astype(str).copy()
    if not mapping.empty and "fake_filename" in mapping.columns:
        link = mapping.copy()
        link["fake_filename"] = link["fake_filename"].map(basename)
        raw_source = link["source_filename"] if "source_filename" in link.columns else ""
        link["source_filename"] = pd.Series(raw_source, index=link.index).map(
            lambda v: basename(v) if pd.notna(v) and str(v).strip() else ""
        )
        if "is_reinforcement" in link.columns:
            link["is_reinforcement"] = pd.to_numeric(link["is_reinforcement"], errors="coerce").fillna(0)
        else:
            link["is_reinforcement"] = 0
        slim = link[["fake_filename", "source_filename", "is_reinforcement"]].drop_duplicates("fake_filename")
        out = out.drop(columns=["is_reinforcement"]).merge(
            slim, left_on="filename", right_on="fake_filename", how="left"
        )
        linked = out["source_filename"].fillna("").astype(str)
        is_real = out["label"].to_numpy() == 0
        source = pd.Series(np.where(is_real, out["filename"].astype(str), linked), index=out.index)
        out["is_reinforcement"] = pd.to_numeric(out["is_reinforcement"], errors="coerce").fillna(0)
    out["source_filename"] = source.astype(str)

    fields = ["pen", "paper", "lighting", "background", "phone_model", "memo_text"]
    if not real_meta.empty and "filename" in real_meta.columns:
        meta = real_meta.copy()
        meta["filename"] = meta["filename"].map(basename)
        keep = ["filename"] + [col for col in fields if col in meta.columns]
        meta = meta[keep].drop_duplicates("filename").rename(columns={"filename": "source_filename"})
        out = out.merge(meta, on="source_filename", how="left")
    for col in fields:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def coverage_table(split: pd.DataFrame, frame: pd.DataFrame, eval_split: str) -> pd.DataFrame:
    expected = set(split.loc[split["split"] == eval_split, "filename"].map(basename))
    rows = []
    if frame.empty:
        return pd.DataFrame(columns=["model", "run", "model_label", "postprocess", "n_expected", "n_found", "coverage"])
    for (model, run, label), group in frame.groupby(["model", "run", "model_label"], sort=False):
        for postprocess, part in group.groupby("postprocess", sort=False):
            found = set(part["filename"]) & expected
            rows.append(
                {
                    "model": model,
                    "run": run,
                    "model_label": label,
                    "postprocess": postprocess,
                    "n_expected": len(expected),
                    "n_found": len(found),
                    "coverage": (len(found) / len(expected)) if expected else None,
                }
            )
    return pd.DataFrame(rows)


def collect_split_warnings(split: pd.DataFrame, frame: pd.DataFrame, eval_split: str) -> list[str]:
    notes = []
    if "group_id" in split.columns:
        grouped = split.copy()
        grouped["group_id"] = grouped["group_id"].fillna("").astype(str)
        grouped = grouped[grouped["group_id"].str.len() > 0]
        nunique = grouped.groupby("group_id")["split"].nunique()
        leaked = nunique[nunique > 1]
        if len(leaked):
            sample = ", ".join(list(leaked.index[:5]))
            notes.append(
                f"같은 group_id가 서로 다른 split에 나뉜 경우가 {len(leaked)}개입니다 ({sample}). 데이터 누수일 수 있습니다."
            )
    if not frame.empty and "is_reinforcement" in frame.columns:
        flagged = frame[(frame["split"] == eval_split) & (pd.to_numeric(frame["is_reinforcement"], errors="coerce") == 1)]
        n_flag = flagged["filename"].nunique()
        if n_flag:
            notes.append(
                f"보강용(is_reinforcement=1) 가짜 {n_flag}개가 {eval_split}에 있습니다. 보강 데이터는 train에만 있어야 합니다."
            )
    return notes
