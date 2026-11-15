"""모델 1 vs 모델 2 비교표: 두 test 예측 CSV를 image_path로 join해 유형·후처리별로 비교한다."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from .dataset import infer_postprocess

REQUIRED = ["image_path", "label", "fake_type", "fake_probability", "prediction"]
POSTPROCESS_ORDER = ["original", "kakao", "resize", "recapture"]
TYPE_ORDER = ["real", "A", "B", "C", "unknown"]
TABLE_COLUMNS = ["postprocess", "fake_type", "n", "metric", "model1_rate", "model2_rate",
                 "model1_auc", "model2_auc"]
NAN = float("nan")


def _read(path) -> pd.DataFrame:
    """예측 CSV를 읽고 컬럼·중복을 검사한다."""
    path = Path(path)
    df = pd.read_csv(path, keep_default_na=False,
                     dtype={"image_path": str, "fake_type": str})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: 필수 컬럼이 없습니다: {missing}")
    if df["image_path"].duplicated().any():
        raise ValueError(f"{path}: image_path가 중복됩니다")
    df = df[REQUIRED].copy()
    df["label"] = df["label"].astype(int)
    df["prediction"] = df["prediction"].astype(int)
    df["fake_probability"] = df["fake_probability"].astype(float)
    return df


def _auc(sub: pd.DataFrame, reals: pd.DataFrame, col: str) -> float:
    """같은 후처리의 real(0) 대 해당 가짜 유형(1) AUC. real이 없으면 NaN."""
    if reals.empty or sub.empty:
        return NAN
    y = [0] * len(reals) + [1] * len(sub)
    return float(roc_auc_score(y, list(reals[col]) + list(sub[col])))


def compare(model1_csv, model2_csv) -> tuple[pd.DataFrame, dict]:
    """두 예측 CSV를 비교한 표와 join 정보를 돌려준다."""
    m1, m2 = _read(model1_csv), _read(model2_csv)
    joined = m1.merge(m2, on="image_path", how="inner", suffixes=("_1", "_2"))
    info = {"n_joined": len(joined),
            "missing_in_model1": int((~m2["image_path"].isin(m1["image_path"])).sum()),
            "missing_in_model2": int((~m1["image_path"].isin(m2["image_path"])).sum())}
    if joined.empty:
        raise ValueError("두 파일에 공통 image_path가 없습니다")
    bad = joined[(joined["label_1"] != joined["label_2"])
                 | (joined["fake_type_1"] != joined["fake_type_2"])]
    if not bad.empty:
        raise ValueError("두 파일의 label/fake_type이 다릅니다: "
                         + ", ".join(bad["image_path"].head(3)))
    joined["label"] = joined["label_1"]
    joined["fake_type"] = joined["fake_type_1"]
    joined["postprocess"] = joined["image_path"].map(infer_postprocess)

    rows = []
    for pp in [p for p in POSTPROCESS_ORDER if p in set(joined["postprocess"])]:
        in_pp = joined[joined["postprocess"] == pp]
        reals = in_pp[in_pp["label"] == 0]
        for ft in [t for t in TYPE_ORDER if t in set(in_pp["fake_type"])]:
            sub = in_pp[in_pp["fake_type"] == ft]
            is_real = ft == "real"
            rows.append({
                "postprocess": pp, "fake_type": ft, "n": len(sub),
                "metric": "fpr" if is_real else "recall",
                "model1_rate": float(sub["prediction_1"].mean()),
                "model2_rate": float(sub["prediction_2"].mean()),
                "model1_auc": NAN if is_real else _auc(sub, reals, "fake_probability_1"),
                "model2_auc": NAN if is_real else _auc(sub, reals, "fake_probability_2"),
            })
    return pd.DataFrame(rows, columns=TABLE_COLUMNS), info


def _pct(v: float) -> str:
    return "-" if pd.isna(v) else f"{v:.1%}"


def _num(v: float) -> str:
    return "-" if pd.isna(v) else f"{v:.3f}"


def _markdown(table: pd.DataFrame, info: dict) -> str:
    lines = []
    a = table[(table["postprocess"] == "original") & (table["fake_type"] == "A")]
    if not a.empty:
        r = a.iloc[0]
        lines += [f"A 유형 탐지율 (original): 모델1 {r.model1_rate:.1%} → 모델2 {r.model2_rate:.1%}", ""]
    lines += ["| " + " | ".join(TABLE_COLUMNS) + " |",
              "|" + "|".join(["---"] * len(TABLE_COLUMNS)) + "|"]
    for r in table.itertuples(index=False):
        cells = [r.postprocess, r.fake_type, str(r.n), r.metric, _pct(r.model1_rate),
                 _pct(r.model2_rate), _num(r.model1_auc), _num(r.model2_auc)]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"공통 행 수 (n_joined): {info['n_joined']}",
              f"모델 1에 없는 행: {info['missing_in_model1']}",
              f"모델 2에 없는 행: {info['missing_in_model2']}"]
    return "\n".join(lines) + "\n"


def write_comparison(table: pd.DataFrame, info: dict, out_dir) -> None:
    """comparison.csv와 comparison.md를 쓴다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "comparison.csv", index=False)
    (out_dir / "comparison.md").write_text(_markdown(table, info), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="모델 1 vs 모델 2 비교표")
    parser.add_argument("--model1", required=True, help="모델 1 test_predictions.csv")
    parser.add_argument("--model2", required=True, help="모델 2 test_predictions.csv")
    parser.add_argument("--out", default="outputs/model2/comparison")
    args = parser.parse_args()
    try:
        table, info = compare(args.model1, args.model2)
        write_comparison(table, info, args.out)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print((Path(args.out) / "comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
