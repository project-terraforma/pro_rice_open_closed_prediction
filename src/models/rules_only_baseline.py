"""Rules-only baseline for open/closed prediction.

This uses low-cost, schema-native features and simple thresholds.
Default behavior: predict OPEN in the middle zone to reduce false-closed risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class RuleConfig:
    open_threshold: float = 3.0
    closed_threshold: float = 0.0
    default_open: bool = True
    fresh_days: int = 180
    stale_days: int = 1500
    fresh_bonus: float = 1.0
    stale_penalty: float = 1.5


def has_value(x) -> int:
    if x is None:
        return 0
    if isinstance(x, float) and np.isnan(x):
        return 0
    try:
        return 1 if len(x) > 0 else 0
    except Exception:
        return 1


def count_sources(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0


def max_source_conf(sources) -> float:
    if sources is None:
        return float("nan")
    try:
        vals = [s.get("confidence") for s in sources if s.get("confidence") is not None]
        return max(vals) if vals else float("nan")
    except Exception:
        return float("nan")


def max_update_time(sources):
    if sources is None:
        return None
    try:
        times = [s.get("update_time") for s in sources if s.get("update_time")]
        return max(times) if times else None
    except Exception:
        return None


def score_row(row: pd.Series, cfg: RuleConfig) -> float:
    score = 0.0

    conf = row.get("confidence", np.nan)
    if conf >= 0.95:
        score += 3.0
    elif conf >= 0.90:
        score += 2.0
    elif conf >= 0.80:
        score += 1.0
    elif conf < 0.65:
        score -= 2.0
    elif conf < 0.75:
        score -= 1.0

    sources_n = row.get("sources_n", 0)
    if sources_n >= 2:
        score += 1.0
    elif sources_n == 1:
        score -= 1.0

    max_conf = row.get("max_source_conf", float("nan"))
    if not np.isnan(max_conf):
        if max_conf >= 0.90:
            score += 1.0
        elif max_conf < 0.70:
            score -= 1.0

    has_website = row.get("websites_present", 0)
    has_phone = row.get("phones_present", 0)
    has_social = row.get("socials_present", 0)
    has_address = row.get("addresses_present", 0)

    if has_website:
        score += 1.0
    if has_phone:
        score += 1.0
    if has_social:
        score += 0.5
    if has_address:
        score += 0.5
    if not has_website and not has_phone:
        score -= 1.0

    recency_days = row.get("recency_days", np.nan)
    if not np.isnan(recency_days):
        if recency_days <= cfg.fresh_days:
            score += cfg.fresh_bonus
        elif recency_days >= cfg.stale_days:
            score -= cfg.stale_penalty

    return score


def predict_from_score(score: float, cfg: RuleConfig) -> int:
    if score >= cfg.open_threshold:
        return 1
    if score <= cfg.closed_threshold:
        return 0
    return 1 if cfg.default_open else 0


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["websites", "phones", "socials", "addresses"]:
        out[f"{col}_present"] = out[col].apply(has_value)
    out["sources_n"] = out["sources"].apply(count_sources)
    out["max_source_conf"] = out["sources"].apply(max_source_conf)
    out["max_update_time"] = out["sources"].apply(max_update_time)
    out["max_update_time"] = pd.to_datetime(out["max_update_time"], errors="coerce")
    snapshot_time = out["max_update_time"].max()
    if pd.notna(snapshot_time):
        out["recency_days"] = (snapshot_time - out["max_update_time"]).dt.days
    else:
        out["recency_days"] = np.nan
    return out


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    def safe_div(n, d):
        return n / d if d else 0.0

    precision_open = safe_div(tp, tp + fp)
    recall_open = safe_div(tp, tp + fn)
    precision_closed = safe_div(tn, tn + fn)
    recall_closed = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    f1_open = safe_div(2 * precision_open * recall_open, precision_open + recall_open)
    f1_closed = safe_div(2 * precision_closed * recall_closed, precision_closed + recall_closed)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision_open": precision_open,
        "recall_open": recall_open,
        "precision_closed": precision_closed,
        "recall_closed": recall_closed,
        "f1_open": f1_open,
        "f1_closed": f1_closed,
        "accuracy": accuracy,
    }


def run_threshold_sweep(scores: np.ndarray, y_true: np.ndarray) -> None:
    results = []
    for open_th in np.arange(2.0, 6.1, 0.5):
        for closed_th in np.arange(-2.0, 1.1, 0.5):
            if closed_th >= open_th:
                continue
            cfg = RuleConfig(open_threshold=open_th, closed_threshold=closed_th, default_open=True)
            preds = np.array([predict_from_score(s, cfg) for s in scores])
            metrics = evaluate(y_true, preds)
            results.append(
                {
                    "open_th": open_th,
                    "closed_th": closed_th,
                    **metrics,
                }
            )

    results = pd.DataFrame(results)
    print("\n=== Threshold sweep (top by open precision) ===")
    print(
        results.sort_values(["precision_open", "recall_open"], ascending=False)
        .head(5)
        .to_string(index=False)
    )
    print("\n=== Threshold sweep (top by closed recall) ===")
    print(
        results.sort_values(["recall_closed", "precision_closed"], ascending=False)
        .head(5)
        .to_string(index=False)
    )
    print("\n=== Threshold sweep (top by closed precision) ===")
    print(
        results.sort_values(["precision_closed", "recall_closed"], ascending=False)
        .head(5)
        .to_string(index=False)
    )


def run_recency_sweep(df: pd.DataFrame) -> None:
    results = []
    for stale_days in [730, 1095, 1500]:
        for stale_penalty in [1.0, 1.5, 2.0]:
            cfg = RuleConfig(stale_days=stale_days, stale_penalty=stale_penalty)
            scores = df.apply(lambda r: score_row(r, cfg), axis=1).to_numpy()
            y_true = df["open"].to_numpy()
            preds = np.array([predict_from_score(s, cfg) for s in scores])
            metrics = evaluate(y_true, preds)
            results.append(
                {
                    "stale_days": stale_days,
                    "stale_penalty": stale_penalty,
                    **metrics,
                }
            )

    results = pd.DataFrame(results)
    print("\n=== Recency sweep (top by closed recall) ===")
    print(
        results.sort_values(["recall_closed", "precision_closed"], ascending=False)
        .head(5)
        .to_string(index=False)
    )
    print("\n=== Recency sweep (top by closed precision) ===")
    print(
        results.sort_values(["precision_closed", "recall_closed"], ascending=False)
        .head(5)
        .to_string(index=False)
    )
    print("\n=== Recency sweep (top by open precision) ===")
    print(
        results.sort_values(["precision_open", "recall_open"], ascending=False)
        .head(5)
        .to_string(index=False)
    )


def main(split: str = "val") -> None:
    split_path = DATA_DIR / f"{split}_split.parquet"
    if not split_path.exists():
        raise SystemExit(f"Missing split file: {split_path}")

    df = pd.read_parquet(split_path)
    df = add_features(df)

    cfg = RuleConfig()
    scores = df.apply(lambda r: score_row(r, cfg), axis=1)
    preds = scores.apply(lambda s: predict_from_score(s, cfg)).to_numpy()
    y_true = df["open"].to_numpy()

    metrics = evaluate(y_true, preds)

    print(f"Split: {split}")
    print(f"Rows: {len(df)}")
    print("Confusion: tp={tp}, tn={tn}, fp={fp}, fn={fn}".format(**metrics))
    print("Precision (open): {:.3f}".format(metrics["precision_open"]))
    print("Recall (open): {:.3f}".format(metrics["recall_open"]))
    print("F1 (open): {:.3f}".format(metrics["f1_open"]))
    print("Precision (closed): {:.3f}".format(metrics["precision_closed"]))
    print("Recall (closed): {:.3f}".format(metrics["recall_closed"]))
    print("F1 (closed): {:.3f}".format(metrics["f1_closed"]))
    print("Accuracy: {:.3f}".format(metrics["accuracy"]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rules-only baseline")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--sweep", action="store_true", help="Run threshold sweep")
    parser.add_argument("--recency-sweep", action="store_true", help="Sweep stale recency settings")
    args = parser.parse_args()

    main(split=args.split)
    if args.sweep:
        df = pd.read_parquet(DATA_DIR / f"{args.split}_split.parquet")
        df = add_features(df)
        scores = df.apply(lambda r: score_row(r, RuleConfig()), axis=1).to_numpy()
        y_true = df["open"].to_numpy()
        run_threshold_sweep(scores, y_true)
    if args.recency_sweep:
        df = pd.read_parquet(DATA_DIR / f"{args.split}_split.parquet")
        df = add_features(df)
        run_recency_sweep(df)
