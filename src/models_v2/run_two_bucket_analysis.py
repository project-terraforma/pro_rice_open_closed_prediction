"""Sweep two-bucket open-safe thresholds over prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_columns(df: pd.DataFrame) -> tuple[str, str]:
    p_closed_candidates = ["p_closed_oof", "p_closed"]
    y_true_candidates = ["y_true_open"]
    p_closed_col = next((c for c in p_closed_candidates if c in df.columns), None)
    y_true_col = next((c for c in y_true_candidates if c in df.columns), None)
    if p_closed_col is None or y_true_col is None:
        raise SystemExit(f"Could not resolve columns in predictions file. Columns: {list(df.columns)}")
    return p_closed_col, y_true_col


def _sweep_thresholds(df: pd.DataFrame, p_closed_col: str, y_true_col: str, thresholds: list[float]) -> pd.DataFrame:
    y_true_open = df[y_true_col].astype(int).values
    y_true_closed = 1 - y_true_open
    n = len(df)
    closed_base_rate = float(y_true_closed.mean()) if n else 0.0
    rows: list[dict] = []

    for t_open in thresholds:
        open_mask = df[p_closed_col].values < t_open
        review_mask = ~open_mask

        n_open = int(open_mask.sum())
        n_review = int(review_mask.sum())

        closed_in_open = int(y_true_closed[open_mask].sum()) if n_open else 0
        closed_in_review = int(y_true_closed[review_mask].sum()) if n_review else 0
        total_closed = int(y_true_closed.sum())

        closed_rate_open = float(closed_in_open / n_open) if n_open else np.nan
        closed_rate_review = float(closed_in_review / n_review) if n_review else np.nan
        capture_review = float(closed_in_review / total_closed) if total_closed else np.nan
        review_lift = float(closed_rate_review / closed_base_rate) if n_review and closed_base_rate > 0 else np.nan

        rows.append(
            {
                "t_open": float(t_open),
                "n_total": int(n),
                "closed_base_rate": closed_base_rate,
                "open_bucket_n": n_open,
                "open_bucket_fraction": float(n_open / n) if n else np.nan,
                "review_bucket_n": n_review,
                "review_bucket_fraction": float(n_review / n) if n else np.nan,
                "closed_in_open_bucket": closed_in_open,
                "closed_rate_in_open_bucket": closed_rate_open,
                "closed_in_review_bucket": closed_in_review,
                "closed_rate_in_review_bucket": closed_rate_review,
                "closure_capture_in_review_bucket": capture_review,
                "review_bucket_lift_vs_base": review_lift,
            }
        )
    return pd.DataFrame(rows)


def _parse_thresholds(raw: str) -> list[float]:
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    thresholds = sorted({float(v) for v in vals})
    if not thresholds:
        raise SystemExit("No thresholds provided.")
    return thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-bucket threshold analysis on prediction artifacts.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--thresholds", type=str, default="0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions_csv)
    p_closed_col, y_true_col = _resolve_columns(df)
    thresholds = _parse_thresholds(args.thresholds)
    out_df = _sweep_thresholds(df, p_closed_col, y_true_col, thresholds)
    out_df.to_csv(args.output_dir / "two_bucket_sweep.csv", index=False)
    (args.output_dir / "two_bucket_run_config.json").write_text(
        json.dumps(
            {
                "predictions_csv": str(args.predictions_csv),
                "p_closed_col": p_closed_col,
                "y_true_col": y_true_col,
                "thresholds": thresholds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
