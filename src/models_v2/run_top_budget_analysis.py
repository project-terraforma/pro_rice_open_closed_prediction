"""Sweep top-budget review slices over prediction artifacts."""

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


def _parse_budgets(raw: str) -> list[float]:
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    budgets = sorted({float(v) for v in vals})
    if not budgets:
        raise SystemExit("No budgets provided.")
    for budget in budgets:
        if budget <= 0 or budget > 100:
            raise SystemExit(f"Budget must be in (0, 100]. Got {budget}.")
    return budgets


def _top_n(n_total: int, budget_pct: float) -> int:
    return min(n_total, max(1, int(np.ceil(n_total * (budget_pct / 100.0)))))


def _sweep_budgets(df: pd.DataFrame, p_closed_col: str, y_true_col: str, budgets: list[float]) -> pd.DataFrame:
    ranked = df.sort_values([p_closed_col, "row_idx"] if "row_idx" in df.columns else [p_closed_col], ascending=[False, True] if "row_idx" in df.columns else [False]).reset_index(drop=True)
    y_true_open = ranked[y_true_col].astype(int).to_numpy()
    y_true_closed = 1 - y_true_open
    p_closed = ranked[p_closed_col].astype(float).to_numpy()
    n_total = len(ranked)
    total_closed = int(y_true_closed.sum())
    total_open = int(y_true_open.sum())
    closed_base_rate = float(y_true_closed.mean()) if n_total else np.nan
    rows: list[dict] = []

    for budget_pct in budgets:
        review_n = _top_n(n_total, budget_pct)
        review_mask = np.zeros(n_total, dtype=bool)
        review_mask[:review_n] = True

        closed_in_review = int(y_true_closed[review_mask].sum())
        open_in_review = int(y_true_open[review_mask].sum())
        closed_rate_in_review = float(closed_in_review / review_n) if review_n else np.nan
        closure_capture = float(closed_in_review / total_closed) if total_closed else np.nan
        lift_vs_base = float(closed_rate_in_review / closed_base_rate) if closed_base_rate and review_n else np.nan
        min_p_closed_in_review = float(p_closed[review_n - 1]) if review_n else np.nan

        rows.append(
            {
                "review_budget_pct": float(budget_pct),
                "review_n": int(review_n),
                "n_total": int(n_total),
                "closed_base_rate": closed_base_rate,
                "total_closed": total_closed,
                "total_open": total_open,
                "closed_in_review": closed_in_review,
                "open_in_review": open_in_review,
                "closed_rate_in_review": closed_rate_in_review,
                "closure_capture": closure_capture,
                "lift_vs_base": lift_vs_base,
                "min_p_closed_in_review": min_p_closed_in_review,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run top-budget review analysis on prediction artifacts.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--budgets", type=str, default="1,2,5,10,15,20,25")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions_csv)
    p_closed_col, y_true_col = _resolve_columns(df)
    budgets = _parse_budgets(args.budgets)
    out_df = _sweep_budgets(df, p_closed_col, y_true_col, budgets)
    out_df.to_csv(args.output_dir / "top_budget_sweep.csv", index=False)
    (args.output_dir / "top_budget_run_config.json").write_text(
        json.dumps(
            {
                "predictions_csv": str(args.predictions_csv),
                "p_closed_col": p_closed_col,
                "y_true_col": y_true_col,
                "budgets": budgets,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
