"""Bootstrap confirm runner: select candidates from existing HPO search trials and re-confirm only those."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_hpo_experiments_weighted import _evaluate_params_cv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap from existing HPO search trials and run confirm CV only on selected candidates."
    )
    parser.add_argument(
        "--search-trials-csv",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "hpo_weighted_pass1" / "hpo_search_trials.csv",
        help="Path to hpo_search_trials.csv from an existing HPO run.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data",
        help="Directory containing train_split.parquet and val_split.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "hpo_weighted_pass1" / "bootstrap_confirm",
    )
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--confirm-n-splits", type=int, default=5)
    parser.add_argument("--confirm-n-repeats", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--show-model-logs",
        action="store_true",
        help="If set, show model/featurizer logs during confirm CV.",
    )
    # Production gate defaults (updated policy)
    parser.add_argument("--prod-accuracy-floor", type=float, default=0.90)
    parser.add_argument("--prod-closed-precision-floor", type=float, default=0.70)
    parser.add_argument("--prod-closed-recall-floor", type=float, default=0.05)
    # Diagnostic gate defaults (updated policy)
    parser.add_argument("--diag-accuracy-floor", type=float, default=0.84)
    parser.add_argument("--diag-closed-precision-floor", type=float, default=0.20)
    return parser.parse_args()


def _pick_production_candidate(group_df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.Series, str]:
    passing = group_df[
        (group_df["accuracy_mean"] >= args.prod_accuracy_floor)
        & (group_df["closed_precision_mean"] >= args.prod_closed_precision_floor)
        & (group_df["closed_recall_mean"] >= args.prod_closed_recall_floor)
    ]
    if not passing.empty:
        winner = passing.sort_values(
            by=["closed_f1_mean", "pr_auc_closed_mean", "accuracy_mean"],
            ascending=False,
        ).iloc[0]
        return winner, "Production gate-pass candidate (closed-F1 tie-break)."

    winner = group_df.sort_values(
        by=["closed_precision_mean", "closed_f1_mean", "accuracy_mean"],
        ascending=False,
    ).iloc[0]
    return winner, "No production gate-pass candidate; fallback to best precision/F1 tradeoff."


def _pick_diagnostic_candidate(group_df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.Series, str]:
    passing = group_df[
        (group_df["accuracy_mean"] >= args.diag_accuracy_floor)
        & (group_df["closed_precision_mean"] >= args.diag_closed_precision_floor)
    ]
    if not passing.empty:
        winner = passing.sort_values(
            by=["closed_f1_mean", "pr_auc_closed_mean", "closed_precision_mean"],
            ascending=False,
        ).iloc[0]
        return winner, "Diagnostic gate-pass candidate (ranked by closed-F1 then PR-AUC)."

    winner = group_df.sort_values(
        by=["closed_f1_mean", "pr_auc_closed_mean", "closed_precision_mean"],
        ascending=False,
    ).iloc[0]
    return winner, "No diagnostic gate-pass candidate; fallback to best closed-F1."


def _load_cv_df(data_dir: Path) -> pd.DataFrame:
    train_path = data_dir / "train_split.parquet"
    val_path = data_dir / "val_split.parquet"
    if not train_path.exists() or not val_path.exists():
        raise SystemExit(f"Missing split files in {data_dir}; expected train_split.parquet and val_split.parquet")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    return pd.concat([train_df, val_df], axis=0, ignore_index=True)


def main() -> None:
    args = _parse_args()
    if not args.search_trials_csv.exists():
        raise SystemExit(f"Search trials file not found: {args.search_trials_csv}")

    search_df = pd.read_csv(args.search_trials_csv)
    if search_df.empty:
        raise SystemExit("Search trials CSV is empty.")

    if "error" in search_df.columns:
        search_df = search_df[search_df["error"].isna()].copy()
    if search_df.empty:
        raise SystemExit("No successful trials found in search CSV after filtering errors.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv_df = _load_cv_df(args.data_dir)

    selected_rows: list[dict] = []
    confirm_rows: list[dict] = []
    confirm_cache: dict[tuple[str, str, int], dict] = {}

    for (model_key, mode), group in search_df.groupby(["model_key", "mode"], sort=True):
        prod_row, prod_reason = _pick_production_candidate(group, args)
        diag_row, diag_reason = _pick_diagnostic_candidate(group, args)

        selections = [
            ("production", prod_row, prod_reason),
            ("diagnostic", diag_row, diag_reason),
        ]
        for gate_type, chosen, rationale in selections:
            trial = int(chosen["trial"])
            cache_key = (model_key, mode, trial)
            if cache_key not in confirm_cache:
                params = json.loads(chosen["params_json"])
                metrics = _evaluate_params_cv(
                    cv_df=cv_df,
                    model_key=model_key,
                    mode=mode,
                    feature_bundle=chosen["feature_bundle"],
                    model_params=params,
                    n_splits=args.confirm_n_splits,
                    n_repeats=args.confirm_n_repeats,
                    random_state=args.random_state,
                    decision_threshold=args.decision_threshold,
                    show_model_logs=args.show_model_logs,
                )
                confirm_cache[cache_key] = metrics

            selected_rows.append(
                {
                    "model_key": model_key,
                    "mode": mode,
                    "gate_type": gate_type,
                    "selected_trial": trial,
                    "feature_bundle": chosen["feature_bundle"],
                    "params_json": chosen["params_json"],
                    "selection_rationale": rationale,
                    "search_accuracy_mean": float(chosen["accuracy_mean"]),
                    "search_closed_precision_mean": float(chosen["closed_precision_mean"]),
                    "search_closed_recall_mean": float(chosen["closed_recall_mean"]),
                    "search_closed_f1_mean": float(chosen["closed_f1_mean"]),
                    "search_pr_auc_closed_mean": float(chosen["pr_auc_closed_mean"]),
                }
            )
            confirm_rows.append(
                {
                    "model_key": model_key,
                    "mode": mode,
                    "gate_type": gate_type,
                    "selected_trial": trial,
                    "feature_bundle": chosen["feature_bundle"],
                    "params_json": chosen["params_json"],
                    **confirm_cache[cache_key],
                }
            )

    selected_df = pd.DataFrame(selected_rows)
    confirm_df = pd.DataFrame(confirm_rows)

    selected_path = args.output_dir / "bootstrap_selected_candidates.csv"
    confirm_path = args.output_dir / "bootstrap_confirm_metrics.csv"
    config_path = args.output_dir / "bootstrap_run_config.json"

    selected_df.to_csv(selected_path, index=False)
    confirm_df.to_csv(confirm_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "search_trials_csv": str(args.search_trials_csv),
                "data_dir": str(args.data_dir),
                "decision_threshold": args.decision_threshold,
                "confirm_cv": {"n_splits": args.confirm_n_splits, "n_repeats": args.confirm_n_repeats},
                "random_state": args.random_state,
                "production_gates": {
                    "accuracy_floor": args.prod_accuracy_floor,
                    "closed_precision_floor": args.prod_closed_precision_floor,
                    "closed_recall_floor": args.prod_closed_recall_floor,
                },
                "diagnostic_gates": {
                    "accuracy_floor": args.diag_accuracy_floor,
                    "closed_precision_floor": args.diag_closed_precision_floor,
                    "ranking": "closed_f1 then pr_auc_closed",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved selected candidates: {selected_path}")
    print(f"Saved confirm metrics: {confirm_path}")
    print(f"Saved run config: {config_path}")


if __name__ == "__main__":
    main()

