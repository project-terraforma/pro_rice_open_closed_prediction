"""Narrow HPO (Optuna) for frontier models with dual-gate selection + confirm CV."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import optuna
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: optuna. Install with `pip install optuna`.") from exc

from run_hpo_experiments_weighted import _evaluate_params_cv


def _suggest_params(trial: "optuna.trial.Trial", model_key: str, mode: str) -> dict:
    if model_key == "lr":
        cw_mode_choices = ["none", "balanced", "custom"] if mode == "single" else ["balanced", "custom"]
        cw_mode = trial.suggest_categorical("class_weight_mode", cw_mode_choices)
        if cw_mode == "none":
            class_weight = None
        elif cw_mode == "balanced":
            class_weight = "balanced"
        else:
            closed_w = trial.suggest_float("closed_class_weight", 2.0, 6.0, step=0.5)
            class_weight = {0: float(closed_w), 1: 1.0}
        return {
            "C": trial.suggest_float("C", 0.005, 0.5, log=True),
            "max_iter": int(trial.suggest_categorical("max_iter", [1000, 1500, 2000, 3000])),
            "class_weight": class_weight,
            "solver": "lbfgs",
        }
    if model_key == "rf":
        cw_mode = trial.suggest_categorical("class_weight_mode", ["balanced", "custom"])
        if cw_mode == "balanced":
            class_weight = "balanced"
        else:
            closed_w = trial.suggest_float("closed_class_weight", 2.0, 6.0, step=0.5)
            class_weight = {0: float(closed_w), 1: 1.0}
        max_depth_choice = trial.suggest_categorical("max_depth", [8, 10, 12, 14, 16, None])
        return {
            "n_estimators": int(trial.suggest_int("n_estimators", 250, 700, step=25)),
            "max_depth": None if max_depth_choice is None else int(max_depth_choice),
            "min_samples_leaf": int(trial.suggest_int("min_samples_leaf", 3, 6)),
            "min_samples_split": int(trial.suggest_int("min_samples_split", 3, 12)),
            "max_features": trial.suggest_categorical("max_features", ["log2", "sqrt"]),
            "class_weight": class_weight,
        }
    raise ValueError(f"Unsupported model_key for narrow Optuna HPO: {model_key}")


def _diagnostic_objective_score(metrics: dict, acc_floor: float, cprec_floor: float) -> float:
    closed_f1 = float(metrics["closed_f1_mean"])
    pr_auc = float(metrics["pr_auc_closed_mean"])
    acc = float(metrics["accuracy_mean"])
    cprec = float(metrics["closed_precision_mean"])

    # Primary: maximize closed F1, with PR-AUC as small tie-break.
    score = closed_f1 + 0.10 * pr_auc
    # Soft penalties for violating diagnostic floors.
    if acc < acc_floor:
        score -= 2.0 * (acc_floor - acc)
    if cprec < cprec_floor:
        score -= 2.5 * (cprec_floor - cprec)
    return float(score)


def _select_production_trial(group_df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.Series, str]:
    passing = group_df[
        (group_df["accuracy_mean"] >= args.prod_accuracy_floor)
        & (group_df["closed_precision_mean"] >= args.prod_closed_precision_floor)
        & (group_df["closed_recall_mean"] >= args.prod_closed_recall_floor)
    ]
    if not passing.empty:
        winner = passing.sort_values(
            by=["closed_precision_mean", "closed_f1_mean", "pr_auc_closed_mean", "accuracy_mean"],
            ascending=False,
        ).iloc[0]
        return winner, "Production gate-pass candidate selected (precision-first tie-break)."

    winner = group_df.sort_values(
        by=["closed_precision_mean", "closed_f1_mean", "pr_auc_closed_mean"],
        ascending=False,
    ).iloc[0]
    return winner, "No production gate-pass candidate; fallback to precision/F1 best-effort."


def _select_diagnostic_trial(group_df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.Series, str]:
    passing = group_df[
        (group_df["accuracy_mean"] >= args.diag_accuracy_floor)
        & (group_df["closed_precision_mean"] >= args.diag_closed_precision_floor)
    ]
    if not passing.empty:
        winner = passing.sort_values(
            by=["closed_f1_mean", "pr_auc_closed_mean", "closed_precision_mean", "accuracy_mean"],
            ascending=False,
        ).iloc[0]
        return winner, "Diagnostic gate-pass candidate selected (closed-F1 first)."

    winner = group_df.sort_values(
        by=["closed_f1_mean", "pr_auc_closed_mean", "closed_precision_mean"],
        ascending=False,
    ).iloc[0]
    return winner, "No diagnostic gate-pass candidate; fallback to best closed-F1."


def _save_outputs(
    output_dir: Path,
    all_trial_rows: list[dict],
    selected_rows: list[dict],
    confirm_rows: list[dict],
    run_config: dict,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = output_dir / "hpo_search_trials.csv"
    selected_path = output_dir / "hpo_selected_trials_dualgate.csv"
    confirm_path = output_dir / "hpo_confirm_metrics_dualgate.csv"
    config_path = output_dir / "hpo_run_config_dualgate.json"

    pd.DataFrame(all_trial_rows).to_csv(trials_path, index=False)
    pd.DataFrame(selected_rows).to_csv(selected_path, index=False)
    pd.DataFrame(confirm_rows).to_csv(confirm_path, index=False)
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    return trials_path, selected_path, confirm_path, config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Narrow Optuna HPO on frontier models with dual-gate selection.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lr", "rf"],
        choices=["lr", "rf"],
        help="Narrow pass defaults to frontier models only.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["single", "two-stage"],
        choices=["single", "two-stage"],
        help="RF uses single-stage only.",
    )
    parser.add_argument(
        "--feature-bundle",
        choices=["low_only", "low_plus_medium", "full_schema_native"],
        default="low_plus_medium",
    )
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--search-n-splits", type=int, default=5)
    parser.add_argument("--search-n-repeats", type=int, default=1)
    parser.add_argument("--confirm-n-splits", type=int, default=5)
    parser.add_argument("--confirm-n-repeats", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--decision-threshold", type=float, default=0.5)

    # Production gates (strict)
    parser.add_argument("--prod-accuracy-floor", type=float, default=0.90)
    parser.add_argument("--prod-closed-precision-floor", type=float, default=0.70)
    parser.add_argument("--prod-closed-recall-floor", type=float, default=0.05)
    # Diagnostic gates (research)
    parser.add_argument("--diag-accuracy-floor", type=float, default=0.84)
    parser.add_argument("--diag-closed-precision-floor", type=float, default=0.20)

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "hpo_optuna_narrow_pass1",
    )
    parser.add_argument(
        "--show-model-logs",
        action="store_true",
        help="If set, show model/featurizer logs during HPO. Default is quiet mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.data_dir / "train_split.parquet"
    val_path = args.data_dir / "val_split.parquet"
    if not train_path.exists() or not val_path.exists():
        raise SystemExit(f"Missing split files in {args.data_dir}; expected train_split.parquet and val_split.parquet")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    cv_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    run_config = {
        "models": args.models,
        "modes": args.modes,
        "feature_bundle": args.feature_bundle,
        "n_trials": args.n_trials,
        "search_cv": {"n_splits": args.search_n_splits, "n_repeats": args.search_n_repeats},
        "confirm_cv": {"n_splits": args.confirm_n_splits, "n_repeats": args.confirm_n_repeats},
        "decision_threshold": args.decision_threshold,
        "sampler": "optuna.tpe",
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
        "random_state": args.random_state,
    }

    all_trial_rows: list[dict] = []
    selected_rows: list[dict] = []
    confirm_rows: list[dict] = []
    confirm_cache: dict[tuple[str, str, int], dict] = {}
    run_start = time.perf_counter()
    model_mode_pairs: list[tuple[str, str]] = []
    for mk in args.models:
        mm = ["single"] if mk == "rf" else args.modes
        for md in mm:
            model_mode_pairs.append((mk, md))
    total_trials_full_run = len(model_mode_pairs) * args.n_trials
    completed_trials_full_run = 0

    for model_key in args.models:
        model_modes = ["single"] if model_key == "rf" else args.modes
        for mode in model_modes:
            print(f"\n=== Narrow Optuna HPO: {model_key} ({mode}) ===")
            mode_rows: list[dict] = []
            mode_start = time.perf_counter()

            def objective(trial: "optuna.trial.Trial") -> float:
                nonlocal completed_trials_full_run
                trial_start = time.perf_counter()
                params = _suggest_params(trial, model_key, mode)
                try:
                    metrics = _evaluate_params_cv(
                        cv_df=cv_df,
                        model_key=model_key,
                        mode=mode,
                        feature_bundle=args.feature_bundle,
                        model_params=params,
                        n_splits=args.search_n_splits,
                        n_repeats=args.search_n_repeats,
                        random_state=args.random_state + trial.number + 1,
                        decision_threshold=args.decision_threshold,
                        show_model_logs=args.show_model_logs,
                    )
                    score = _diagnostic_objective_score(
                        metrics,
                        acc_floor=args.diag_accuracy_floor,
                        cprec_floor=args.diag_closed_precision_floor,
                    )
                    row = {
                        "model_key": model_key,
                        "mode": mode,
                        "trial": int(trial.number + 1),
                        "feature_bundle": args.feature_bundle,
                        "params_json": json.dumps(params, sort_keys=True),
                        "objective_score": score,
                        **metrics,
                    }
                except Exception as exc:
                    row = {
                        "model_key": model_key,
                        "mode": mode,
                        "trial": int(trial.number + 1),
                        "feature_bundle": args.feature_bundle,
                        "params_json": json.dumps(params, sort_keys=True),
                        "error": str(exc),
                    }
                    score = -1e9

                mode_rows.append(row)
                elapsed_total = time.perf_counter() - mode_start
                done = len(mode_rows)
                avg_t = elapsed_total / max(done, 1)
                eta_minutes = (avg_t * max(args.n_trials - done, 0)) / 60.0
                completed_trials_full_run += 1
                run_elapsed_total = time.perf_counter() - run_start
                run_avg_per_trial = run_elapsed_total / max(completed_trials_full_run, 1)
                run_remaining_trials = max(total_trials_full_run - completed_trials_full_run, 0)
                run_eta_minutes = (run_avg_per_trial * run_remaining_trials) / 60.0
                print(
                    f"  trial {done}/{args.n_trials} done "
                    f"(trial {time.perf_counter() - trial_start:.1f}s, avg {avg_t:.1f}s, "
                    f"block ETA {eta_minutes:.1f}m, full ETA {run_eta_minutes:.1f}m)"
                )
                return score

            seed_offset = {
                ("lr", "single"): 11,
                ("lr", "two-stage"): 17,
                ("rf", "single"): 23,
            }.get((model_key, mode), 0)
            sampler = optuna.samplers.TPESampler(seed=args.random_state + seed_offset)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(objective, n_trials=args.n_trials)

            all_trial_rows.extend(mode_rows)
            trial_df = pd.DataFrame(mode_rows)
            ok_df = trial_df[trial_df.get("error").isna()] if "error" in trial_df.columns else trial_df
            if ok_df.empty:
                print(f"[skip] {model_key} ({mode}) had no successful trials")
                continue

            prod_row, prod_reason = _select_production_trial(ok_df, args)
            diag_row, diag_reason = _select_diagnostic_trial(ok_df, args)
            picks = [("production", prod_row, prod_reason), ("diagnostic", diag_row, diag_reason)]

            for gate_type, winner, rationale in picks:
                trial_num = int(winner["trial"])
                selected_rows.append(
                    {
                        "model_key": model_key,
                        "mode": mode,
                        "gate_type": gate_type,
                        "feature_bundle": args.feature_bundle,
                        "selected_trial": trial_num,
                        "params_json": winner["params_json"],
                        "selection_rationale": rationale,
                        "search_accuracy_mean": float(winner["accuracy_mean"]),
                        "search_closed_precision_mean": float(winner["closed_precision_mean"]),
                        "search_closed_recall_mean": float(winner["closed_recall_mean"]),
                        "search_closed_f1_mean": float(winner["closed_f1_mean"]),
                        "search_pr_auc_closed_mean": float(winner["pr_auc_closed_mean"]),
                    }
                )

                cache_key = (model_key, mode, trial_num)
                if cache_key not in confirm_cache:
                    params = json.loads(winner["params_json"])
                    confirm_cache[cache_key] = _evaluate_params_cv(
                        cv_df=cv_df,
                        model_key=model_key,
                        mode=mode,
                        feature_bundle=args.feature_bundle,
                        model_params=params,
                        n_splits=args.confirm_n_splits,
                        n_repeats=args.confirm_n_repeats,
                        random_state=args.random_state,
                        decision_threshold=args.decision_threshold,
                        show_model_logs=args.show_model_logs,
                    )
                confirm_rows.append(
                    {
                        "model_key": model_key,
                        "mode": mode,
                        "gate_type": gate_type,
                        "feature_bundle": args.feature_bundle,
                        "selected_trial": trial_num,
                        "params_json": winner["params_json"],
                        **confirm_cache[cache_key],
                    }
                )

            _save_outputs(
                output_dir=args.output_dir,
                all_trial_rows=all_trial_rows,
                selected_rows=selected_rows,
                confirm_rows=confirm_rows,
                run_config=run_config,
            )

    trials_path, selected_path, confirm_path, config_path = _save_outputs(
        output_dir=args.output_dir,
        all_trial_rows=all_trial_rows,
        selected_rows=selected_rows,
        confirm_rows=confirm_rows,
        run_config=run_config,
    )
    print(f"\nSaved trials: {trials_path}")
    print(f"Saved selected trials (dual-gate): {selected_path}")
    print(f"Saved confirm metrics (dual-gate): {confirm_path}")
    print(f"Saved run config: {config_path}")


if __name__ == "__main__":
    main()
