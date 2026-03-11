"""LR-only micro Optuna HPO around current best regions, with dual-gate confirm."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import optuna
import pandas as pd

from run_hpo_experiments_weighted import _evaluate_params_cv
from run_hpo_optuna_narrow import (
    _diagnostic_objective_score,
    _save_outputs,
    _select_diagnostic_trial,
    _select_production_trial,
)


def _suggest_lr_micro_params(trial: "optuna.trial.Trial", mode: str) -> dict:
    if mode == "single":
        # Centered around pass-1 Optuna winners (~C 0.03-0.04, class weight ~4.0)
        cw_mode = trial.suggest_categorical("class_weight_mode", ["balanced", "custom"])
        if cw_mode == "balanced":
            class_weight = "balanced"
        else:
            closed_w = trial.suggest_float("closed_class_weight", 3.0, 5.5, step=0.5)
            class_weight = {0: float(closed_w), 1: 1.0}
        c_val = trial.suggest_float("C", 0.015, 0.12, log=True)
    elif mode == "two-stage":
        # Centered around pass-1 Optuna winners (~C 0.024-0.030, class weight ~5.0)
        cw_mode = trial.suggest_categorical("class_weight_mode", ["balanced", "custom"])
        if cw_mode == "balanced":
            class_weight = "balanced"
        else:
            closed_w = trial.suggest_float("closed_class_weight", 3.5, 6.0, step=0.5)
            class_weight = {0: float(closed_w), 1: 1.0}
        c_val = trial.suggest_float("C", 0.012, 0.06, log=True)
    else:
        raise ValueError(f"Unsupported mode for LR micro HPO: {mode}")

    return {
        "C": float(c_val),
        "max_iter": int(trial.suggest_categorical("max_iter", [1000, 1500, 2000])),
        "class_weight": class_weight,
        "solver": "lbfgs",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LR-only micro Optuna HPO with dual-gate selection.")
    parser.add_argument("--modes", nargs="+", default=["single", "two-stage"], choices=["single", "two-stage"])
    parser.add_argument(
        "--feature-bundle",
        choices=["low_only", "low_plus_medium", "full_schema_native", "v2_lr2", "v2_rf_single", "v2_rf_single_no_spatial_prior"],
        default="low_plus_medium",
    )
    parser.add_argument("--n-trials", type=int, default=30)
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
    # Diagnostic gates
    parser.add_argument("--diag-accuracy-floor", type=float, default=0.84)
    parser.add_argument("--diag-closed-precision-floor", type=float, default=0.20)

    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "hpo_optuna_lr_micro_pass1",
    )
    parser.add_argument("--show-model-logs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.data_dir / "train_split.parquet"
    val_path = args.data_dir / "val_split.parquet"
    if not train_path.exists() or not val_path.exists():
        raise SystemExit(f"Missing split files in {args.data_dir}; expected train_split.parquet and val_split.parquet")
    cv_df = pd.concat([pd.read_parquet(train_path), pd.read_parquet(val_path)], axis=0, ignore_index=True)

    run_config = {
        "model": "lr",
        "modes": args.modes,
        "feature_bundle": args.feature_bundle,
        "n_trials": args.n_trials,
        "search_cv": {"n_splits": args.search_n_splits, "n_repeats": args.search_n_repeats},
        "confirm_cv": {"n_splits": args.confirm_n_splits, "n_repeats": args.confirm_n_repeats},
        "decision_threshold": args.decision_threshold,
        "sampler": "optuna.tpe (lr micro)",
        "production_gates": {
            "accuracy_floor": args.prod_accuracy_floor,
            "closed_precision_floor": args.prod_closed_precision_floor,
            "closed_recall_floor": args.prod_closed_recall_floor,
        },
        "diagnostic_gates": {
            "accuracy_floor": args.diag_accuracy_floor,
            "closed_precision_floor": args.diag_closed_precision_floor,
        },
        "random_state": args.random_state,
    }

    all_trial_rows: list[dict] = []
    selected_rows: list[dict] = []
    confirm_rows: list[dict] = []
    confirm_cache: dict[tuple[str, str, int], dict] = {}

    run_start = time.perf_counter()
    total_trials = len(args.modes) * args.n_trials
    done_trials = 0

    for mode in args.modes:
        print(f"\n=== LR Micro Optuna HPO: {mode} ===")
        mode_rows: list[dict] = []
        mode_start = time.perf_counter()

        def objective(trial: "optuna.trial.Trial") -> float:
            nonlocal done_trials
            t0 = time.perf_counter()
            params = _suggest_lr_micro_params(trial, mode)
            try:
                metrics = _evaluate_params_cv(
                    cv_df=cv_df,
                    model_key="lr",
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
                    "model_key": "lr",
                    "mode": mode,
                    "trial": int(trial.number + 1),
                    "feature_bundle": args.feature_bundle,
                    "params_json": json.dumps(params, sort_keys=True),
                    "objective_score": score,
                    **metrics,
                }
            except Exception as exc:  # pragma: no cover
                row = {
                    "model_key": "lr",
                    "mode": mode,
                    "trial": int(trial.number + 1),
                    "feature_bundle": args.feature_bundle,
                    "params_json": json.dumps(params, sort_keys=True),
                    "error": str(exc),
                }
                score = -1e9

            mode_rows.append(row)
            done_trials += 1
            mode_elapsed = time.perf_counter() - mode_start
            mode_done = len(mode_rows)
            mode_avg = mode_elapsed / max(mode_done, 1)
            block_eta = (mode_avg * max(args.n_trials - mode_done, 0)) / 60.0
            run_elapsed = time.perf_counter() - run_start
            run_avg = run_elapsed / max(done_trials, 1)
            full_eta = (run_avg * max(total_trials - done_trials, 0)) / 60.0
            print(
                f"  trial {mode_done}/{args.n_trials} done "
                f"(trial {time.perf_counter() - t0:.1f}s, avg {mode_avg:.1f}s, "
                f"block ETA {block_eta:.1f}m, full ETA {full_eta:.1f}m)"
            )
            return score

        sampler_seed = args.random_state + (11 if mode == "single" else 17)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=sampler_seed))
        study.optimize(objective, n_trials=args.n_trials)

        all_trial_rows.extend(mode_rows)
        trial_df = pd.DataFrame(mode_rows)
        ok_df = trial_df[trial_df.get("error").isna()] if "error" in trial_df.columns else trial_df
        if ok_df.empty:
            print(f"[skip] lr ({mode}) had no successful trials")
            continue

        prod_row, prod_reason = _select_production_trial(ok_df, args)
        diag_row, diag_reason = _select_diagnostic_trial(ok_df, args)
        picks = [("production", prod_row, prod_reason), ("diagnostic", diag_row, diag_reason)]
        for gate_type, winner, rationale in picks:
            trial_num = int(winner["trial"])
            selected_rows.append(
                {
                    "model_key": "lr",
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

            key = ("lr", mode, trial_num)
            if key not in confirm_cache:
                params = json.loads(winner["params_json"])
                confirm_cache[key] = _evaluate_params_cv(
                    cv_df=cv_df,
                    model_key="lr",
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
                    "model_key": "lr",
                    "mode": mode,
                    "gate_type": gate_type,
                    "feature_bundle": args.feature_bundle,
                    "selected_trial": trial_num,
                    "params_json": winner["params_json"],
                    **confirm_cache[key],
                }
            )

        _save_outputs(args.output_dir, all_trial_rows, selected_rows, confirm_rows, run_config)

    t, s, c, cfg = _save_outputs(args.output_dir, all_trial_rows, selected_rows, confirm_rows, run_config)
    print(f"\nSaved trials: {t}")
    print(f"Saved selected trials (dual-gate): {s}")
    print(f"Saved confirm metrics (dual-gate): {c}")
    print(f"Saved run config: {cfg}")


if __name__ == "__main__":
    main()
