"""Per-model hyperparameter search with policy-gate-aware selection."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from shared_metrics import compute_metrics


def _safe_predict_open_proba(model, df: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(df)[:, 1]
        except TypeError:
            pass
    if hasattr(model, "model") and hasattr(model, "extract_features"):
        x = model.extract_features(df)
        return model.model.predict_proba(x)[:, 1]
    raise ValueError(f"Could not obtain probabilities from model type: {type(model)}")


def _fit_model(model, train_df: pd.DataFrame):
    try:
        return model.fit(train_df, val_df=None)
    except TypeError:
        return model.fit(train_df)


def _sample_log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def _sample_params(rng: np.random.Generator, model_key: str) -> dict:
    if model_key == "lr":
        # LR search: focus on regularization strength (C) and convergence budget.
        # - C on log scale spans strong -> weak regularization.
        # - max_iter allows convergence on wide/sparse feature sets.
        return {
            "C": _sample_log_uniform(rng, 0.01, 20.0),
            "max_iter": int(rng.choice([1000, 1500, 2000, 3000])),
            "class_weight": "balanced",
            "solver": "lbfgs",
        }
    if model_key == "lightgbm":
        # LightGBM search:
        # - learning_rate + n_estimators: core boosting tradeoff (step size vs rounds).
        # - num_leaves controls tree complexity.
        # - subsample/colsample add regularization and robustness.
        # - min_child_samples + reg_lambda reduce overfitting on sparse/noisy slices.
        return {
            "n_estimators": int(rng.integers(150, 650)),
            "learning_rate": _sample_log_uniform(rng, 0.01, 0.2),
            "num_leaves": int(rng.integers(15, 128)),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.6, 1.0)),
            "min_child_samples": int(rng.integers(10, 81)),
            "reg_lambda": _sample_log_uniform(rng, 0.1, 10.0),
            "class_weight": "balanced",
        }
    if model_key == "rf":
        # RF search:
        # - n_estimators drives ensemble stability.
        # - max_depth / min_samples_* control tree smoothness vs variance.
        # - max_features controls de-correlation among trees.
        max_depth_choice = rng.choice([None, 6, 8, 10, 12, 16, 20], p=[0.1, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1])
        return {
            "n_estimators": int(rng.integers(200, 1000)),
            "max_depth": None if max_depth_choice is None else int(max_depth_choice),
            "min_samples_leaf": int(rng.integers(1, 6)),
            "min_samples_split": int(rng.integers(2, 11)),
            "max_features": rng.choice(["sqrt", "log2", None]),
            "class_weight": "balanced",
        }
    if model_key == "xgboost":
        # XGBoost search mirrors LightGBM principles:
        # - learning_rate + n_estimators: boosting tradeoff.
        # - max_depth + min_child_weight: tree complexity and split conservatism.
        # - subsample/colsample and reg_lambda: regularization against overfit.
        return {
            "n_estimators": int(rng.integers(150, 700)),
            "learning_rate": _sample_log_uniform(rng, 0.01, 0.2),
            "max_depth": int(rng.integers(3, 11)),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.6, 1.0)),
            "min_child_weight": float(rng.uniform(1.0, 10.0)),
            "reg_lambda": _sample_log_uniform(rng, 0.1, 10.0),
        }
    raise ValueError(f"Unknown model_key: {model_key}")


def _build_model(model_key: str, mode: str, feature_bundle: str, model_params: dict):
    if model_key == "lr":
        from logistic_regression_v2 import UnifiedLogisticRegression

        return UnifiedLogisticRegression(mode=mode, feature_bundle=feature_bundle, model_params=model_params)
    if model_key == "lightgbm":
        from lightgbm_model_v2 import LightGBMModel

        return LightGBMModel(
            mode=mode,
            feature_bundle=feature_bundle,
            use_source_confidence=False,
            model_params=model_params,
        )
    if model_key == "rf":
        from random_forest_model_v2 import RandomForestModel

        return RandomForestModel(feature_bundle=feature_bundle, model_params=model_params)
    if model_key == "xgboost":
        from xgboost_model_v2 import XGBoostModel

        return XGBoostModel(mode=mode, feature_bundle=feature_bundle, model_params=model_params)
    raise ValueError(f"Unknown model_key: {model_key}")


def _evaluate_params_cv(
    cv_df: pd.DataFrame,
    model_key: str,
    mode: str,
    feature_bundle: str,
    model_params: dict,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    decision_threshold: float,
    show_model_logs: bool,
) -> dict:
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    y = cv_df["open"].astype(int).values
    fold_metrics: list[dict] = []

    for tr_idx, va_idx in splitter.split(cv_df, y):
        tr_df = cv_df.iloc[tr_idx].reset_index(drop=True)
        va_df = cv_df.iloc[va_idx].reset_index(drop=True)
        model = _build_model(model_key, mode, feature_bundle, model_params)

        if show_model_logs:
            _fit_model(model, tr_df)
            p_open = _safe_predict_open_proba(model, va_df)
        else:
            # Keep HPO output readable by suppressing noisy model-level logs/warnings.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    _fit_model(model, tr_df)
                    p_open = _safe_predict_open_proba(model, va_df)

        y_pred = (p_open >= decision_threshold).astype(int)
        fold_metrics.append(
            compute_metrics(
                y_true=va_df["open"].astype(int).values,
                y_pred=y_pred,
                y_score_open=p_open,
            )
        )

    fm = pd.DataFrame(fold_metrics)
    out = {}
    for col in fm.columns:
        out[f"{col}_mean"] = float(fm[col].mean())
        out[f"{col}_std"] = float(fm[col].std(ddof=0))
    return out


def _select_best_trial(
    trial_df: pd.DataFrame,
    accuracy_floor: float,
    closed_precision_floor: float,
    pr_auc_top_band: float,
) -> tuple[pd.Series, str]:
    df = trial_df.copy()
    df["passes_accuracy_floor"] = df["accuracy_mean"] >= accuracy_floor
    df["passes_closed_precision_floor"] = df["closed_precision_mean"] >= closed_precision_floor
    df["passes_policy_gates"] = df["passes_accuracy_floor"] & df["passes_closed_precision_floor"]

    passing = df[df["passes_policy_gates"]]
    if not passing.empty:
        best_pr = float(passing["pr_auc_closed_mean"].max())
        band = passing[passing["pr_auc_closed_mean"] >= (best_pr - pr_auc_top_band)]
        winner = band.sort_values(
            by=["closed_f1_mean", "closed_precision_mean", "accuracy_mean"],
            ascending=False,
        ).iloc[0]
        return winner, "Selected from gate-pass set using PR-AUC top-band then closed-F1 tie-break."

    winner = df.sort_values(
        by=["closed_f1_mean", "closed_precision_mean", "pr_auc_closed_mean"],
        ascending=False,
    ).iloc[0]
    return winner, "No trial passed gates; fallback to max closed-F1."


def parse_args():
    parser = argparse.ArgumentParser(description="Per-model HPO with policy-gate-aware selection.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lr", "lightgbm", "rf", "xgboost"],
        choices=["lr", "lightgbm", "rf", "xgboost"],
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
    parser.add_argument("--accuracy-floor", type=float, default=0.85)
    parser.add_argument("--closed-precision-floor", type=float, default=0.30)
    parser.add_argument("--pr-auc-top-band", type=float, default=0.01)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "hpo",
    )
    parser.add_argument(
        "--show-model-logs",
        action="store_true",
        help="If set, show model/featurizer logs during HPO. Default is quiet mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_path = args.data_dir / "train_split.parquet"
    val_path = args.data_dir / "val_split.parquet"
    if not train_path.exists() or not val_path.exists():
        raise SystemExit(f"Missing split files in {args.data_dir}; expected train_split.parquet and val_split.parquet")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    cv_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.random_state)
    all_trial_rows: list[dict] = []
    selected_rows: list[dict] = []
    confirm_rows: list[dict] = []

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
            print(f"\n=== HPO search: {model_key} ({mode}) ===")
            trial_rows = []
            mode_start = time.perf_counter()
            for trial in range(1, args.n_trials + 1):
                trial_start = time.perf_counter()
                params = _sample_params(rng, model_key)
                try:
                    metrics = _evaluate_params_cv(
                        cv_df=cv_df,
                        model_key=model_key,
                        mode=mode,
                        feature_bundle=args.feature_bundle,
                        model_params=params,
                        n_splits=args.search_n_splits,
                        n_repeats=args.search_n_repeats,
                        random_state=args.random_state + trial,
                        decision_threshold=args.decision_threshold,
                        show_model_logs=args.show_model_logs,
                    )
                    row = {
                        "model_key": model_key,
                        "mode": mode,
                        "trial": trial,
                        "feature_bundle": args.feature_bundle,
                        "params_json": json.dumps(params, sort_keys=True),
                        **metrics,
                    }
                except Exception as exc:
                    row = {
                        "model_key": model_key,
                        "mode": mode,
                        "trial": trial,
                        "feature_bundle": args.feature_bundle,
                        "params_json": json.dumps(params, sort_keys=True),
                        "error": str(exc),
                    }
                trial_rows.append(row)
                trial_elapsed = time.perf_counter() - trial_start
                elapsed_total = time.perf_counter() - mode_start
                avg_per_trial = elapsed_total / trial
                remaining = max(args.n_trials - trial, 0)
                eta_seconds = avg_per_trial * remaining
                eta_minutes = eta_seconds / 60.0
                completed_trials_full_run += 1
                run_elapsed_total = time.perf_counter() - run_start
                run_avg_per_trial = run_elapsed_total / max(completed_trials_full_run, 1)
                run_remaining_trials = max(total_trials_full_run - completed_trials_full_run, 0)
                run_eta_minutes = (run_avg_per_trial * run_remaining_trials) / 60.0
                print(
                    f"  trial {trial}/{args.n_trials} done "
                    f"(trial {trial_elapsed:.1f}s, avg {avg_per_trial:.1f}s, "
                    f"block ETA {eta_minutes:.1f}m, full ETA {run_eta_minutes:.1f}m)"
                )

            trial_df = pd.DataFrame(trial_rows)
            all_trial_rows.extend(trial_rows)
            ok_df = trial_df[trial_df.get("error").isna()] if "error" in trial_df.columns else trial_df
            if ok_df.empty:
                print(f"[skip] {model_key} ({mode}) had no successful trials")
                continue

            winner, rationale = _select_best_trial(
                trial_df=ok_df,
                accuracy_floor=args.accuracy_floor,
                closed_precision_floor=args.closed_precision_floor,
                pr_auc_top_band=args.pr_auc_top_band,
            )
            selected_rows.append(
                {
                    "model_key": model_key,
                    "mode": mode,
                    "feature_bundle": args.feature_bundle,
                    "selected_trial": int(winner["trial"]),
                    "params_json": winner["params_json"],
                    "selection_rationale": rationale,
                    "accuracy_mean": float(winner["accuracy_mean"]),
                    "closed_precision_mean": float(winner["closed_precision_mean"]),
                    "closed_recall_mean": float(winner["closed_recall_mean"]),
                    "closed_f1_mean": float(winner["closed_f1_mean"]),
                    "pr_auc_closed_mean": float(winner["pr_auc_closed_mean"]),
                }
            )

            winner_params = json.loads(winner["params_json"])
            confirm_metrics = _evaluate_params_cv(
                cv_df=cv_df,
                model_key=model_key,
                mode=mode,
                feature_bundle=args.feature_bundle,
                model_params=winner_params,
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
                    "feature_bundle": args.feature_bundle,
                    "selected_trial": int(winner["trial"]),
                    "params_json": winner["params_json"],
                    **confirm_metrics,
                }
            )

    trials_df = pd.DataFrame(all_trial_rows)
    selected_df = pd.DataFrame(selected_rows)
    confirm_df = pd.DataFrame(confirm_rows)

    trials_path = args.output_dir / "hpo_search_trials.csv"
    selected_path = args.output_dir / "hpo_selected_trials.csv"
    confirm_path = args.output_dir / "hpo_confirm_metrics.csv"
    config_path = args.output_dir / "hpo_run_config.json"
    trials_df.to_csv(trials_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    confirm_df.to_csv(confirm_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "models": args.models,
                "modes": args.modes,
                "feature_bundle": args.feature_bundle,
                "n_trials": args.n_trials,
                "search_cv": {"n_splits": args.search_n_splits, "n_repeats": args.search_n_repeats},
                "confirm_cv": {"n_splits": args.confirm_n_splits, "n_repeats": args.confirm_n_repeats},
                "decision_threshold": args.decision_threshold,
                "accuracy_floor": args.accuracy_floor,
                "closed_precision_floor": args.closed_precision_floor,
                "pr_auc_top_band": args.pr_auc_top_band,
                "random_state": args.random_state,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved trials: {trials_path}")
    print(f"Saved selected trials: {selected_path}")
    print(f"Saved confirm metrics: {confirm_path}")
    print(f"Saved run config: {config_path}")
    if not confirm_df.empty:
        print("\nConfirmed configs (sorted by closed_f1_mean):")
        cols = [
            "model_key",
            "mode",
            "accuracy_mean",
            "closed_precision_mean",
            "closed_recall_mean",
            "closed_f1_mean",
            "pr_auc_closed_mean",
        ]
        print(confirm_df.sort_values("closed_f1_mean", ascending=False)[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
