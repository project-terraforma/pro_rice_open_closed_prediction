"""Unified cross-validation runner for v2 model family experiments.

This runner enforces fold-safe training lifecycle:
- fresh model/featurizer per fold
- label-derived priors fit only on fold-train data
- fold-val transformed using fold-train fitted state
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from shared_metrics import compute_metrics


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    builder: Callable[[str, str], object]
    modes: tuple[str, ...]


def _build_lr(mode: str, feature_bundle: str):
    from logistic_regression_v2 import UnifiedLogisticRegression

    return UnifiedLogisticRegression(mode=mode, feature_bundle=feature_bundle)


def _build_lgbm(mode: str, feature_bundle: str):
    from lightgbm_model_v2 import LightGBMModel

    return LightGBMModel(mode=mode, feature_bundle=feature_bundle, use_source_confidence=False)


def _build_rf(mode: str, feature_bundle: str):
    from random_forest_model_v2 import RandomForestModel

    _ = mode  # RF v2 is single-stage only.
    return RandomForestModel(feature_bundle=feature_bundle)


def _build_xgb(mode: str, feature_bundle: str):
    from xgboost_model_v2 import XGBoostModel

    return XGBoostModel(mode=mode, feature_bundle=feature_bundle)


def get_model_specs(selected: set[str]) -> list[ModelSpec]:
    specs = [
        ModelSpec("lr", "LogisticRegression", _build_lr, ("single", "two-stage")),
        ModelSpec("lightgbm", "LightGBM", _build_lgbm, ("single", "two-stage")),
        ModelSpec("rf", "RandomForest", _build_rf, ("single",)),
        ModelSpec("xgboost", "XGBoost", _build_xgb, ("single", "two-stage")),
    ]
    return [s for s in specs if s.key in selected]


def safe_predict_open_proba(model, df: pd.DataFrame):
    """Return open-class probabilities for a fitted model."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(df)[:, 1]
        except TypeError:
            # Some model classes may expose sklearn model directly with X input only.
            pass

    if hasattr(model, "model") and hasattr(model, "extract_features"):
        X = model.extract_features(df)
        return model.model.predict_proba(X)[:, 1]

    raise ValueError(f"Could not obtain probabilities from model type: {type(model)}")


def fit_model(model, train_df: pd.DataFrame):
    """Fit model with best-effort signature compatibility across model families."""
    try:
        return model.fit(train_df, val_df=None)
    except TypeError:
        # xgboost_model_v2.fit accepts scale_pos_weight kwarg, but val_df is still valid.
        return model.fit(train_df)


def run_cv(
    cv_df: pd.DataFrame,
    model_specs: list[ModelSpec],
    feature_bundle: str,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    decision_threshold: float,
) -> pd.DataFrame:
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    y = cv_df["open"].astype(int).values
    rows: list[dict] = []

    for split_idx, (train_idx, val_idx) in enumerate(rskf.split(cv_df, y), start=1):
        repeat_idx = (split_idx - 1) // n_splits + 1
        fold_idx = (split_idx - 1) % n_splits + 1

        train_df = cv_df.iloc[train_idx].reset_index(drop=True)
        val_df = cv_df.iloc[val_idx].reset_index(drop=True)

        for spec in model_specs:
            for mode in spec.modes:
                try:
                    model = spec.builder(mode, feature_bundle)
                except Exception as exc:
                    print(f"[skip] {spec.label} ({mode}) init failed: {exc}")
                    continue

                try:
                    fit_model(model, train_df)
                    p_open = safe_predict_open_proba(model, val_df)
                    y_pred = (p_open >= decision_threshold).astype(int)
                    metrics = compute_metrics(
                        y_true=val_df["open"].astype(int).values,
                        y_pred=y_pred,
                        y_score_open=p_open,
                    )
                except Exception as exc:
                    print(
                        f"[skip] {spec.label} ({mode}) failed on repeat={repeat_idx}, fold={fold_idx}: {exc}"
                    )
                    continue

                rows.append(
                    {
                        "model_family": spec.label,
                        "model_key": spec.key,
                        "mode": mode,
                        "feature_bundle": feature_bundle,
                        "repeat": repeat_idx,
                        "fold": fold_idx,
                        "n_train": len(train_df),
                        "n_val": len(val_df),
                        "decision_threshold": decision_threshold,
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)


def summarize_metrics(cv_metrics: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model_family", "model_key", "mode", "feature_bundle", "decision_threshold"]
    metric_cols = [
        "accuracy",
        "open_precision",
        "open_recall",
        "open_f1",
        "closed_precision",
        "closed_recall",
        "closed_f1",
        "pr_auc_closed",
    ]
    summary = (
        cv_metrics.groupby(group_cols)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([c for c in col if c]).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run repeated stratified CV for v2 model families.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lr", "lightgbm", "rf", "xgboost"],
        choices=["lr", "lightgbm", "rf", "xgboost"],
        help="Model families to include",
    )
    parser.add_argument(
        "--feature-bundle",
        choices=["low_only", "low_plus_medium", "full_schema_native"],
        default="low_plus_medium",
        help="Feature bundle applied to all selected models",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="CV folds")
    parser.add_argument("--n-repeats", type=int, default=3, help="Repeated CV count")
    parser.add_argument("--random-state", type=int, default=42, help="CV random seed")
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Open-class threshold used to derive class predictions from probabilities",
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
        default=Path(__file__).resolve().parents[2] / "artifacts" / "cv",
        help="Directory to write metrics artifacts",
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

    model_specs = get_model_specs(set(args.models))
    cv_metrics = run_cv(
        cv_df=cv_df,
        model_specs=model_specs,
        feature_bundle=args.feature_bundle,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
        decision_threshold=args.decision_threshold,
    )
    if cv_metrics.empty:
        raise SystemExit("No CV results produced (all model runs failed/skipped).")

    summary = summarize_metrics(cv_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv_path = args.output_dir / "metrics_cv.csv"
    summary_path = args.output_dir / "metrics_cv_summary.csv"
    cv_metrics.to_csv(cv_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved fold metrics: {cv_path}")
    print(f"Saved summary metrics: {summary_path}")
    print("\nTop rows by closed_f1_mean (compact view):")
    compact_cols = [
        "model_family",
        "mode",
        "feature_bundle",
        "accuracy_mean",
        "closed_precision_mean",
        "closed_recall_mean",
        "closed_f1_mean",
        "pr_auc_closed_mean",
    ]
    compact = (
        summary.sort_values(
            by=["closed_f1_mean", "pr_auc_closed_mean", "closed_precision_mean"],
            ascending=False,
        )[compact_cols]
        .head(10)
        .round(3)
    )
    print(compact.to_string(index=False))
    print("\n(See full metrics with std columns in metrics_cv_summary.csv)")


if __name__ == "__main__":
    main()
