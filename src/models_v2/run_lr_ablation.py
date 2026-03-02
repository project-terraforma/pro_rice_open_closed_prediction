"""Quick LR ablation runner to isolate feature-group impact.

Runs repeated stratified CV on logistic regression with shared metrics and
shared featurizer, then compares a few column-group ablations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

from shared_featurizer import SharedPlaceFeaturizer
from shared_metrics import compute_metrics


def _ablation_columns(all_cols: list[str], config: str) -> list[str]:
    def is_geo_ohe(col: str) -> bool:
        return col.startswith("ohe_geo_cluster__")

    drop = set()
    if config in {"no_raw_geo_ids", "core_no_geo_priors"}:
        drop.update({"geo_cluster_id", "geo_h3_cell_id"})

    if config == "core_no_geo_priors":
        drop.update(
            {
                "category_closure_risk",
                "spatial_cluster_closed_rate",
                "spatial_local_density",
                "same_category_neighbor_closed_rate",
                "neighbor_closed_rate",
            }
        )
        drop.update({c for c in all_cols if is_geo_ohe(c)})

    selected = [c for c in all_cols if c not in drop]
    if not selected:
        raise ValueError(f"Ablation '{config}' selected zero columns")
    return selected


def run_ablation(
    cv_df: pd.DataFrame,
    feature_bundle: str,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = ("full_bundle", "no_raw_geo_ids", "core_no_geo_priors")
    y = cv_df["open"].astype(int).values

    rows: list[dict] = []
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    for split_idx, (train_idx, val_idx) in enumerate(rskf.split(cv_df, y), start=1):
        repeat_idx = (split_idx - 1) // n_splits + 1
        fold_idx = (split_idx - 1) % n_splits + 1

        train_df = cv_df.iloc[train_idx].reset_index(drop=True)
        val_df = cv_df.iloc[val_idx].reset_index(drop=True)

        featurizer = SharedPlaceFeaturizer(
            feature_bundle=feature_bundle,
            use_source_confidence=False,
            use_interactions=True,
        )
        featurizer.fit(train_df, label_col="open")
        X_train_all = featurizer.transform(train_df)
        X_val_all = featurizer.transform(val_df)

        all_cols = list(X_train_all.columns)
        for config in configs:
            cols = _ablation_columns(all_cols, config)
            X_train = X_train_all[cols]
            X_val = X_val_all[cols]

            clf = LogisticRegression(
                class_weight="balanced",
                random_state=42,
                max_iter=1000,
            )
            clf.fit(X_train, train_df["open"].astype(int).values)

            p_open = clf.predict_proba(X_val)[:, 1]
            y_pred = (p_open >= threshold).astype(int)
            metrics = compute_metrics(
                y_true=val_df["open"].astype(int).values,
                y_pred=y_pred,
                y_score_open=p_open,
            )

            rows.append(
                {
                    "config": config,
                    "feature_bundle": feature_bundle,
                    "repeat": repeat_idx,
                    "fold": fold_idx,
                    "n_features": len(cols),
                    **metrics,
                }
            )

    fold_df = pd.DataFrame(rows)
    summary = (
        fold_df.groupby(["config", "feature_bundle"])[
            [
                "n_features",
                "accuracy",
                "closed_precision",
                "closed_recall",
                "closed_f1",
                "pr_auc_closed",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([c for c in col if c]).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return fold_df, summary


def parse_args():
    parser = argparse.ArgumentParser(description="Quick logistic-regression feature ablation runner.")
    parser.add_argument(
        "--feature-bundle",
        choices=["low_only", "low_plus_medium", "full_schema_native"],
        default="low_plus_medium",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "ablation",
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

    fold_df, summary = run_ablation(
        cv_df=cv_df,
        feature_bundle=args.feature_bundle,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
        threshold=args.decision_threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = args.output_dir / "lr_ablation_folds.csv"
    summary_path = args.output_dir / "lr_ablation_summary.csv"
    fold_df.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved fold metrics: {fold_path}")
    print(f"Saved summary metrics: {summary_path}")
    display_cols = [
        "config",
        "n_features_mean",
        "accuracy_mean",
        "closed_precision_mean",
        "closed_recall_mean",
        "closed_f1_mean",
        "pr_auc_closed_mean",
    ]
    print("\nLR ablation summary (sorted by closed_f1_mean):")
    print(summary.sort_values("closed_f1_mean", ascending=False)[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()

