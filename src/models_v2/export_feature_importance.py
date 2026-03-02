"""Export per-model feature importances to CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _build_model(model_key: str, mode: str, feature_bundle: str):
    if model_key == "lr":
        from logistic_regression_v2 import UnifiedLogisticRegression

        return UnifiedLogisticRegression(mode=mode, feature_bundle=feature_bundle)
    if model_key == "lightgbm":
        from lightgbm_model_v2 import LightGBMModel

        return LightGBMModel(mode=mode, feature_bundle=feature_bundle, use_source_confidence=False)
    if model_key == "xgboost":
        from xgboost_model_v2 import XGBoostModel

        return XGBoostModel(mode=mode, feature_bundle=feature_bundle)
    if model_key == "rf":
        from random_forest_model_v2 import RandomForestModel

        return RandomForestModel(feature_bundle=feature_bundle)
    raise ValueError(f"Unknown model key: {model_key}")


def _normalize_importance(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    out = df.copy()
    if model_key == "lr":
        base_col = "abs_coefficient"
    else:
        base_col = "importance"

    total = float(out[base_col].sum())
    out["importance_norm"] = out[base_col] / total if total > 0 else 0.0
    out["rank"] = range(1, len(out) + 1)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Export feature importance tables for v2 models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lr", "lightgbm", "rf", "xgboost"],
        choices=["lr", "lightgbm", "rf", "xgboost"],
        help="Model families to run",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["single"],
        choices=["single", "two-stage"],
        help="Model modes (rf always uses single)",
    )
    parser.add_argument(
        "--feature-bundle",
        choices=["low_only", "low_plus_medium", "full_schema_native"],
        default="low_plus_medium",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data",
    )
    parser.add_argument(
        "--use-train-plus-val",
        action="store_true",
        help="Train on train+val split instead of train split only",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "feature_importance",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    train_path = args.data_dir / "train_split.parquet"
    val_path = args.data_dir / "val_split.parquet"
    if not train_path.exists():
        raise SystemExit(f"Missing {train_path}")

    train_df = pd.read_parquet(train_path)
    if args.use_train_plus_val:
        if not val_path.exists():
            raise SystemExit(f"Missing {val_path} for --use-train-plus-val")
        val_df = pd.read_parquet(val_path)
        fit_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    else:
        fit_df = train_df

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for model_key in args.models:
        modes = ["single"] if model_key == "rf" else args.modes
        for mode in modes:
            model = _build_model(model_key, mode, args.feature_bundle)
            model.fit(fit_df, val_df=None)
            imp = model.get_feature_importances()
            imp = _normalize_importance(imp, model_key=model_key)

            out_name = f"{model_key}_{mode}_{args.feature_bundle}_importance.csv"
            out_path = args.output_dir / out_name
            imp.to_csv(out_path, index=False)

            top = imp.head(10)
            summary_rows.append(
                {
                    "model_key": model_key,
                    "mode": mode,
                    "feature_bundle": args.feature_bundle,
                    "n_features": len(imp),
                    "top_feature": top.iloc[0]["feature"] if len(top) else "",
                    "top_feature_importance_norm": float(top.iloc[0]["importance_norm"]) if len(top) else 0.0,
                    "output_path": str(out_path),
                }
            )
            print(f"Saved: {out_path}")
            print(top[["rank", "feature", "importance_norm"]].to_string(index=False))
            print()

    summary_df = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "feature_importance_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

