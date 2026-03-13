"""Train frozen configs on train+val and evaluate once on the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_hpo_experiments_weighted import _build_model, _normalize_model_params
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


def _parse_feature_bundle_overrides(raw: str | None) -> dict[tuple[str, str], str]:
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise SystemExit("--feature-bundle-overrides must be a JSON object.")
    parsed: dict[tuple[str, str], str] = {}
    for key, value in data.items():
        if not isinstance(key, str) or ":" not in key:
            raise SystemExit(f"Invalid override key '{key}'. Expected 'model:mode'.")
        model_key, mode = [part.strip() for part in key.split(":", 1)]
        parsed[(model_key, mode)] = str(value).strip()
    return parsed


def _load_eval_configs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Config CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"Config CSV is empty: {path}")
    required = {
        "model_key",
        "mode",
        "gate_type",
        "feature_bundle",
        "selected_trial",
        "params_json",
        "category_top_k",
        "dataset_top_k",
        "cluster_top_k",
        "threshold",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Config CSV missing required columns: {sorted(missing)}")
    return df


def _apply_overrides(df: pd.DataFrame, overrides: dict[tuple[str, str], str]) -> pd.DataFrame:
    if not overrides:
        return df
    out = df.copy()
    for (model_key, mode), bundle_name in overrides.items():
        mask = (out["model_key"] == model_key) & (out["mode"] == mode)
        if mask.any():
            out.loc[mask, "feature_bundle"] = bundle_name
    return out


def _filter_configs(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if args.models:
        out = out[out["model_key"].isin(set(args.models))]
    if args.modes:
        out = out[out["mode"].isin(set(args.modes))]
    if args.gate_types:
        out = out[out["gate_type"].isin(set(args.gate_types))]
    if out.empty:
        raise SystemExit("No configs left after filters.")
    return out


def _load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "train_split.parquet"
    val_path = data_dir / "val_split.parquet"
    test_path = data_dir / "test_split.parquet"
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise SystemExit(f"Missing required split file: {path}")
    train_val_df = pd.concat([pd.read_parquet(train_path), pd.read_parquet(val_path)], ignore_index=True)
    test_df = pd.read_parquet(test_path)
    return train_val_df, test_df


def _infer_id_column(df: pd.DataFrame) -> str | None:
    for candidate in ["id", "record_id", "place_id"]:
        if candidate in df.columns:
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final held-out test evaluation for frozen configs.")
    parser.add_argument(
        "--config-csv",
        type=Path,
        required=True,
        help="CSV containing frozen configs with params, k values, and threshold.",
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--modes", nargs="+", default=None)
    parser.add_argument("--gate-types", nargs="+", default=None)
    parser.add_argument(
        "--feature-bundle-overrides",
        type=str,
        default=None,
        help="Optional JSON map, e.g. '{\"rf:single\":\"v2_rf_single_no_spatial_prior\"}'.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "holdout_eval",
    )
    args = parser.parse_args()
    args.feature_bundle_overrides = _parse_feature_bundle_overrides(args.feature_bundle_overrides)
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs = _load_eval_configs(args.config_csv)
    configs = _apply_overrides(configs, args.feature_bundle_overrides)
    configs = _filter_configs(configs, args)
    train_val_df, test_df = _load_splits(args.data_dir)
    id_col = _infer_id_column(test_df)

    metrics_rows: list[dict] = []
    pred_rows: list[dict] = []

    for row in configs.itertuples(index=False):
        model_key = str(row.model_key)
        mode = str(row.mode)
        gate_type = str(row.gate_type)
        feature_bundle = str(row.feature_bundle)
        selected_trial = int(row.selected_trial)
        category_top_k = int(row.category_top_k)
        dataset_top_k = int(row.dataset_top_k)
        cluster_top_k = int(row.cluster_top_k)
        threshold = float(row.threshold)
        params = _normalize_model_params(model_key, json.loads(str(row.params_json)))

        print(
            f"\n=== Holdout Eval: {model_key} ({mode}, {gate_type}) "
            f"bundle={feature_bundle} trial={selected_trial} "
            f"k=({category_top_k},{dataset_top_k},{cluster_top_k}) th={threshold:.3f} ==="
        )

        model = _build_model(model_key, mode, feature_bundle, params)
        featurizer = getattr(model, "featurizer", None)
        if featurizer is not None:
            featurizer.category_top_k = category_top_k
            featurizer.dataset_top_k = dataset_top_k
            featurizer.cluster_top_k = cluster_top_k

        _fit_model(model, train_val_df)
        p_open = _safe_predict_open_proba(model, test_df)
        y_true = test_df["open"].astype(int).values
        y_pred = (p_open >= threshold).astype(int)
        metrics = compute_metrics(y_true=y_true, y_pred=y_pred, y_score_open=p_open)

        metrics_rows.append(
            {
                "model_key": model_key,
                "mode": mode,
                "gate_type": gate_type,
                "feature_bundle": feature_bundle,
                "selected_trial": selected_trial,
                "params_json": json.dumps(params, sort_keys=True),
                "category_top_k": category_top_k,
                "dataset_top_k": dataset_top_k,
                "cluster_top_k": cluster_top_k,
                "threshold": threshold,
                "n_train_plus_val": int(len(train_val_df)),
                "n_test": int(len(test_df)),
                **metrics,
            }
        )

        for idx, (true_open, pred_open, prob_open) in enumerate(zip(y_true, y_pred, p_open)):
            pred_rows.append(
                {
                    "model_key": model_key,
                    "mode": mode,
                    "gate_type": gate_type,
                    "feature_bundle": feature_bundle,
                    "selected_trial": selected_trial,
                    "row_idx": idx,
                    "id": test_df.iloc[idx][id_col] if id_col is not None else None,
                    "y_true_open": int(true_open),
                    "y_pred_open": int(pred_open),
                    "p_open": float(prob_open),
                    "p_closed": float(1.0 - prob_open),
                    "error_type": (
                        "tp_open" if true_open == 1 and pred_open == 1 else
                        "tn_open" if true_open == 0 and pred_open == 0 else
                        "fp_open" if true_open == 0 and pred_open == 1 else
                        "fn_open"
                    ),
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    preds_df = pd.DataFrame(pred_rows)
    metrics_path = args.output_dir / "holdout_test_metrics.csv"
    preds_path = args.output_dir / "holdout_test_predictions.csv"
    cfg_path = args.output_dir / "holdout_test_run_config.json"

    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_csv(preds_path, index=False)
    cfg_path.write_text(
        json.dumps(
            {
                "config_csv": str(args.config_csv),
                "filters": {
                    "models": args.models,
                    "modes": args.modes,
                    "gate_types": args.gate_types,
                },
                "feature_bundle_overrides": {
                    f"{model_key}:{mode}": bundle_name
                    for (model_key, mode), bundle_name in args.feature_bundle_overrides.items()
                },
                "data_dir": str(args.data_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved holdout metrics: {metrics_path}")
    print(f"Saved holdout predictions: {preds_path}")
    print(f"Saved run config: {cfg_path}")


if __name__ == "__main__":
    main()
