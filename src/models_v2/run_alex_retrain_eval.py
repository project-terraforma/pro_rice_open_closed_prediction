"""Retrain frozen configs on Alex's labeled data and evaluate with repeated CV."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from run_alex_transfer_eval import (
    _apply_overrides,
    _filter_configs,
    _format_seconds,
    _load_alex_eval_data,
    _load_eval_configs,
    _parse_feature_bundle_overrides,
)
from run_hpo_experiments_weighted import _build_model, _normalize_model_params
from shared_metrics import compute_metrics


def _safe_predict_open_proba(model, df: pd.DataFrame) -> np.ndarray:
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


def _apply_featurizer_k(model, category_top_k: int, dataset_top_k: int, cluster_top_k: int) -> None:
    featurizer = getattr(model, "featurizer", None)
    if featurizer is None:
        raise ValueError(f"Model does not expose a featurizer: {type(model)}")
    featurizer.category_top_k = int(category_top_k)
    featurizer.dataset_top_k = int(dataset_top_k)
    featurizer.cluster_top_k = int(cluster_top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain frozen configs on Alex's labeled city data with repeated CV.")
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
        "--alex-assets-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "alex_assets",
    )
    parser.add_argument("--alex-cities", nargs="+", default=["sf", "nyc"])
    parser.add_argument("--cv-n-splits", type=int, default=5)
    parser.add_argument("--cv-n-repeats", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "alex_retrain_eval",
    )
    parser.add_argument("--show-model-logs", action="store_true")
    args = parser.parse_args()
    args.feature_bundle_overrides = _parse_feature_bundle_overrides(args.feature_bundle_overrides)
    return args


def _append_scope_metrics(
    metrics_rows: list[dict],
    *,
    eval_scope: str,
    config_meta: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_open: np.ndarray,
    fold_metrics_df: pd.DataFrame,
) -> None:
    point_metrics = compute_metrics(y_true=y_true, y_pred=y_pred, y_score_open=p_open)
    row = {
        **config_meta,
        "eval_scope": eval_scope,
        "n_alex_eval": int(len(y_true)),
        **point_metrics,
    }
    for col in fold_metrics_df.columns:
        row[f"{col}_mean"] = float(fold_metrics_df[col].mean())
        row[f"{col}_std"] = float(fold_metrics_df[col].std(ddof=0))
    metrics_rows.append(row)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs = _load_eval_configs(args.config_csv)
    configs = _apply_overrides(configs, args.feature_bundle_overrides)
    configs = _filter_configs(configs, args)
    alex_df = _load_alex_eval_data(args.alex_assets_dir, args.alex_cities).reset_index(drop=True)
    alex_city_order = [city for city in args.alex_cities if city in set(alex_df["alex_city"].unique())]

    metrics_rows: list[dict] = []
    pred_rows: list[dict] = []

    for row in configs.itertuples(index=False):
        config_start = time.perf_counter()
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
            f"\n=== Alex Retrain Eval: {model_key} ({mode}, {gate_type}) "
            f"bundle={feature_bundle} trial={selected_trial} "
            f"k=({category_top_k},{dataset_top_k},{cluster_top_k}) th={threshold:.3f} ==="
        )
        print(
            f"Alex labeled rows={len(alex_df):,} across cities={alex_city_order}; "
            f"CV={args.cv_n_splits} folds x {args.cv_n_repeats} repeats"
        )

        splitter = RepeatedStratifiedKFold(
            n_splits=args.cv_n_splits,
            n_repeats=args.cv_n_repeats,
            random_state=args.random_state,
        )
        y_all = alex_df["open"].astype(int).values
        p_open_sum = np.zeros(len(alex_df), dtype=float)
        p_open_count = np.zeros(len(alex_df), dtype=int)
        fold_metrics: list[dict] = []
        fold_city_metrics: dict[str, list[dict]] = {city: [] for city in alex_city_order}

        total_folds = args.cv_n_splits * args.cv_n_repeats
        fold_start_all = time.perf_counter()

        for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(alex_df, y_all), start=1):
            tr_df = alex_df.iloc[tr_idx].reset_index(drop=True)
            va_df = alex_df.iloc[va_idx].reset_index(drop=True)

            model = _build_model(model_key, mode, feature_bundle, params)
            _apply_featurizer_k(
                model=model,
                category_top_k=category_top_k,
                dataset_top_k=dataset_top_k,
                cluster_top_k=cluster_top_k,
            )

            fold_start = time.perf_counter()
            print(
                f"Fold {fold_idx}/{total_folds}: train={len(tr_df):,} rows, val={len(va_df):,} rows"
            )
            if args.show_model_logs:
                _fit_model(model, tr_df)
                p_open = _safe_predict_open_proba(model, va_df)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        _fit_model(model, tr_df)
                        p_open = _safe_predict_open_proba(model, va_df)

            y_true = va_df["open"].astype(int).values
            y_pred = (p_open >= threshold).astype(int)
            fold_metrics.append(compute_metrics(y_true=y_true, y_pred=y_pred, y_score_open=p_open))
            for city in alex_city_order:
                city_mask_local = va_df["alex_city"].eq(city).values
                if not np.any(city_mask_local):
                    continue
                fold_city_metrics[city].append(
                    compute_metrics(
                        y_true=y_true[city_mask_local],
                        y_pred=y_pred[city_mask_local],
                        y_score_open=p_open[city_mask_local],
                    )
                )

            p_open_sum[va_idx] += p_open
            p_open_count[va_idx] += 1

            fold_elapsed = time.perf_counter() - fold_start
            elapsed = time.perf_counter() - fold_start_all
            avg_fold = elapsed / fold_idx if fold_idx > 0 else 0.0
            remaining = avg_fold * (total_folds - fold_idx)
            print(
                f"Completed fold {fold_idx}/{total_folds} in {_format_seconds(fold_elapsed)}; "
                f"ETA remaining={_format_seconds(remaining)}"
            )

        if np.any(p_open_count == 0):
            raise RuntimeError("Some Alex rows never received an out-of-fold prediction.")

        p_open_oof = p_open_sum / p_open_count
        y_pred_oof = (p_open_oof >= threshold).astype(int)
        fold_metrics_df = pd.DataFrame(fold_metrics)
        config_meta = {
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
            "cv_n_splits": args.cv_n_splits,
            "cv_n_repeats": args.cv_n_repeats,
        }

        _append_scope_metrics(
            metrics_rows,
            eval_scope="all",
            config_meta=config_meta,
            y_true=y_all,
            y_pred=y_pred_oof,
            p_open=p_open_oof,
            fold_metrics_df=fold_metrics_df,
        )

        for city in alex_city_order:
            city_mask = alex_df["alex_city"].eq(city).values
            city_fold_df = pd.DataFrame(fold_city_metrics[city])
            _append_scope_metrics(
                metrics_rows,
                eval_scope=city,
                config_meta=config_meta,
                y_true=y_all[city_mask],
                y_pred=y_pred_oof[city_mask],
                p_open=p_open_oof[city_mask],
                fold_metrics_df=city_fold_df,
            )

        for idx, row_df in alex_df.iterrows():
            pred_rows.append(
                {
                    **config_meta,
                    "row_idx": idx,
                    "id": row_df["id"],
                    "alex_city": row_df["alex_city"],
                    "fsq_label": row_df["fsq_label"],
                    "match_status": row_df["match_status"],
                    "oof_prediction_count": int(p_open_count[idx]),
                    "y_true_open": int(y_all[idx]),
                    "y_pred_open": int(y_pred_oof[idx]),
                    "p_open_oof": float(p_open_oof[idx]),
                    "p_closed_oof": float(1.0 - p_open_oof[idx]),
                    "error_type": (
                        "tp_open" if y_all[idx] == 1 and y_pred_oof[idx] == 1 else
                        "tn_open" if y_all[idx] == 0 and y_pred_oof[idx] == 0 else
                        "fp_open" if y_all[idx] == 0 and y_pred_oof[idx] == 1 else
                        "fn_open"
                    ),
                }
            )

        total_elapsed = time.perf_counter() - config_start
        print(f"Finished config in {_format_seconds(total_elapsed)}")

    metrics_df = pd.DataFrame(metrics_rows)
    preds_df = pd.DataFrame(pred_rows)

    metrics_path = args.output_dir / "alex_retrain_metrics.csv"
    preds_path = args.output_dir / "alex_retrain_oof_predictions.csv"
    cfg_path = args.output_dir / "alex_retrain_run_config.json"

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
                "alex_assets_dir": str(args.alex_assets_dir),
                "alex_cities": list(args.alex_cities),
                "cv_n_splits": args.cv_n_splits,
                "cv_n_repeats": args.cv_n_repeats,
                "random_state": args.random_state,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
