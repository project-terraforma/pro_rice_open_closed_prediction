"""Generate out-of-fold predictions for frozen configs on project train+val."""

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


def _load_train_val(data_dir: Path) -> pd.DataFrame:
    train_path = data_dir / "train_split.parquet"
    val_path = data_dir / "val_split.parquet"
    return pd.concat([pd.read_parquet(train_path), pd.read_parquet(val_path)], ignore_index=True)


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OOF predictions on project train+val for frozen configs.")
    parser.add_argument("--config-csv", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--modes", nargs="+", default=None)
    parser.add_argument("--gate-types", nargs="+", default=None)
    parser.add_argument("--feature-bundle-overrides", type=str, default=None)
    parser.add_argument("--cv-n-splits", type=int, default=5)
    parser.add_argument("--cv-n-repeats", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "projectc_oof_eval",
    )
    parser.add_argument("--show-model-logs", action="store_true")
    args = parser.parse_args()
    args.feature_bundle_overrides = _parse_feature_bundle_overrides(args.feature_bundle_overrides)
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs = _load_eval_configs(args.config_csv)
    configs = _apply_overrides(configs, args.feature_bundle_overrides)
    configs = _filter_configs(configs, args)
    cv_df = _load_train_val(args.data_dir).reset_index(drop=True)
    y_all = cv_df["open"].astype(int).values

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
            f"\n=== Project OOF Eval: {model_key} ({mode}, {gate_type}) "
            f"bundle={feature_bundle} trial={selected_trial} "
            f"k=({category_top_k},{dataset_top_k},{cluster_top_k}) th={threshold:.3f} ==="
        )
        splitter = RepeatedStratifiedKFold(
            n_splits=args.cv_n_splits,
            n_repeats=args.cv_n_repeats,
            random_state=args.random_state,
        )
        p_open_sum = np.zeros(len(cv_df), dtype=float)
        p_open_count = np.zeros(len(cv_df), dtype=int)
        fold_metrics: list[dict] = []
        total_folds = args.cv_n_splits * args.cv_n_repeats
        loop_start = time.perf_counter()

        for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(cv_df, y_all), start=1):
            tr_df = cv_df.iloc[tr_idx].reset_index(drop=True)
            va_df = cv_df.iloc[va_idx].reset_index(drop=True)
            model = _build_model(model_key, mode, feature_bundle, params)
            featurizer = getattr(model, "featurizer", None)
            if featurizer is not None:
                featurizer.category_top_k = category_top_k
                featurizer.dataset_top_k = dataset_top_k
                featurizer.cluster_top_k = cluster_top_k

            print(f"Fold {fold_idx}/{total_folds}: train={len(tr_df):,}, val={len(va_df):,}")
            fold_start = time.perf_counter()
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
            p_open_sum[va_idx] += p_open
            p_open_count[va_idx] += 1
            elapsed = time.perf_counter() - loop_start
            eta = (elapsed / fold_idx) * (total_folds - fold_idx) if fold_idx else 0.0
            print(
                f"Completed fold {fold_idx}/{total_folds} in {_format_seconds(time.perf_counter() - fold_start)}; "
                f"ETA remaining={_format_seconds(eta)}"
            )

        p_open_oof = p_open_sum / p_open_count
        y_pred_oof = (p_open_oof >= threshold).astype(int)
        point_metrics = compute_metrics(y_true=y_all, y_pred=y_pred_oof, y_score_open=p_open_oof)
        fold_df = pd.DataFrame(fold_metrics)
        metrics_row = {
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
            "n_oof_eval": int(len(cv_df)),
            **point_metrics,
        }
        for col in fold_df.columns:
            metrics_row[f"{col}_mean"] = float(fold_df[col].mean())
            metrics_row[f"{col}_std"] = float(fold_df[col].std(ddof=0))
        metrics_rows.append(metrics_row)

        id_col = "id" if "id" in cv_df.columns else None
        for idx, row_df in cv_df.iterrows():
            pred_rows.append(
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
                    "cv_n_splits": args.cv_n_splits,
                    "cv_n_repeats": args.cv_n_repeats,
                    "row_idx": idx,
                    "id": row_df[id_col] if id_col else None,
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
        print(f"Finished config in {_format_seconds(time.perf_counter() - config_start)}")

    pd.DataFrame(metrics_rows).to_csv(args.output_dir / "projectc_oof_metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(args.output_dir / "projectc_oof_predictions.csv", index=False)
    (args.output_dir / "projectc_oof_run_config.json").write_text(
        json.dumps(
            {
                "config_csv": str(args.config_csv),
                "filters": {"models": args.models, "modes": args.modes, "gate_types": args.gate_types},
                "feature_bundle_overrides": {
                    f"{model_key}:{mode}": bundle_name
                    for (model_key, mode), bundle_name in args.feature_bundle_overrides.items()
                },
                "cv_n_splits": args.cv_n_splits,
                "cv_n_repeats": args.cv_n_repeats,
                "random_state": args.random_state,
                "data_dir": str(args.data_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
