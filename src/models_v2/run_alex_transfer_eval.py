"""Evaluate frozen in-domain configs on Alex's labeled city datasets."""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
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


def _load_train_val(data_dir: Path) -> pd.DataFrame:
    train_path = data_dir / "train_split.parquet"
    val_path = data_dir / "val_split.parquet"
    for path in [train_path, val_path]:
        if not path.exists():
            raise SystemExit(f"Missing required training split file: {path}")
    return pd.concat([pd.read_parquet(train_path), pd.read_parquet(val_path)], ignore_index=True)


def _point_wkb(lon: float, lat: float) -> bytes:
    return struct.pack("<BI2d", 1, 1, float(lon), float(lat))


def _prepare_alex_eval_df(raw_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    merged = raw_df.merge(labels_df, left_on="id", right_on="overture_id", how="inner")
    merged = merged[merged["fsq_label"].isin(["open", "closed"])].copy()
    merged["open"] = (merged["fsq_label"] == "open").astype(int)
    merged["geometry"] = [
        _point_wkb(lon, lat) for lon, lat in zip(merged["lon"].astype(float), merged["lat"].astype(float))
    ]
    merged["bbox"] = None
    merged["type"] = "place"
    merged["version"] = 0
    return merged


def _load_alex_eval_data(alex_assets_dir: Path, cities: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for city in cities:
        raw_path = alex_assets_dir / f"{city}_places_raw.parquet"
        labels_path = alex_assets_dir / f"{city}_places_labeled_checkpoint.parquet"
        if not raw_path.exists():
            raise SystemExit(f"Missing Alex raw parquet: {raw_path}")
        if not labels_path.exists():
            raise SystemExit(f"Missing Alex label parquet: {labels_path}")
        city_df = _prepare_alex_eval_df(pd.read_parquet(raw_path), pd.read_parquet(labels_path))
        city_df["alex_city"] = city
        frames.append(city_df)
    if not frames:
        raise SystemExit("No Alex city data loaded.")
    return pd.concat(frames, ignore_index=True)


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
    parser = argparse.ArgumentParser(description="Run frozen-config transfer evaluation on Alex's labeled city data.")
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
        help="Directory containing the repo's train_split.parquet and val_split.parquet.",
    )
    parser.add_argument(
        "--alex-assets-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "alex_assets",
        help="Directory containing Alex raw and labeled city parquet files.",
    )
    parser.add_argument(
        "--alex-cities",
        nargs="+",
        default=["sf", "nyc"],
        help="Alex city datasets to evaluate on.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "alex_transfer_eval",
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
    train_val_df = _load_train_val(args.data_dir)
    alex_eval_df = _load_alex_eval_data(args.alex_assets_dir, args.alex_cities)
    alex_city_order = [city for city in args.alex_cities if city in set(alex_eval_df["alex_city"].unique())]

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
            f"\n=== Alex Transfer Eval: {model_key} ({mode}, {gate_type}) "
            f"bundle={feature_bundle} trial={selected_trial} "
            f"k=({category_top_k},{dataset_top_k},{cluster_top_k}) th={threshold:.3f} ==="
        )
        print(
            f"Training rows={len(train_val_df):,}; "
            f"Alex labeled eval rows={len(alex_eval_df):,} across cities={alex_city_order}"
        )

        model = _build_model(model_key, mode, feature_bundle, params)
        featurizer = getattr(model, "featurizer", None)
        if featurizer is not None:
            featurizer.category_top_k = category_top_k
            featurizer.dataset_top_k = dataset_top_k
            featurizer.cluster_top_k = cluster_top_k

        train_start = time.perf_counter()
        print("Training frozen config on in-domain train+val...")
        _fit_model(model, train_val_df)
        train_elapsed = time.perf_counter() - train_start
        print(f"Training complete in {_format_seconds(train_elapsed)}")

        city_frames: list[pd.DataFrame] = []
        city_probs: list[np.ndarray] = []
        scored_rows = 0
        score_start = time.perf_counter()
        total_eval_rows = len(alex_eval_df)
        for idx, city in enumerate(alex_city_order, start=1):
            city_df = alex_eval_df[alex_eval_df["alex_city"] == city].reset_index(drop=True)
            city_rows = len(city_df)
            city_start = time.perf_counter()
            print(f"Scoring city {idx}/{len(alex_city_order)}: {city} ({city_rows:,} rows)")
            city_p_open = _safe_predict_open_proba(model, city_df)
            city_elapsed = time.perf_counter() - city_start
            scored_rows += city_rows
            elapsed = time.perf_counter() - score_start
            rows_per_sec = scored_rows / elapsed if elapsed > 0 else 0.0
            remaining_rows = total_eval_rows - scored_rows
            eta_seconds = remaining_rows / rows_per_sec if rows_per_sec > 0 else 0.0
            print(
                f"Completed {city} in {_format_seconds(city_elapsed)}; "
                f"progress={scored_rows:,}/{total_eval_rows:,} rows; "
                f"ETA remaining={_format_seconds(eta_seconds)}"
            )
            city_frames.append(city_df)
            city_probs.append(np.asarray(city_p_open))

        eval_df = pd.concat(city_frames, ignore_index=True)
        p_open = np.concatenate(city_probs) if city_probs else np.array([], dtype=float)
        y_true = eval_df["open"].astype(int).values
        y_pred = (p_open >= threshold).astype(int)
        metrics = compute_metrics(y_true=y_true, y_pred=y_pred, y_score_open=p_open)
        total_elapsed = time.perf_counter() - config_start
        print(
            f"Finished config in {_format_seconds(total_elapsed)} "
            f"(train={_format_seconds(train_elapsed)}, score={_format_seconds(time.perf_counter() - score_start)})"
        )

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
                "n_alex_eval": int(len(eval_df)),
                **metrics,
            }
        )

        per_city = eval_df.assign(
            p_open=p_open,
            y_pred_open=y_pred,
            y_true_open=y_true,
        ).groupby("alex_city", dropna=False)
        for city, city_df in per_city:
            city_metrics = compute_metrics(
                y_true=city_df["y_true_open"].values,
                y_pred=city_df["y_pred_open"].values,
                y_score_open=city_df["p_open"].values,
            )
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
                    "eval_scope": str(city),
                    "n_train_plus_val": int(len(train_val_df)),
                    "n_alex_eval": int(len(city_df)),
                    **city_metrics,
                }
            )

        for idx, row_df in eval_df.iterrows():
            pred_rows.append(
                {
                    "model_key": model_key,
                    "mode": mode,
                    "gate_type": gate_type,
                    "feature_bundle": feature_bundle,
                    "selected_trial": selected_trial,
                    "row_idx": idx,
                    "id": row_df["id"],
                    "alex_city": row_df["alex_city"],
                    "fsq_label": row_df["fsq_label"],
                    "match_status": row_df["match_status"],
                    "y_true_open": int(y_true[idx]),
                    "y_pred_open": int(y_pred[idx]),
                    "p_open": float(p_open[idx]),
                    "p_closed": float(1.0 - p_open[idx]),
                    "error_type": (
                        "tp_open" if y_true[idx] == 1 and y_pred[idx] == 1 else
                        "tn_open" if y_true[idx] == 0 and y_pred[idx] == 0 else
                        "fp_open" if y_true[idx] == 0 and y_pred[idx] == 1 else
                        "fn_open"
                    ),
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    if "eval_scope" not in metrics_df.columns:
        metrics_df["eval_scope"] = "all"
    else:
        metrics_df["eval_scope"] = metrics_df["eval_scope"].fillna("all")
    preds_df = pd.DataFrame(pred_rows)

    metrics_path = args.output_dir / "alex_transfer_metrics.csv"
    preds_path = args.output_dir / "alex_transfer_predictions.csv"
    cfg_path = args.output_dir / "alex_transfer_run_config.json"

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
                "alex_assets_dir": str(args.alex_assets_dir),
                "alex_cities": list(args.alex_cities),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
