"""Model-agnostic phased runner: coarse-k, narrow-k, threshold sweep, and confirm."""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from run_hpo_experiments_weighted import _build_model, _normalize_model_params
from shared_metrics import compute_metrics

BASE_GROUP_COLS = ["model_key", "mode", "gate_type", "feature_bundle", "selected_trial", "params_json"]


def _parse_int_grid(raw: str) -> list[int]:
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    out = sorted({int(v) for v in vals})
    if not out:
        raise ValueError(f"Empty integer grid: '{raw}'")
    return out


def _parse_feature_bundle_overrides(raw: str | None) -> dict[tuple[str, str], str]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON for --feature-bundle-overrides: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("--feature-bundle-overrides must be a JSON object like {'rf:single':'v2_rf_single_no_spatial_prior'}")

    parsed: dict[tuple[str, str], str] = {}
    for key, value in data.items():
        if not isinstance(key, str) or ":" not in key:
            raise SystemExit(f"Invalid override key '{key}'. Expected format 'model:mode'.")
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(f"Invalid override bundle for '{key}': {value!r}")
        model_key, mode = [part.strip() for part in key.split(":", 1)]
        if not model_key or not mode:
            raise SystemExit(f"Invalid override key '{key}'. Expected format 'model:mode'.")
        parsed[(model_key, mode)] = value.strip()
    return parsed


def _apply_feature_bundle_overrides(df: pd.DataFrame, overrides: dict[tuple[str, str], str]) -> pd.DataFrame:
    if not overrides or df.empty:
        return df
    out = df.copy()
    for (model_key, mode), bundle_name in overrides.items():
        mask = (out["model_key"] == model_key) & (out["mode"] == mode)
        if mask.any():
            out.loc[mask, "feature_bundle"] = bundle_name
    return out


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
        raise ValueError(f"Model does not expose a featurizer for k-sweep: {type(model)}")
    featurizer.category_top_k = int(category_top_k)
    featurizer.dataset_top_k = int(dataset_top_k)
    featurizer.cluster_top_k = int(cluster_top_k)


def _evaluate_config_folds(
    cv_df: pd.DataFrame,
    model_key: str,
    mode: str,
    feature_bundle: str,
    model_params: dict,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    category_top_k: int,
    dataset_top_k: int,
    cluster_top_k: int,
    show_model_logs: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    y = cv_df["open"].astype(int).values
    fold_scores: list[tuple[np.ndarray, np.ndarray]] = []

    for tr_idx, va_idx in splitter.split(cv_df, y):
        tr_df = cv_df.iloc[tr_idx].reset_index(drop=True)
        va_df = cv_df.iloc[va_idx].reset_index(drop=True)
        params = _normalize_model_params(model_key, model_params)
        model = _build_model(model_key, mode, feature_bundle, params)
        _apply_featurizer_k(
            model=model,
            category_top_k=category_top_k,
            dataset_top_k=dataset_top_k,
            cluster_top_k=cluster_top_k,
        )
        if show_model_logs:
            _fit_model(model, tr_df)
            p_open = _safe_predict_open_proba(model, va_df)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    _fit_model(model, tr_df)
                    p_open = _safe_predict_open_proba(model, va_df)
        y_true = va_df["open"].astype(int).values
        fold_scores.append((y_true, np.asarray(p_open)))
    return fold_scores


def _metrics_from_fold_scores(fold_scores: list[tuple[np.ndarray, np.ndarray]], threshold: float) -> dict:
    fold_metrics = []
    for y_true, p_open in fold_scores:
        y_pred = (p_open >= threshold).astype(int)
        fold_metrics.append(compute_metrics(y_true=y_true, y_pred=y_pred, y_score_open=p_open))
    fm = pd.DataFrame(fold_metrics)
    out = {"threshold": float(threshold)}
    for col in fm.columns:
        out[f"{col}_mean"] = float(fm[col].mean())
        out[f"{col}_std"] = float(fm[col].std(ddof=0))
    return out


def _threshold_grid(start: float, end: float, step: float) -> list[float]:
    vals = np.arange(start, end + (step * 0.5), step)
    vals = np.clip(vals, 0.0, 1.0)
    return [float(round(v, 6)) for v in vals]


def _select_for_gate(df: pd.DataFrame, gate_type: str, args: argparse.Namespace) -> tuple[pd.Series, str]:
    gate_type = gate_type.strip().lower()
    if gate_type == "production":
        passing = df[
            (df["accuracy_mean"] >= args.prod_accuracy_floor)
            & (df["closed_precision_mean"] >= args.prod_closed_precision_floor)
            & (df["closed_recall_mean"] >= args.prod_closed_recall_floor)
        ]
        if not passing.empty:
            winner = passing.sort_values(
                by=["closed_precision_mean", "closed_f1_mean", "pr_auc_closed_mean", "accuracy_mean"],
                ascending=False,
            ).iloc[0]
            return winner, "Production gate-pass candidate selected."
        winner = df.sort_values(
            by=["closed_precision_mean", "closed_f1_mean", "pr_auc_closed_mean", "accuracy_mean"],
            ascending=False,
        ).iloc[0]
        return winner, "No production gate-pass candidate; fallback to precision/F1 best-effort."
    if gate_type == "diagnostic":
        passing = df[
            (df["accuracy_mean"] >= args.diag_accuracy_floor)
            & (df["closed_precision_mean"] >= args.diag_closed_precision_floor)
        ]
        if not passing.empty:
            winner = passing.sort_values(
                by=["closed_f1_mean", "pr_auc_closed_mean", "closed_precision_mean", "accuracy_mean"],
                ascending=False,
            ).iloc[0]
            return winner, "Diagnostic gate-pass candidate selected."
        winner = df.sort_values(
            by=["closed_f1_mean", "pr_auc_closed_mean", "closed_precision_mean", "accuracy_mean"],
            ascending=False,
        ).iloc[0]
        return winner, "No diagnostic gate-pass candidate; fallback to best closed-F1."
    raise ValueError(f"Unsupported gate_type: {gate_type}")


def _filter_selected_configs(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.models:
        allowed = {m.strip() for m in args.models}
        df = df[df["model_key"].isin(allowed)]
    if args.modes:
        allowed = {m.strip() for m in args.modes}
        df = df[df["mode"].isin(allowed)]
    if args.gate_types:
        allowed = {g.strip() for g in args.gate_types}
        df = df[df["gate_type"].isin(allowed)]
    return df


def _evaluate_k_combos(
    selected_df: pd.DataFrame,
    k_combos: list[tuple[int, int, int]],
    eval_threshold: float,
    cv_df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict] = []
    run_start = time.perf_counter()
    total_blocks = len(selected_df) * max(len(k_combos), 1)
    done_blocks = 0
    for cfg in selected_df.itertuples(index=False):
        model_key = str(cfg.model_key)
        mode = str(cfg.mode)
        gate_type = str(cfg.gate_type)
        feature_bundle = str(cfg.feature_bundle)
        params = json.loads(str(cfg.params_json))
        selected_trial = int(cfg.selected_trial) if hasattr(cfg, "selected_trial") else -1
        print(f"\n=== K sweep: {model_key} ({mode}, {gate_type}) trial={selected_trial} ===")
        for cat_k, ds_k, cl_k in k_combos:
            t0 = time.perf_counter()
            fold_scores = _evaluate_config_folds(
                cv_df=cv_df,
                model_key=model_key,
                mode=mode,
                feature_bundle=feature_bundle,
                model_params=params,
                n_splits=args.cv_n_splits,
                n_repeats=args.cv_n_repeats,
                random_state=args.random_state,
                category_top_k=cat_k,
                dataset_top_k=ds_k,
                cluster_top_k=cl_k,
                show_model_logs=args.show_model_logs,
            )
            metrics = _metrics_from_fold_scores(fold_scores, eval_threshold)
            rows.append(
                {
                    "model_key": model_key,
                    "mode": mode,
                    "gate_type": gate_type,
                    "feature_bundle": feature_bundle,
                    "selected_trial": selected_trial,
                    "params_json": json.dumps(params, sort_keys=True),
                    "category_top_k": int(cat_k),
                    "dataset_top_k": int(ds_k),
                    "cluster_top_k": int(cl_k),
                    **metrics,
                }
            )
            done_blocks += 1
            elapsed = time.perf_counter() - run_start
            avg = elapsed / max(done_blocks, 1)
            eta_min = (avg * max(total_blocks - done_blocks, 0)) / 60.0
            print(
                f"  k=({cat_k},{ds_k},{cl_k}) done in {time.perf_counter() - t0:.1f}s "
                f"[{done_blocks}/{total_blocks}, ETA {eta_min:.1f}m]"
            )
    return pd.DataFrame(rows)


def _rank_top_n_by_gate(metrics_df: pd.DataFrame, top_n: int, args: argparse.Namespace) -> pd.DataFrame:
    out_rows = []
    for _, grp in metrics_df.groupby(BASE_GROUP_COLS, sort=False):
        gate_type = str(grp.iloc[0]["gate_type"])
        gate_rank = grp.copy()
        if gate_type == "production":
            gate_rank["passes_gate"] = (
                (gate_rank["accuracy_mean"] >= args.prod_accuracy_floor)
                & (gate_rank["closed_precision_mean"] >= args.prod_closed_precision_floor)
                & (gate_rank["closed_recall_mean"] >= args.prod_closed_recall_floor)
            )
            gate_rank = gate_rank.sort_values(
                by=["passes_gate", "closed_precision_mean", "closed_f1_mean", "pr_auc_closed_mean", "accuracy_mean"],
                ascending=[False, False, False, False, False],
            )
        else:
            gate_rank["passes_gate"] = (
                (gate_rank["accuracy_mean"] >= args.diag_accuracy_floor)
                & (gate_rank["closed_precision_mean"] >= args.diag_closed_precision_floor)
            )
            gate_rank = gate_rank.sort_values(
                by=["passes_gate", "closed_f1_mean", "pr_auc_closed_mean", "closed_precision_mean", "accuracy_mean"],
                ascending=[False, False, False, False, False],
            )
        out_rows.extend(gate_rank.head(max(top_n, 1)).to_dict(orient="records"))
    return pd.DataFrame(out_rows)


def _build_narrow_k_combos(anchors_df: pd.DataFrame, args: argparse.Namespace) -> list[tuple[int, int, int]]:
    # Deltas can include negatives (e.g., -5,0,5).
    cat_d = [int(v) for v in args.k_narrow_category_deltas.split(",") if v.strip()]
    ds_d = [int(v) for v in args.k_narrow_dataset_deltas.split(",") if v.strip()]
    cl_d = [int(v) for v in args.k_narrow_cluster_deltas.split(",") if v.strip()]
    combos: set[tuple[int, int, int]] = set()
    for row in anchors_df.itertuples(index=False):
        for dc, dd, dl in itertools.product(cat_d, ds_d, cl_d):
            c = max(args.k_min_value, int(row.category_top_k) + int(dc))
            d = max(args.k_min_value, int(row.dataset_top_k) + int(dd))
            l = max(args.k_min_value, int(row.cluster_top_k) + int(dl))
            combos.add((c, d, l))
    return sorted(combos)


def _select_final_best(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for _, grp in df.groupby(BASE_GROUP_COLS, sort=False):
        winner, rationale = _select_for_gate(grp, str(grp.iloc[0]["gate_type"]), args)
        row = winner.to_dict()
        row["selection_rationale"] = rationale
        rows.append(row)
    return pd.DataFrame(rows)


def _run_confirm_phase(cv_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    confirm_input = args.confirm_input_csv or (args.output_dir / "threshold_final_best.csv")
    if not confirm_input.exists():
        raise SystemExit(f"Confirm phase input not found: {confirm_input}")

    finalists = pd.read_csv(confirm_input)
    finalists = _filter_selected_configs(finalists, args)
    finalists = _apply_feature_bundle_overrides(finalists, args.feature_bundle_overrides)
    if finalists.empty:
        raise SystemExit("Confirm phase finalist set is empty after filters.")

    required_cols = set(BASE_GROUP_COLS) | {
        "category_top_k",
        "dataset_top_k",
        "cluster_top_k",
        "threshold",
    }
    missing = required_cols - set(finalists.columns)
    if missing:
        raise SystemExit(f"Confirm phase input missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    run_start = time.perf_counter()
    total = len(finalists)
    for idx, row in enumerate(finalists.itertuples(index=False), start=1):
        model_key = str(row.model_key)
        mode = str(row.mode)
        gate_type = str(row.gate_type)
        feature_bundle = str(row.feature_bundle)
        params = json.loads(str(row.params_json))
        selected_trial = int(row.selected_trial)
        cat_k = int(row.category_top_k)
        ds_k = int(row.dataset_top_k)
        cl_k = int(row.cluster_top_k)
        threshold = float(row.threshold)

        print(
            f"\n=== Confirm: {model_key} ({mode}, {gate_type}) "
            f"trial={selected_trial} k=({cat_k},{ds_k},{cl_k}) th={threshold:.3f} ==="
        )
        t0 = time.perf_counter()
        fold_scores = _evaluate_config_folds(
            cv_df=cv_df,
            model_key=model_key,
            mode=mode,
            feature_bundle=feature_bundle,
            model_params=params,
            n_splits=args.confirm_n_splits,
            n_repeats=args.confirm_n_repeats,
            random_state=args.confirm_random_state,
            category_top_k=cat_k,
            dataset_top_k=ds_k,
            cluster_top_k=cl_k,
            show_model_logs=args.show_model_logs,
        )
        metrics = _metrics_from_fold_scores(fold_scores, threshold)
        rows.append(
            {
                "model_key": model_key,
                "mode": mode,
                "gate_type": gate_type,
                "feature_bundle": feature_bundle,
                "selected_trial": selected_trial,
                "params_json": json.dumps(params, sort_keys=True),
                "category_top_k": cat_k,
                "dataset_top_k": ds_k,
                "cluster_top_k": cl_k,
                "threshold": threshold,
                "confirm_n_splits": int(args.confirm_n_splits),
                "confirm_n_repeats": int(args.confirm_n_repeats),
                "confirm_random_state": int(args.confirm_random_state),
                **metrics,
            }
        )
        elapsed = time.perf_counter() - run_start
        avg = elapsed / max(idx, 1)
        eta_min = (avg * max(total - idx, 0)) / 60.0
        print(f"  done in {time.perf_counter() - t0:.1f}s [{idx}/{total}, ETA {eta_min:.1f}m]")

    return pd.DataFrame(rows)


def _load_selected_configs(args: argparse.Namespace) -> pd.DataFrame:
    if not args.selected_configs_csv.exists():
        raise SystemExit(f"Selected config CSV not found: {args.selected_configs_csv}")
    selected_df = pd.read_csv(args.selected_configs_csv)
    if selected_df.empty:
        raise SystemExit("Selected config CSV is empty.")
    required_cols = {"model_key", "mode", "gate_type", "feature_bundle", "params_json"}
    missing_cols = required_cols - set(selected_df.columns)
    if missing_cols:
        raise SystemExit(f"Selected config CSV missing required columns: {sorted(missing_cols)}")
    selected_df = _filter_selected_configs(selected_df, args)
    selected_df = _apply_feature_bundle_overrides(selected_df, args.feature_bundle_overrides)
    if selected_df.empty:
        raise SystemExit("No configs left after filters.")
    return selected_df


def _load_cv_df(data_dir: Path) -> pd.DataFrame:
    train_path = data_dir / "train_split.parquet"
    val_path = data_dir / "val_split.parquet"
    if not train_path.exists() or not val_path.exists():
        raise SystemExit(f"Missing split files in {data_dir}; expected train_split.parquet and val_split.parquet")
    return pd.concat([pd.read_parquet(train_path), pd.read_parquet(val_path)], axis=0, ignore_index=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phased sweep for k controls and decision thresholds.")
    parser.add_argument(
        "--phase",
        choices=["k_coarse", "k_narrow", "threshold", "confirm", "full"],
        default="full",
    )
    parser.add_argument(
        "--selected-configs-csv",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "artifacts"
        / "hpo_optuna_lr_micro_pass1"
        / "hpo_selected_trials_dualgate.csv",
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

    parser.add_argument("--k-coarse-category-grid", type=str, default="15,25,35,50")
    parser.add_argument("--k-coarse-dataset-grid", type=str, default="10,20,30,40")
    parser.add_argument("--k-coarse-cluster-grid", type=str, default="20,30,45,60")
    parser.add_argument("--k-coarse-top-n", type=int, default=3)
    parser.add_argument("--k-eval-threshold", type=float, default=0.5)

    parser.add_argument("--k-narrow-input-csv", type=Path, default=None)
    parser.add_argument("--k-narrow-category-deltas", type=str, default="-5,0,5")
    parser.add_argument("--k-narrow-dataset-deltas", type=str, default="-5,0,5")
    parser.add_argument("--k-narrow-cluster-deltas", type=str, default="-10,0,10")
    parser.add_argument("--k-narrow-top-n", type=int, default=2)
    parser.add_argument("--k-min-value", type=int, default=1)

    parser.add_argument("--threshold-input-csv", type=Path, default=None)
    parser.add_argument("--threshold-top-k", type=int, default=2)
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--confirm-input-csv", type=Path, default=None)
    parser.add_argument("--confirm-n-splits", type=int, default=5)
    parser.add_argument("--confirm-n-repeats", type=int, default=10)
    parser.add_argument("--confirm-random-state", type=int, default=142)

    parser.add_argument("--cv-n-splits", type=int, default=5)
    parser.add_argument("--cv-n-repeats", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data")

    parser.add_argument("--prod-accuracy-floor", type=float, default=0.90)
    parser.add_argument("--prod-closed-precision-floor", type=float, default=0.70)
    parser.add_argument("--prod-closed-recall-floor", type=float, default=0.05)
    parser.add_argument("--diag-accuracy-floor", type=float, default=0.84)
    parser.add_argument("--diag-closed-precision-floor", type=float, default=0.20)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "k_threshold_sweep",
    )
    parser.add_argument("--show-model-logs", action="store_true")
    args = parser.parse_args()
    args.feature_bundle_overrides = _parse_feature_bundle_overrides(args.feature_bundle_overrides)
    return args


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv_df = _load_cv_df(args.data_dir)
    selected_df = None
    if args.phase in {"k_coarse", "k_narrow", "threshold", "full"}:
        selected_df = _load_selected_configs(args)

    run_cfg = {
        "phase": args.phase,
        "selected_configs_csv": str(args.selected_configs_csv),
        "filters": {"models": args.models, "modes": args.modes, "gate_types": args.gate_types},
        "feature_bundle_overrides": {
            f"{model_key}:{mode}": bundle_name
            for (model_key, mode), bundle_name in args.feature_bundle_overrides.items()
        },
        "k_coarse_grid": {
            "category_top_k": _parse_int_grid(args.k_coarse_category_grid),
            "dataset_top_k": _parse_int_grid(args.k_coarse_dataset_grid),
            "cluster_top_k": _parse_int_grid(args.k_coarse_cluster_grid),
            "top_n": args.k_coarse_top_n,
            "eval_threshold": args.k_eval_threshold,
        },
        "k_narrow": {
            "input_csv": str(args.k_narrow_input_csv) if args.k_narrow_input_csv else None,
            "category_deltas": args.k_narrow_category_deltas,
            "dataset_deltas": args.k_narrow_dataset_deltas,
            "cluster_deltas": args.k_narrow_cluster_deltas,
            "top_n": args.k_narrow_top_n,
            "min_value": args.k_min_value,
        },
        "threshold": {
            "input_csv": str(args.threshold_input_csv) if args.threshold_input_csv else None,
            "top_k": args.threshold_top_k,
            "start": args.threshold_start,
            "end": args.threshold_end,
            "step": args.threshold_step,
        },
        "confirm": {
            "input_csv": str(args.confirm_input_csv) if args.confirm_input_csv else None,
            "n_splits": args.confirm_n_splits,
            "n_repeats": args.confirm_n_repeats,
            "random_state": args.confirm_random_state,
        },
        "cv": {"n_splits": args.cv_n_splits, "n_repeats": args.cv_n_repeats},
        "random_state": args.random_state,
    }

    coarse_best_path = args.output_dir / "k_coarse_best.csv"
    narrow_best_path = args.output_dir / "k_narrow_best.csv"

    if args.phase in {"k_coarse", "full"}:
        assert selected_df is not None
        coarse_combos = list(
            itertools.product(
                _parse_int_grid(args.k_coarse_category_grid),
                _parse_int_grid(args.k_coarse_dataset_grid),
                _parse_int_grid(args.k_coarse_cluster_grid),
            )
        )
        coarse_metrics = _evaluate_k_combos(selected_df, coarse_combos, args.k_eval_threshold, cv_df, args)
        coarse_best = _rank_top_n_by_gate(coarse_metrics, args.k_coarse_top_n, args)
        coarse_metrics.to_csv(args.output_dir / "k_coarse_metrics.csv", index=False)
        coarse_best.to_csv(coarse_best_path, index=False)
        print(f"\nSaved coarse-k metrics: {args.output_dir / 'k_coarse_metrics.csv'}")
        print(f"Saved coarse-k shortlist: {coarse_best_path}")

    if args.phase in {"k_narrow", "full"}:
        assert selected_df is not None
        anchor_path = args.k_narrow_input_csv or coarse_best_path
        if not anchor_path.exists():
            raise SystemExit(f"Narrow phase needs anchors CSV; not found: {anchor_path}")
        anchors = pd.read_csv(anchor_path)
        anchors = _filter_selected_configs(anchors, args)
        if anchors.empty:
            raise SystemExit("Narrow phase anchor set is empty after filters.")
        narrow_combos = _build_narrow_k_combos(anchors, args)
        narrow_metrics = _evaluate_k_combos(selected_df, narrow_combos, args.k_eval_threshold, cv_df, args)
        narrow_best = _rank_top_n_by_gate(narrow_metrics, args.k_narrow_top_n, args)
        narrow_metrics.to_csv(args.output_dir / "k_narrow_metrics.csv", index=False)
        narrow_best.to_csv(narrow_best_path, index=False)
        print(f"\nSaved narrow-k metrics: {args.output_dir / 'k_narrow_metrics.csv'}")
        print(f"Saved narrow-k shortlist: {narrow_best_path}")

    if args.phase in {"threshold", "full"}:
        assert selected_df is not None
        input_path = args.threshold_input_csv or narrow_best_path
        if not input_path.exists():
            raise SystemExit(f"Threshold phase needs k shortlist CSV; not found: {input_path}")
        k_candidates = pd.read_csv(input_path)
        k_candidates = _filter_selected_configs(k_candidates, args)
        if k_candidates.empty:
            raise SystemExit("Threshold phase k-candidate set is empty after filters.")
        k_candidates = _rank_top_n_by_gate(k_candidates, args.threshold_top_k, args)
        thresholds = _threshold_grid(args.threshold_start, args.threshold_end, args.threshold_step)

        all_rows: list[dict] = []
        selected_rows: list[dict] = []
        run_start = time.perf_counter()
        total_blocks = len(k_candidates)
        done_blocks = 0
        for row in k_candidates.itertuples(index=False):
            model_key = str(row.model_key)
            mode = str(row.mode)
            gate_type = str(row.gate_type)
            feature_bundle = str(row.feature_bundle)
            params = json.loads(str(row.params_json))
            selected_trial = int(row.selected_trial) if hasattr(row, "selected_trial") else -1
            cat_k = int(row.category_top_k)
            ds_k = int(row.dataset_top_k)
            cl_k = int(row.cluster_top_k)
            print(f"\n=== Threshold sweep: {model_key} ({mode}, {gate_type}) k=({cat_k},{ds_k},{cl_k}) ===")
            fold_scores = _evaluate_config_folds(
                cv_df=cv_df,
                model_key=model_key,
                mode=mode,
                feature_bundle=feature_bundle,
                model_params=params,
                n_splits=args.cv_n_splits,
                n_repeats=args.cv_n_repeats,
                random_state=args.random_state,
                category_top_k=cat_k,
                dataset_top_k=ds_k,
                cluster_top_k=cl_k,
                show_model_logs=args.show_model_logs,
            )
            t_df = pd.DataFrame([_metrics_from_fold_scores(fold_scores, th) for th in thresholds])
            t_df["model_key"] = model_key
            t_df["mode"] = mode
            t_df["gate_type"] = gate_type
            t_df["feature_bundle"] = feature_bundle
            t_df["selected_trial"] = selected_trial
            t_df["params_json"] = json.dumps(params, sort_keys=True)
            t_df["category_top_k"] = cat_k
            t_df["dataset_top_k"] = ds_k
            t_df["cluster_top_k"] = cl_k
            all_rows.extend(t_df.to_dict(orient="records"))

            winner, rationale = _select_for_gate(t_df, gate_type, args)
            selected_rows.append({**winner.to_dict(), "selection_rationale": rationale})
            done_blocks += 1
            elapsed = time.perf_counter() - run_start
            avg = elapsed / max(done_blocks, 1)
            eta_min = (avg * max(total_blocks - done_blocks, 0)) / 60.0
            print(f"  block done [{done_blocks}/{total_blocks}, ETA {eta_min:.1f}m]")

        all_df = pd.DataFrame(all_rows)
        selected_df_out = pd.DataFrame(selected_rows)
        final_df = _select_final_best(selected_df_out, args)
        all_df.to_csv(args.output_dir / "threshold_sweep_metrics.csv", index=False)
        selected_df_out.to_csv(args.output_dir / "threshold_selected_candidates.csv", index=False)
        final_df.to_csv(args.output_dir / "threshold_final_best.csv", index=False)
        print(f"\nSaved threshold metrics: {args.output_dir / 'threshold_sweep_metrics.csv'}")
        print(f"Saved threshold selected: {args.output_dir / 'threshold_selected_candidates.csv'}")
        print(f"Saved threshold final best: {args.output_dir / 'threshold_final_best.csv'}")

    if args.phase in {"confirm"}:
        confirm_df = _run_confirm_phase(cv_df=cv_df, args=args)
        confirm_df.to_csv(args.output_dir / "threshold_confirm_metrics.csv", index=False)
        print(f"\nSaved threshold confirm metrics: {args.output_dir / 'threshold_confirm_metrics.csv'}")

    (args.output_dir / "k_threshold_run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    print(f"Saved run config: {args.output_dir / 'k_threshold_run_config.json'}")


if __name__ == "__main__":
    main()
