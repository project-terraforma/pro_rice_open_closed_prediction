"""Quick group-ablation screen on frozen configs with fixed k/threshold."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
import types
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

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


def _load_bundle_tokens(bundle_name: str, repo_root: Path) -> list[str]:
    p = repo_root / "docs" / "feature_bundles.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    bundles = data.get("bundles", {})
    if bundle_name not in bundles:
        raise ValueError(f"Unknown feature bundle: {bundle_name}")
    return list(bundles[bundle_name])


def _variant_drop_predicates() -> dict[str, callable]:
    spatial_prior_exact = {"spatial_cluster_closed_rate"}
    spatial_id_exact = {"geo_cluster_id"}
    category_prior_exact = {"category_closure_risk"}
    category_metadata_exact = {"has_primary_category", "has_alternate_categories", "num_categories"}
    source_counts_exact = {
        "num_sources",
        "sources_n",
        "single_source",
        "has_multiple_sources",
        "has_many_sources",
        "num_datasets",
        "has_multiple_datasets",
    }
    temporal_recency_bins_exact = {"fresh", "very_fresh", "stale", "very_stale"}
    temporal_continuous_exact = {"max_update_time", "recency_days"}
    temporal_diversity_exact = {"source_temporal_diversity"}
    contact_presence_counts_exact = {
        "has_phones",
        "num_phones",
        "phones_present",
        "has_socials",
        "num_socials",
        "socials_present",
        "has_websites",
        "num_websites",
        "websites_present",
        "has_full_contact_info",
        "contact_diversity",
    }
    completeness_aggregates_exact = {"address_completeness", "has_addresses", "num_addresses", "addresses_present", "completeness_score", "rich_profile"}
    name_lexical_exact = {"has_name", "name_length", "has_long_name", "has_short_name", "has_chain_pattern"}
    brand_exact = {"has_brand"}
    interaction_contact_temporal_exact = {"recent_with_contacts"}
    interaction_source_contact_exact = {"multi_dataset_with_contacts", "multiple_sources_with_contacts"}
    interaction_sparse_risk_exact = {"single_source_no_socials"}
    interaction_brand_contact_exact = {"brand_with_contacts"}

    preds = {
        # Coarse groups
        "minus_spatial": lambda t: t in spatial_prior_exact or t in spatial_id_exact or t.startswith("ohe_geo_cluster__"),
        "minus_category": lambda t: t in category_prior_exact or t in category_metadata_exact or t.startswith("ohe_primary_category__"),
        "minus_source_provenance": lambda t: t in source_counts_exact or t.startswith("ohe_source_dataset__"),
        "minus_temporal": lambda t: t in temporal_recency_bins_exact or t in temporal_continuous_exact or t in temporal_diversity_exact,
        "minus_completeness_contact": lambda t: t in contact_presence_counts_exact or t in completeness_aggregates_exact or t in name_lexical_exact or t in brand_exact,
        "minus_interactions": lambda t: (
            t in interaction_contact_temporal_exact
            or t in interaction_source_contact_exact
            or t in interaction_sparse_risk_exact
            or t in interaction_brand_contact_exact
        ),
        # Split groups
        "minus_spatial_prior_only": lambda t: t in spatial_prior_exact,
        "minus_spatial_ids_only": lambda t: t in spatial_id_exact or t.startswith("ohe_geo_cluster__"),
        "minus_category_prior_only": lambda t: t in category_prior_exact,
        "minus_category_ohe_only": lambda t: t.startswith("ohe_primary_category__"),
        "minus_category_metadata_only": lambda t: t in category_metadata_exact,
        "minus_source_counts_only": lambda t: t in source_counts_exact,
        "minus_source_ohe_only": lambda t: t.startswith("ohe_source_dataset__"),
        "minus_temporal_recency_bins": lambda t: t in temporal_recency_bins_exact,
        "minus_temporal_continuous": lambda t: t in temporal_continuous_exact,
        "minus_temporal_diversity_only": lambda t: t in temporal_diversity_exact,
        "minus_contact_presence_counts": lambda t: t in contact_presence_counts_exact,
        "minus_completeness_aggregates": lambda t: t in completeness_aggregates_exact,
        "minus_name_lexical": lambda t: t in name_lexical_exact,
        "minus_brand_only": lambda t: t in brand_exact,
        "minus_interactions_contact_temporal": lambda t: t in interaction_contact_temporal_exact,
        "minus_interactions_source_contact": lambda t: t in interaction_source_contact_exact,
        "minus_interactions_sparse_risk": lambda t: t in interaction_sparse_risk_exact,
        "minus_interactions_brand_contact": lambda t: t in interaction_brand_contact_exact,
    }
    return preds


def _tokens_for_variant(base_tokens: list[str], variant: str) -> list[str]:
    if variant == "baseline":
        return list(base_tokens)
    preds = _variant_drop_predicates()
    if variant not in preds:
        raise ValueError(f"Unknown variant: {variant}")
    keep = [t for t in base_tokens if not preds[variant](t)]
    if not keep:
        raise ValueError(f"Variant {variant} produced zero features")
    return keep


def _resolve_variants(args: argparse.Namespace) -> list[str]:
    coarse = [
        "minus_spatial",
        "minus_category",
        "minus_source_provenance",
        "minus_temporal",
        "minus_completeness_contact",
        "minus_interactions",
    ]
    split = [
        "minus_spatial_prior_only",
        "minus_spatial_ids_only",
        "minus_category_prior_only",
        "minus_category_ohe_only",
        "minus_category_metadata_only",
        "minus_source_counts_only",
        "minus_source_ohe_only",
        "minus_temporal_recency_bins",
        "minus_temporal_continuous",
        "minus_temporal_diversity_only",
        "minus_contact_presence_counts",
        "minus_completeness_aggregates",
        "minus_name_lexical",
        "minus_brand_only",
        "minus_interactions_contact_temporal",
        "minus_interactions_source_contact",
        "minus_interactions_sparse_risk",
        "minus_interactions_brand_contact",
    ]
    if args.variants:
        variants = [v.strip() for v in args.variants if v.strip()]
    elif args.variant_scope == "coarse":
        variants = coarse
    elif args.variant_scope == "split":
        variants = split
    else:
        variants = coarse + split

    preds = _variant_drop_predicates()
    unknown = [v for v in variants if v not in preds]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    return ["baseline"] + variants


def _apply_bundle_tokens(model, tokens: list[str]) -> None:
    featurizer = getattr(model, "featurizer", None)
    if featurizer is None:
        raise ValueError(f"Model does not expose featurizer: {type(model)}")

    def _load_bundle_features(self, bundle_name: str) -> list[str]:
        return list(tokens)

    featurizer._load_bundle_features = types.MethodType(_load_bundle_features, featurizer)


def _evaluate_variant_cv(
    cv_df: pd.DataFrame,
    model_key: str,
    mode: str,
    feature_bundle: str,
    model_params: dict,
    bundle_tokens: list[str],
    n_splits: int,
    n_repeats: int,
    random_state: int,
    decision_threshold: float,
    category_top_k: int,
    dataset_top_k: int,
    cluster_top_k: int,
    show_model_logs: bool,
) -> tuple[pd.DataFrame, dict]:
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    y = cv_df["open"].astype(int).values
    fold_rows: list[dict] = []

    for split_idx, (tr_idx, va_idx) in enumerate(splitter.split(cv_df, y), start=1):
        repeat_idx = (split_idx - 1) // n_splits + 1
        fold_idx = (split_idx - 1) % n_splits + 1

        tr_df = cv_df.iloc[tr_idx].reset_index(drop=True)
        va_df = cv_df.iloc[va_idx].reset_index(drop=True)
        params = _normalize_model_params(model_key, model_params)
        model = _build_model(model_key, mode, feature_bundle, params)
        _apply_bundle_tokens(model, bundle_tokens)

        featurizer = getattr(model, "featurizer", None)
        if featurizer is not None:
            featurizer.category_top_k = int(category_top_k)
            featurizer.dataset_top_k = int(dataset_top_k)
            featurizer.cluster_top_k = int(cluster_top_k)

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
        y_pred = (p_open >= decision_threshold).astype(int)
        m = compute_metrics(y_true=y_true, y_pred=y_pred, y_score_open=p_open)
        fold_rows.append({"repeat": repeat_idx, "fold": fold_idx, "n_features": len(bundle_tokens), **m})

    fold_df = pd.DataFrame(fold_rows)
    out = {"n_features": float(fold_df["n_features"].mean())}
    for c in ["accuracy", "open_precision", "open_recall", "open_f1", "closed_precision", "closed_recall", "closed_f1", "pr_auc_closed"]:
        out[f"{c}_mean"] = float(fold_df[c].mean())
        out[f"{c}_std"] = float(fold_df[c].std(ddof=0))
    return fold_df, out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline + minus-one-group ablation screen.")
    parser.add_argument(
        "--selected-configs-csv",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "hpo_optuna_narrow_pass1" / "hpo_selected_trials_dualgate.csv",
        help="Frozen configs source CSV (expects params_json/model/mode/gate_type).",
    )
    parser.add_argument("--models", nargs="+", default=["lr", "rf"])
    parser.add_argument("--modes", nargs="+", default=["single", "two-stage"])
    parser.add_argument("--gate-types", nargs="+", default=["diagnostic"], help="Usually diagnostic for ablation screen.")
    parser.add_argument(
        "--feature-bundle-map-json",
        type=Path,
        default=None,
        help="Optional JSON map, e.g. {'lr:two-stage':'v2_lr2','rf:single':'v2_rf_single'}",
    )
    parser.add_argument("--category-top-k", type=int, default=25)
    parser.add_argument("--dataset-top-k", type=int, default=20)
    parser.add_argument("--cluster-top-k", type=int, default=30)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts" / "ablation_v2")
    parser.add_argument("--show-model-logs", action="store_true")
    parser.add_argument(
        "--variant-scope",
        choices=["coarse", "split", "all"],
        default="all",
        help="Which ablation set to run when --variants is not provided.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Optional explicit minus_* variants to run (overrides --variant-scope).",
    )
    return parser.parse_args()


def _resolve_feature_bundle(model_key: str, mode: str, default_bundle_from_row: str, bundle_map: dict[str, str]) -> str:
    key = f"{model_key}:{mode}"
    if key in bundle_map:
        return bundle_map[key]
    if model_key == "lr" and mode == "two-stage":
        return "v2_lr2"
    if model_key == "rf" and mode == "single":
        return "v2_rf_single"
    return default_bundle_from_row


def main() -> None:
    args = _parse_args()
    if not args.selected_configs_csv.exists():
        raise SystemExit(f"Selected configs CSV not found: {args.selected_configs_csv}")

    train_path = args.data_dir / "train_split.parquet"
    val_path = args.data_dir / "val_split.parquet"
    if not train_path.exists() or not val_path.exists():
        raise SystemExit(f"Missing split files in {args.data_dir}; expected train_split.parquet and val_split.parquet")
    cv_df = pd.concat([pd.read_parquet(train_path), pd.read_parquet(val_path)], axis=0, ignore_index=True)

    cfg_df = pd.read_csv(args.selected_configs_csv)
    cfg_df = cfg_df[cfg_df["model_key"].isin(set(args.models))]
    cfg_df = cfg_df[cfg_df["mode"].isin(set(args.modes))]
    cfg_df = cfg_df[cfg_df["gate_type"].isin(set(args.gate_types))]
    if cfg_df.empty:
        raise SystemExit("No configs left after filters.")

    bundle_map = {}
    if args.feature_bundle_map_json:
        bundle_map = json.loads(args.feature_bundle_map_json.read_text(encoding="utf-8"))

    repo_root = Path(__file__).resolve().parents[2]
    variants = _resolve_variants(args)

    fold_rows: list[dict] = []
    summary_rows: list[dict] = []
    run_start = time.perf_counter()
    total_tasks = len(cfg_df) * len(variants)
    done_tasks = 0

    for row in cfg_df.itertuples(index=False):
        model_key = str(row.model_key)
        mode = str(row.mode)
        gate_type = str(row.gate_type)
        selected_trial = int(row.selected_trial) if hasattr(row, "selected_trial") else -1
        params = json.loads(str(row.params_json))
        feature_bundle = _resolve_feature_bundle(model_key, mode, str(row.feature_bundle), bundle_map)
        base_tokens = _load_bundle_tokens(feature_bundle, repo_root=repo_root)

        print(f"\n=== Ablation: {model_key} ({mode}, {gate_type}) trial={selected_trial} bundle={feature_bundle} ===")
        base_closed_f1 = None
        base_pr_auc = None

        for variant_idx, variant in enumerate(variants, start=1):
            t0 = time.perf_counter()
            tokens = _tokens_for_variant(base_tokens, variant)
            folds, summary = _evaluate_variant_cv(
                cv_df=cv_df,
                model_key=model_key,
                mode=mode,
                feature_bundle=feature_bundle,
                model_params=params,
                bundle_tokens=tokens,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                random_state=args.random_state,
                decision_threshold=args.decision_threshold,
                category_top_k=args.category_top_k,
                dataset_top_k=args.dataset_top_k,
                cluster_top_k=args.cluster_top_k,
                show_model_logs=args.show_model_logs,
            )

            folds["model_key"] = model_key
            folds["mode"] = mode
            folds["gate_type"] = gate_type
            folds["selected_trial"] = selected_trial
            folds["feature_bundle"] = feature_bundle
            folds["variant"] = variant
            fold_rows.extend(folds.to_dict(orient="records"))

            if variant == "baseline":
                base_closed_f1 = summary["closed_f1_mean"]
                base_pr_auc = summary["pr_auc_closed_mean"]

            summary_rows.append(
                {
                    "model_key": model_key,
                    "mode": mode,
                    "gate_type": gate_type,
                    "selected_trial": selected_trial,
                    "feature_bundle": feature_bundle,
                    "variant": variant,
                    **summary,
                    "delta_closed_f1_vs_baseline": float(summary["closed_f1_mean"] - (base_closed_f1 or summary["closed_f1_mean"])),
                    "delta_pr_auc_closed_vs_baseline": float(summary["pr_auc_closed_mean"] - (base_pr_auc or summary["pr_auc_closed_mean"])),
                }
            )
            done_tasks += 1
            elapsed_total = time.perf_counter() - run_start
            avg_task = elapsed_total / max(done_tasks, 1)
            remaining = max(total_tasks - done_tasks, 0)
            eta_min = (avg_task * remaining) / 60.0
            print(
                f"  [{variant_idx}/{len(variants)} | {done_tasks}/{total_tasks}] "
                f"{variant:<24} n={len(tokens):>2} "
                f"cf1={summary['closed_f1_mean']:.3f} "
                f"pr_auc={summary['pr_auc_closed_mean']:.3f} "
                f"(variant {time.perf_counter() - t0:.1f}s, ETA {eta_min:.1f}m)"
            )

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_path = args.output_dir / "group_ablation_folds.csv"
    summary_path = args.output_dir / "group_ablation_summary.csv"
    config_path = args.output_dir / "group_ablation_run_config.json"

    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "selected_configs_csv": str(args.selected_configs_csv),
                "models": args.models,
                "modes": args.modes,
                "gate_types": args.gate_types,
                "feature_bundle_map_json": str(args.feature_bundle_map_json) if args.feature_bundle_map_json else None,
                "k": {
                    "category_top_k": args.category_top_k,
                    "dataset_top_k": args.dataset_top_k,
                    "cluster_top_k": args.cluster_top_k,
                },
                "decision_threshold": args.decision_threshold,
                "cv": {"n_splits": args.n_splits, "n_repeats": args.n_repeats},
                "random_state": args.random_state,
                "variants": variants,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved fold metrics: {fold_path}")
    print(f"Saved summary metrics: {summary_path}")
    print(f"Saved run config: {config_path}")


if __name__ == "__main__":
    main()
