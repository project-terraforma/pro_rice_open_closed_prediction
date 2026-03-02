"""Shared feature extraction and bundle selection for open/closed models."""

from __future__ import annotations

import json
import struct
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


class SharedPlaceFeaturizer:
    def __init__(
        self,
        feature_bundle: str = "low_plus_medium",
        use_source_confidence: bool = False,
        use_interactions: bool = True,
        fill_missing_bundle_features: bool = True,
        auto_fit_on_transform: bool = True,
        category_top_k: int = 25,
        dataset_top_k: int = 20,
        cluster_top_k: int = 30,
        spatial_round_decimals: int = 2,
        target_smoothing: float = 20.0,
    ):
        self.feature_bundle = feature_bundle
        self.use_source_confidence = use_source_confidence
        self.use_interactions = use_interactions
        self.fill_missing_bundle_features = fill_missing_bundle_features
        self.auto_fit_on_transform = auto_fit_on_transform

        self.category_top_k = category_top_k
        self.dataset_top_k = dataset_top_k
        self.cluster_top_k = cluster_top_k
        self.spatial_round_decimals = spatial_round_decimals
        self.target_smoothing = target_smoothing

        self._feature_names: list[str] | None = None
        self._is_fitted = False

        # Train-fitted artifacts (reused for val/test to avoid leakage by default)
        self._global_closed_rate = 0.5
        self._category_vocab: list[str] = []
        self._dataset_vocab: list[str] = []
        self._cluster_vocab: list[str] = []
        self._category_risk_map: dict[str, float] = {}
        self._cluster_risk_map: dict[str, float] = {}
        self._category_cluster_risk_map: dict[tuple[str, str], float] = {}
        self._categorical_code_maps: dict[str, dict[str, int]] = {}

    @property
    def feature_names(self) -> list[str] | None:
        return self._feature_names

    def fit(self, df: pd.DataFrame, label_col: str = "open") -> "SharedPlaceFeaturizer":
        """Fit vocabularies and label-derived priors from training data only."""
        if label_col not in df.columns:
            raise ValueError(f"fit requires label column '{label_col}'")

        # Reset train-fitted categorical code maps each fit call so folds/runs
        # do not accidentally reuse mappings from prior training data.
        self._categorical_code_maps = {}

        y_closed = 1 - df[label_col].astype(int)
        self._global_closed_rate = float(y_closed.mean()) if len(y_closed) else 0.5

        category_series = df["categories"].apply(self._extract_primary_category).fillna("UNKNOWN")
        cluster_series = df["geometry"].apply(self._derive_geo_cluster_id)

        self._category_vocab = self._top_k_from_series(category_series, self.category_top_k)

        dataset_tokens = []
        for sources in df["sources"]:
            dataset_tokens.extend(self._extract_source_datasets(sources))
        self._dataset_vocab = [k for k, _ in Counter(dataset_tokens).most_common(self.dataset_top_k)]

        self._cluster_vocab = self._top_k_from_series(cluster_series, self.cluster_top_k)

        # Smoothed target encodings
        self._category_risk_map = self._build_smoothed_rate_map(category_series, y_closed)
        self._cluster_risk_map = self._build_smoothed_rate_map(cluster_series, y_closed)

        pair_keys = pd.Series(list(zip(category_series, cluster_series)), index=df.index)
        self._category_cluster_risk_map = self._build_smoothed_rate_map(pair_keys, y_closed)

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.auto_fit_on_transform and not self._is_fitted and "open" in df.columns:
            self.fit(df, label_col="open")

        features = self._extract_all_features(df)
        features = self._apply_bundle(features)
        self._feature_names = list(features.columns)
        return features

    def _extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)

        # Rules-baseline aliases
        features["websites_present"] = df["websites"].apply(lambda x: 1 if x is not None and len(x) > 0 else 0)
        features["phones_present"] = df["phones"].apply(lambda x: 1 if x is not None and len(x) > 0 else 0)
        features["socials_present"] = df["socials"].apply(lambda x: 1 if x is not None and len(x) > 0 else 0)
        features["addresses_present"] = df["addresses"].apply(lambda x: 1 if x is not None and len(x) > 0 else 0)
        features["sources_n"] = df["sources"].apply(len)

        # Core counts/booleans
        features["num_sources"] = df["sources"].apply(len)
        features["has_multiple_sources"] = (features["num_sources"] >= 2).astype(int)
        features["has_many_sources"] = (features["num_sources"] >= 3).astype(int)
        features["single_source"] = (features["num_sources"] == 1).astype(int)

        features["has_websites"] = features["websites_present"]
        features["num_websites"] = df["websites"].apply(lambda x: len(x) if x is not None else 0)
        features["has_multiple_websites"] = (features["num_websites"] >= 2).astype(int)

        features["has_phones"] = features["phones_present"]
        features["num_phones"] = df["phones"].apply(lambda x: len(x) if x is not None else 0)
        features["has_multiple_phones"] = (features["num_phones"] >= 2).astype(int)

        features["has_socials"] = features["socials_present"]
        features["num_socials"] = df["socials"].apply(lambda x: len(x) if x is not None else 0)
        features["has_multiple_socials"] = (features["num_socials"] >= 2).astype(int)

        features["has_brand"] = df["brand"].apply(lambda x: 1 if x is not None else 0)
        features["has_primary_category"] = df["categories"].apply(lambda x: 1 if x and x.get("primary") else 0)

        def count_categories(categories):
            if not categories:
                return 0
            primary_count = 1 if categories.get("primary") else 0
            alternate_count = len(categories.get("alternate", [])) if categories.get("alternate") is not None else 0
            return primary_count + alternate_count

        features["num_categories"] = df["categories"].apply(count_categories)
        features["has_alternate_categories"] = df["categories"].apply(
            lambda x: 1 if x and x.get("alternate") is not None and len(x.get("alternate", [])) > 0 else 0
        )

        features["num_addresses"] = df["addresses"].apply(lambda x: len(x) if x is not None else 0)
        features["has_addresses"] = (features["num_addresses"] > 0).astype(int)
        features["has_multiple_addresses"] = (features["num_addresses"] >= 2).astype(int)

        def address_completeness(addresses):
            if not addresses:
                return 0
            total_fields = 0
            filled_fields = 0
            for addr in addresses:
                if isinstance(addr, dict):
                    fields = ["country", "region", "locality", "postcode", "address"]
                    total_fields += len(fields)
                    filled_fields += sum(1 for f in fields if addr.get(f))
            return filled_fields / total_fields if total_fields > 0 else 0

        features["address_completeness"] = df["addresses"].apply(address_completeness)

        features["has_name"] = df["names"].apply(lambda x: 1 if x and x.get("primary") else 0)
        features["name_length"] = df["names"].apply(lambda x: len(x["primary"]) if x and x.get("primary") else 0)
        features["has_long_name"] = (features["name_length"] > 20).astype(int)
        features["has_short_name"] = (features["name_length"] <= 5).astype(int)

        def has_chain_pattern(names):
            if not names or not names.get("primary"):
                return 0
            name = names["primary"].lower()
            chain_words = [
                "mcdonalds", "starbucks", "subway", "pizza hut", "kfc", "burger king",
                "walmart", "target", "cvs", "walgreens", "shell", "bp", "chevron",
            ]
            return 1 if any(word in name for word in chain_words) else 0

        features["has_chain_pattern"] = df["names"].apply(has_chain_pattern)

        # Recency
        max_update_times = df["sources"].apply(
            lambda x: max([d.get("update_time") for d in x if d.get("update_time")] or [None])
        )
        max_update_times = pd.to_datetime(max_update_times, errors="coerce", utc=True).dt.tz_localize(None)
        snapshot_time = max_update_times.max()

        if pd.notna(snapshot_time):
            features["recency_days"] = (snapshot_time - max_update_times).dt.days.fillna(-1)
        else:
            features["recency_days"] = -1
        features["max_update_time"] = max_update_times

        features["very_fresh"] = (features["recency_days"] <= 90).astype(int)
        features["fresh"] = ((features["recency_days"] > 90) & (features["recency_days"] <= 365)).astype(int)
        features["stale"] = (features["recency_days"] > 730).astype(int)
        features["very_stale"] = (features["recency_days"] > 1825).astype(int)

        def temporal_diversity(sources):
            times = [pd.to_datetime(s.get("update_time"), errors="coerce", utc=True) for s in sources if s.get("update_time")]
            times = [t.tz_localize(None) if pd.notna(t) and t.tz is not None else t for t in times]
            times = [t for t in times if pd.notna(t)]
            if len(times) <= 1:
                return 0
            time_diffs = [(times[i] - times[0]).days for i in range(1, len(times))]
            return np.std(time_diffs) if time_diffs else 0

        features["source_temporal_diversity"] = df["sources"].apply(temporal_diversity)

        # Dataset diversity + dataset one-hot
        def extract_datasets(sources):
            return self._extract_source_datasets(sources)

        dataset_lists = df["sources"].apply(extract_datasets)
        features["num_datasets"] = dataset_lists.apply(lambda ds: len(set(ds)))
        features["has_multiple_datasets"] = (features["num_datasets"] >= 2).astype(int)

        for dataset in self._dataset_vocab:
            col = f"ohe_source_dataset__{dataset}"
            features[col] = dataset_lists.apply(lambda ds: 1 if dataset in ds else 0)

        # Category one-hot + category prior
        primary_category = df["categories"].apply(self._extract_primary_category).fillna("UNKNOWN")
        for category in self._category_vocab:
            col = f"ohe_primary_category__{category}"
            features[col] = (primary_category == category).astype(int)

        features["category_closure_risk"] = primary_category.map(self._category_risk_map).fillna(self._global_closed_rate)

        # Geometry-derived features
        geom = df["geometry"].apply(self._parse_wkb_point)
        features["_lon"] = geom.apply(lambda xy: xy[0])
        features["_lat"] = geom.apply(lambda xy: xy[1])

        features["geo_h3_cell_id"] = geom.apply(self._derive_geo_h3_id)
        features["geo_cluster_id"] = geom.apply(self._derive_geo_cluster_id)

        # Cluster one-hot + cluster priors
        for cluster in self._cluster_vocab:
            col = f"ohe_geo_cluster__{cluster}"
            features[col] = (features["geo_cluster_id"] == cluster).astype(int)

        cluster_counts = features["geo_cluster_id"].value_counts(dropna=False)
        features["spatial_local_density"] = features["geo_cluster_id"].map(cluster_counts).fillna(0)

        features["spatial_cluster_closed_rate"] = features["geo_cluster_id"].map(self._cluster_risk_map).fillna(self._global_closed_rate)

        cat_cluster_keys = list(zip(primary_category, features["geo_cluster_id"]))
        cat_cluster_prior = (
            pd.Series(cat_cluster_keys, index=df.index).astype(str).map(self._category_cluster_risk_map)
        )
        features["same_category_neighbor_closed_rate"] = cat_cluster_prior.fillna(features["spatial_cluster_closed_rate"])
        features["neighbor_closed_rate"] = features["spatial_cluster_closed_rate"]

        # Composite
        features["completeness_score"] = (
            features["has_websites"] + features["has_phones"] + features["has_socials"]
            + features["has_addresses"] + features["has_brand"] + features["has_primary_category"]
            + features["has_name"]
        )
        features["rich_profile"] = (features["completeness_score"] >= 5).astype(int)

        features["contact_diversity"] = (
            features["has_websites"] + features["has_phones"] + features["has_socials"]
        )
        features["has_full_contact_info"] = (features["contact_diversity"] == 3).astype(int)

        # Interactions
        if self.use_interactions:
            features["brand_with_contacts"] = features["has_brand"] * features["contact_diversity"]
            features["recent_with_contacts"] = features["very_fresh"] * features["contact_diversity"]
            features["multiple_sources_with_contacts"] = (
                features["has_multiple_sources"] * features["contact_diversity"]
            )
            features["multi_dataset_with_contacts"] = (
                features["has_multiple_datasets"] * features["contact_diversity"]
            )
            features["single_source_no_socials"] = (
                features["single_source"] * (1 - features["has_socials"])
            ).astype(int)

        # Confidence-derived features (optional)
        if self.use_source_confidence:
            features["mean_source_conf"] = df["sources"].apply(
                lambda x: np.mean([d.get("confidence") for d in x if d.get("confidence") is not None])
                if len(x) > 0 else 0
            )
            features["max_source_conf"] = df["sources"].apply(
                lambda x: np.max([d.get("confidence") for d in x if d.get("confidence") is not None])
                if len(x) > 0 else 0
            )
            features["min_source_conf"] = df["sources"].apply(
                lambda x: np.min([d.get("confidence") for d in x if d.get("confidence") is not None])
                if len(x) > 0 else 0
            )
            features["source_conf_std"] = df["sources"].apply(
                lambda x: np.std([d.get("confidence") for d in x if d.get("confidence") is not None])
                if len(x) > 1 else 0
            )
            features["high_source_conf"] = (features["max_source_conf"] >= 0.90).astype(int)
            features["low_source_conf"] = (features["max_source_conf"] < 0.70).astype(int)
            features["high_conf_with_contacts"] = features["high_source_conf"] * features["contact_diversity"]

        # Keep human-readable categorical IDs for wildcard expansion then drop lat/lon helpers.
        features = features.fillna(0)
        features = features.drop(columns=["_lon", "_lat"], errors="ignore")
        return features

    def _apply_bundle(self, features: pd.DataFrame) -> pd.DataFrame:
        bundle_features = self._load_bundle_features(self.feature_bundle)
        selected_cols: dict[str, pd.Series] = {}

        def _assign_column(col_name: str):
            # Ensure all model inputs are numeric.
            if not is_numeric_dtype(features[col_name]):
                selected_cols[col_name] = self._encode_categorical_column(col_name, features[col_name])
            else:
                selected_cols[col_name] = features[col_name]

        for token in bundle_features:
            expanded = self._expand_feature_token(token, features.columns)
            if expanded:
                for col in expanded:
                    _assign_column(col)
                continue
            if self.fill_missing_bundle_features:
                selected_cols[token] = pd.Series(0.0, index=features.index)
            else:
                raise ValueError(f"Bundle feature '{token}' not present in extracted features")

        selected = pd.DataFrame(selected_cols, index=features.index)

        # Final guardrail: all selected features must be numeric for sklearn/tree libs.
        for col in selected.columns:
            if not is_numeric_dtype(selected[col]):
                selected[col] = pd.to_numeric(selected[col], errors="coerce").fillna(-1)

        return selected

    def _encode_categorical_column(self, col_name: str, series: pd.Series) -> pd.Series:
        """Encode categorical/string columns using a stable map learned on train."""
        str_series = series.astype(str)
        code_map = self._categorical_code_maps.get(col_name)

        if code_map is None:
            # Build once (typically on first train transform), then reuse for val/test.
            uniques = sorted(str_series.dropna().unique().tolist())
            code_map = {val: idx for idx, val in enumerate(uniques)}
            self._categorical_code_maps[col_name] = code_map

        return str_series.map(code_map).fillna(-1).astype(int)

    @staticmethod
    def _expand_feature_token(token: str, available_columns: Iterable[str]) -> list[str]:
        if "*" not in token:
            return [token] if token in available_columns else []

        prefix = token.split("*")[0]
        matches = [c for c in available_columns if c.startswith(prefix)]
        return sorted(matches)

    @staticmethod
    def _bundle_file_path() -> Path:
        return Path(__file__).resolve().parents[2] / "docs" / "feature_bundles.json"

    def _load_bundle_features(self, bundle_name: str) -> list[str]:
        path = self._bundle_file_path()
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        bundles = data.get("bundles", {})
        if bundle_name not in bundles:
            raise ValueError(f"Unknown feature bundle '{bundle_name}'. Available: {sorted(bundles)}")
        return list(bundles[bundle_name])

    def _top_k_from_series(self, series: pd.Series, top_k: int) -> list[str]:
        counts = Counter(series.astype(str).tolist())
        return [k for k, _ in counts.most_common(top_k)]

    def _build_smoothed_rate_map(self, keys: pd.Series, y_closed: pd.Series) -> dict:
        df_tmp = pd.DataFrame({"key": keys.astype(str), "y_closed": y_closed.astype(float)})
        grouped = df_tmp.groupby("key", dropna=False)["y_closed"].agg(["sum", "count"])
        prior = self._global_closed_rate
        m = self.target_smoothing
        smoothed = (grouped["sum"] + m * prior) / (grouped["count"] + m)
        return smoothed.to_dict()

    @staticmethod
    def _extract_primary_category(categories) -> str:
        if not categories:
            return "UNKNOWN"
        val = categories.get("primary")
        return str(val) if val else "UNKNOWN"

    @staticmethod
    def _extract_source_datasets(sources) -> list[str]:
        out = []
        if sources is None:
            return out
        for s in sources:
            if isinstance(s, dict):
                ds = s.get("dataset")
                if ds:
                    out.append(str(ds))
        return out

    @staticmethod
    def _parse_wkb_point(geom) -> tuple[float, float]:
        if isinstance(geom, (bytes, bytearray)) and len(geom) >= 21:
            byte_order = geom[0]
            if byte_order == 1:
                endian = "<"
            elif byte_order == 0:
                endian = ">"
            else:
                return (np.nan, np.nan)
            try:
                geom_type = struct.unpack(endian + "I", geom[1:5])[0]
                if geom_type != 1:  # point
                    return (np.nan, np.nan)
                x, y = struct.unpack(endian + "dd", geom[5:21])
                return (float(x), float(y))
            except Exception:
                return (np.nan, np.nan)

        # Fallback for dict-like geometry
        if isinstance(geom, dict):
            coords = geom.get("coordinates")
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                return (float(coords[0]), float(coords[1]))

        return (np.nan, np.nan)

    def _derive_geo_h3_id(self, xy: tuple[float, float]) -> str:
        lon, lat = xy
        if np.isnan(lon) or np.isnan(lat):
            return "H3_UNKNOWN"
        return f"H3_{round(lat, self.spatial_round_decimals)}_{round(lon, self.spatial_round_decimals)}"

    def _derive_geo_cluster_id(self, geom) -> str:
        if isinstance(geom, tuple) and len(geom) >= 2:
            lon, lat = geom[0], geom[1]
        else:
            lon, lat = self._parse_wkb_point(geom)
        if np.isnan(lon) or np.isnan(lat):
            return "CL_UNKNOWN"
        lat_bin = round(lat, max(self.spatial_round_decimals - 1, 1))
        lon_bin = round(lon, max(self.spatial_round_decimals - 1, 1))
        return f"CL_{lat_bin}_{lon_bin}"
