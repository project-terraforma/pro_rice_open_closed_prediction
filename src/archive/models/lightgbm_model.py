"""
LightGBM Model for Open/Closed Prediction

Supports:
1) Single-stage model
2) Two-stage model (rule filter -> model on uncertain cases)
3) Optional per-source confidence features

No global confidence features are used.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib


class LightGBMModel:
    def __init__(
        self,
        mode: str = "two-stage",
        use_source_confidence: bool = False,
        use_interactions: bool = True,
    ):
        if mode not in ("single", "two-stage"):
            raise ValueError("mode must be 'single' or 'two-stage'")
        self.mode = mode
        self.use_source_confidence = use_source_confidence
        self.use_interactions = use_interactions
        self.model = None
        self._feature_names = None

    def _get_variant_name(self) -> str:
        conf_str = "with Source Confidence" if self.use_source_confidence else "No Confidence"
        mode_str = "Two-Stage" if self.mode == "two-stage" else "Single-Stage"
        return f"{mode_str} LightGBM ({conf_str})"

    def stage1_filter(self, df: pd.DataFrame) -> pd.Series:
        """Rule-based filter to identify obviously open places."""
        obviously_open = (
            ((df["websites"].apply(lambda x: x is not None and len(x) > 0)) &
             (df["phones"].apply(lambda x: x is not None and len(x) > 0)) &
             (df["socials"].apply(lambda x: x is not None and len(x) > 0))) |
            ((df["brand"].apply(lambda x: x is not None)) &
             (df["websites"].apply(lambda x: x is not None and len(x) > 0)) &
             (df["phones"].apply(lambda x: x is not None and len(x) > 0))) |
            ((df["sources"].apply(lambda x: len(x) >= 3)) &
             (df["websites"].apply(lambda x: x is not None and len(x) > 0)) &
             (df["phones"].apply(lambda x: x is not None and len(x) > 0))) |
            ((df["websites"].apply(lambda x: x is not None and len(x) >= 2)) &
             (df["phones"].apply(lambda x: x is not None and len(x) > 0)))
        )
        return obviously_open

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature extraction mirrored from logistic_regression.py."""
        features = pd.DataFrame(index=df.index)

        # === BASIC SOURCE & DATA FEATURES ===
        features["num_sources"] = df["sources"].apply(len)
        features["has_multiple_sources"] = (features["num_sources"] >= 2).astype(int)
        features["has_many_sources"] = (features["num_sources"] >= 3).astype(int)

        # === CONTACT INFORMATION FEATURES ===
        features["has_websites"] = df["websites"].apply(lambda x: 1 if x is not None and len(x) > 0 else 0)
        features["num_websites"] = df["websites"].apply(lambda x: len(x) if x is not None else 0)
        features["has_multiple_websites"] = (features["num_websites"] >= 2).astype(int)

        features["has_phones"] = df["phones"].apply(lambda x: 1 if x is not None and len(x) > 0 else 0)
        features["num_phones"] = df["phones"].apply(lambda x: len(x) if x is not None else 0)
        features["has_multiple_phones"] = (features["num_phones"] >= 2).astype(int)

        features["has_socials"] = df["socials"].apply(lambda x: 1 if x is not None and len(x) > 0 else 0)
        features["num_socials"] = df["socials"].apply(lambda x: len(x) if x is not None else 0)
        features["has_multiple_socials"] = (features["num_socials"] >= 2).astype(int)

        # === BRAND & CATEGORY FEATURES ===
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

        # === ADDRESS FEATURES ===
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

        # === NAME FEATURES ===
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
                "walmart", "target", "cvs", "walgreens", "shell", "bp", "chevron"
            ]
            return 1 if any(word in name for word in chain_words) else 0

        features["has_chain_pattern"] = df["names"].apply(has_chain_pattern)

        # === TEMPORAL FEATURES ===
        max_update_times = df["sources"].apply(
            lambda x: max([d.get("update_time") for d in x if d.get("update_time")] or [None])
        )
        max_update_times = pd.to_datetime(max_update_times, errors="coerce")
        snapshot_time = max_update_times.max()
        if pd.notna(snapshot_time):
            features["recency_days"] = (snapshot_time - max_update_times).dt.days.fillna(-1)
        else:
            features["recency_days"] = -1

        features["very_fresh"] = (features["recency_days"] <= 90).astype(int)
        features["fresh"] = ((features["recency_days"] > 90) & (features["recency_days"] <= 365)).astype(int)
        features["stale"] = (features["recency_days"] > 730).astype(int)
        features["very_stale"] = (features["recency_days"] > 1825).astype(int)

        def temporal_diversity(sources):
            times = [pd.to_datetime(s.get("update_time"), errors="coerce") for s in sources if s.get("update_time")]
            times = [t for t in times if pd.notna(t)]
            if len(times) <= 1:
                return 0
            time_diffs = [(times[i] - times[0]).days for i in range(1, len(times))]
            return np.std(time_diffs) if time_diffs else 0

        features["source_temporal_diversity"] = df["sources"].apply(temporal_diversity)

        # === DATASET DIVERSITY FEATURES ===
        def dataset_diversity(sources):
            datasets = set()
            for source in sources:
                if source.get("dataset"):
                    datasets.add(source["dataset"])
            return len(datasets)

        features["num_datasets"] = df["sources"].apply(dataset_diversity)
        features["has_multiple_datasets"] = (features["num_datasets"] >= 2).astype(int)

        # === COMPOSITE COMPLETENESS FEATURES ===
        features["completeness_score"] = (
            features["has_websites"] + features["has_phones"] +
            features["has_socials"] + features["has_addresses"] +
            features["has_brand"] + features["has_primary_category"] +
            features["has_name"]
        )
        features["rich_profile"] = (features["completeness_score"] >= 5).astype(int)

        features["contact_diversity"] = (
            features["has_websites"] + features["has_phones"] + features["has_socials"]
        )
        features["has_full_contact_info"] = (features["contact_diversity"] == 3).astype(int)

        # === INTERACTION FEATURES (optional) ===
        if self.use_interactions:
            features["brand_with_contacts"] = features["has_brand"] * features["contact_diversity"]
            features["recent_with_contacts"] = features["very_fresh"] * features["contact_diversity"]
            features["multiple_sources_with_contacts"] = (
                features["has_multiple_sources"] * features["contact_diversity"]
            )

        # === SOURCE CONFIDENCE FEATURES (optional) ===
        if self.use_source_confidence:
            features["mean_source_conf"] = df["sources"].apply(
                lambda x: np.mean([d["confidence"] for d in x]) if len(x) > 0 else 0
            )
            features["max_source_conf"] = df["sources"].apply(
                lambda x: np.max([d["confidence"] for d in x]) if len(x) > 0 else 0
            )
            features["min_source_conf"] = df["sources"].apply(
                lambda x: np.min([d["confidence"] for d in x]) if len(x) > 0 else 0
            )
            features["source_conf_std"] = df["sources"].apply(
                lambda x: np.std([d["confidence"] for d in x]) if len(x) > 1 else 0
            )
            features["high_source_conf"] = (features["max_source_conf"] >= 0.90).astype(int)
            features["low_source_conf"] = (features["max_source_conf"] < 0.70).astype(int)
            features["high_conf_with_contacts"] = features["high_source_conf"] * features["contact_diversity"]

        self._feature_names = list(features.columns)
        return features

    def _init_model(self, scale_pos_weight: float | None = None) -> LGBMClassifier:
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            class_weight=None if scale_pos_weight is not None else "balanced",
            scale_pos_weight=scale_pos_weight,
        )

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        print(f"\n{'='*60}")
        print(f"Training: {self._get_variant_name()}")
        print(f"{'='*60}")

        if self.mode == "two-stage":
            self._fit_two_stage(train_df, val_df)
        else:
            self._fit_single_stage(train_df, val_df)
        return self

    def _fit_single_stage(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        X_train = self.extract_features(train_df)
        y_train = train_df["open"]

        print(f"Training on {len(train_df)} samples:")
        print(f"  Open: {y_train.sum()} ({y_train.mean():.1%})")
        print(f"  Closed: {(~y_train.astype(bool)).sum()} ({(~y_train.astype(bool)).mean():.1%})")
        print(f"  Features: {X_train.shape[1]}")

        self.model = self._init_model()
        if val_df is not None:
            X_val = self.extract_features(val_df)
            y_val = val_df["open"]
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        else:
            self.model.fit(X_train, y_train)

        if val_df is not None:
            self._print_validation_report(val_df)

    def _fit_two_stage(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        obviously_open_mask = self.stage1_filter(train_df)

        print("Stage 1 Filter Results:")
        print(f"  Total training samples: {len(train_df)}")
        print(f"  Obviously open (filtered): {obviously_open_mask.sum()} ({obviously_open_mask.mean():.1%})")
        print(f"  Uncertain (to classify): {(~obviously_open_mask).sum()} ({(~obviously_open_mask).mean():.1%})")

        stage1_correct = train_df[obviously_open_mask]["open"].mean()
        print(f"  Stage 1 accuracy on 'obviously open': {stage1_correct:.3f}")
        print()

        uncertain_train = train_df[~obviously_open_mask].copy()
        if len(uncertain_train) > 0:
            X_uncertain = self.extract_features(uncertain_train)
            y_uncertain = uncertain_train["open"]
            print(f"Stage 2 Training on {len(uncertain_train)} uncertain samples:")
            print(f"  Open: {y_uncertain.sum()} ({y_uncertain.mean():.1%})")
            print(f"  Closed: {(~y_uncertain.astype(bool)).sum()} ({(~y_uncertain.astype(bool)).mean():.1%})")
            print(f"  Features: {X_uncertain.shape[1]}")

            self.model = self._init_model()
            self.model.fit(X_uncertain, y_uncertain)
        else:
            print("No uncertain cases to train Stage 2 model on.")

        if val_df is not None:
            self._print_validation_report(val_df)

    def _print_validation_report(self, val_df: pd.DataFrame):
        predictions = self.predict(val_df)
        y_val = val_df["open"]

        print("\nValidation Report:")
        prec_open = precision_score(y_val, predictions, pos_label=1)
        rec_open = recall_score(y_val, predictions, pos_label=1)
        f1_open = f1_score(y_val, predictions, pos_label=1)
        prec_closed = precision_score(y_val, predictions, pos_label=0, zero_division=0)
        rec_closed = recall_score(y_val, predictions, pos_label=0, zero_division=0)
        f1_closed = f1_score(y_val, predictions, pos_label=0, zero_division=0)
        acc = accuracy_score(y_val, predictions)

        print(f"  Open:   Precision: {prec_open:.3f}  Recall: {rec_open:.3f}  F1: {f1_open:.3f}")
        print(f"  Closed: Precision: {prec_closed:.3f}  Recall: {rec_closed:.3f}  F1: {f1_closed:.3f}")
        print(f"  Accuracy: {acc:.3f}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet.")
        if self.mode == "two-stage":
            return self._predict_two_stage(df)
        return self._predict_single_stage(df)

    def _predict_single_stage(self, df: pd.DataFrame) -> np.ndarray:
        X = self.extract_features(df)
        return self.model.predict(X)

    def _predict_two_stage(self, df: pd.DataFrame) -> np.ndarray:
        predictions = np.ones(len(df))
        obviously_open_mask = self.stage1_filter(df)
        predictions[obviously_open_mask] = 1

        uncertain_mask = ~obviously_open_mask
        if uncertain_mask.sum() > 0:
            X_uncertain = self.extract_features(df[uncertain_mask])
            uncertain_predictions = self.model.predict(X_uncertain)
            predictions[uncertain_mask] = uncertain_predictions
        return predictions.astype(int)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet.")
        if self.mode == "two-stage":
            return self._predict_proba_two_stage(df)
        return self._predict_proba_single_stage(df)

    def _predict_proba_single_stage(self, df: pd.DataFrame) -> np.ndarray:
        X = self.extract_features(df)
        return self.model.predict_proba(X)

    def _predict_proba_two_stage(self, df: pd.DataFrame) -> np.ndarray:
        proba = np.ones((len(df), 2))
        proba[:, 0] = 0
        proba[:, 1] = 1

        obviously_open_mask = self.stage1_filter(df)
        proba[obviously_open_mask, 0] = 0.01
        proba[obviously_open_mask, 1] = 0.99

        uncertain_mask = ~obviously_open_mask
        if uncertain_mask.sum() > 0:
            X_uncertain = self.extract_features(df[uncertain_mask])
            uncertain_proba = self.model.predict_proba(X_uncertain)
            proba[uncertain_mask] = uncertain_proba
        return proba

    def get_feature_importances(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained yet.")
        importances = pd.DataFrame({
            "feature": self._feature_names,
            "importance": self.model.feature_importances_,
        })
        return importances.sort_values("importance", ascending=False)

    def save_model(self, path: str):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        model_data = {
            "model": self.model,
            "mode": self.mode,
            "use_source_confidence": self.use_source_confidence,
            "feature_names": self._feature_names,
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")


def evaluate_model(y_true, y_pred, model_name: str) -> dict:
    prec_open = precision_score(y_true, y_pred, pos_label=1)
    rec_open = recall_score(y_true, y_pred, pos_label=1)
    f1_open = f1_score(y_true, y_pred, pos_label=1)
    prec_closed = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec_closed = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_closed = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    print(f"\n{model_name} - Test Results:")
    print(f"  Open:   Precision: {prec_open:.3f}  Recall: {rec_open:.3f}  F1: {f1_open:.3f}")
    print(f"  Closed: Precision: {prec_closed:.3f}  Recall: {rec_closed:.3f}  F1: {f1_closed:.3f}")
    print(f"  Accuracy: {acc:.3f}")

    return {
        "model": model_name,
        "open_precision": prec_open,
        "open_recall": rec_open,
        "open_f1": f1_open,
        "closed_precision": prec_closed,
        "closed_recall": rec_closed,
        "closed_f1": f1_closed,
        "accuracy": acc,
    }


def print_stage_breakdown(model: LightGBMModel, eval_df: pd.DataFrame, decision_threshold: float = 0.5):
    """
    Compare stage contributions for two-stage models:
    1) Stage 1 signal only (open if filtered, else closed)
    2) Stage 2 model over all rows (no stage 1 gate)
    3) Full two-stage pipeline
    4) Stage 2 performance on uncertain subset only
    """
    if model.mode != "two-stage":
        print("\nStage breakdown is only available for two-stage mode.")
        return

    print("\n" + "=" * 60)
    print("STAGE CONTRIBUTION BREAKDOWN")
    print("=" * 60)

    y_true = eval_df["open"].values
    stage1_mask = model.stage1_filter(eval_df)
    uncertain_mask = ~stage1_mask

    print(f"Stage 1 filtered-open coverage: {stage1_mask.mean():.3f} ({stage1_mask.sum()}/{len(eval_df)})")
    if stage1_mask.sum() > 0:
        stage1_open_precision = eval_df.loc[stage1_mask, "open"].mean()
        leaked_closed = int((eval_df.loc[stage1_mask, "open"] == 0).sum())
        print(f"Stage 1 precision within filtered-open: {stage1_open_precision:.3f}")
        print(f"True closed leaked into stage 1 open bucket: {leaked_closed}")

    # 1) Stage 1 signal only: open if filtered, else closed
    stage1_only_pred = stage1_mask.astype(int).values
    evaluate_model(y_true, stage1_only_pred, "Stage 1 signal only")

    # 2) Stage 2 model only on all rows (no gate)
    X_all = model.extract_features(eval_df)
    stage2_all_prob = model.model.predict_proba(X_all)[:, 1]
    stage2_all_pred = (stage2_all_prob >= decision_threshold).astype(int)
    evaluate_model(y_true, stage2_all_pred, "Stage 2 only (no stage 1 gate)")

    # 3) Full two-stage (using thresholded probabilities for parity)
    full_prob = model.predict_proba(eval_df)[:, 1]
    full_pred = (full_prob >= decision_threshold).astype(int)
    evaluate_model(y_true, full_pred, "Full two-stage pipeline")

    # 4) Stage 2 on uncertain subset only
    if uncertain_mask.sum() > 0:
        X_uncertain = model.extract_features(eval_df.loc[uncertain_mask])
        y_uncertain = eval_df.loc[uncertain_mask, "open"].values
        stage2_uncertain_prob = model.model.predict_proba(X_uncertain)[:, 1]
        stage2_uncertain_pred = (stage2_uncertain_prob >= decision_threshold).astype(int)
        evaluate_model(y_uncertain, stage2_uncertain_pred, "Stage 2 on uncertain subset only")


def run_param_sweep(train_df: pd.DataFrame, val_df: pd.DataFrame, use_source_confidence: bool, use_interactions: bool) -> None:
    grid = [
        {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20},
        {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20},
        {"n_estimators": 300, "learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 20},
        {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20},
        {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 50},
    ]

    results = []
    for params in grid:
        model = LightGBMModel(mode="single", use_source_confidence=use_source_confidence, use_interactions=use_interactions)
        X_train = model.extract_features(train_df)
        y_train = train_df["open"]
        X_val = model.extract_features(val_df)
        y_val = val_df["open"]

        clf = LGBMClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            min_child_samples=params["min_child_samples"],
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            class_weight="balanced",
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        metrics = evaluate_model(y_val, preds, "sweep")
        results.append({**params, **metrics})

    results = pd.DataFrame(results)
    print("\n=== LightGBM Hyperparameter Sweep (sorted by closed F1) ===")
    print(results.sort_values(["closed_f1", "closed_precision"], ascending=False).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGBM Model for Open/Closed Prediction")
    parser.add_argument("--mode", choices=["single", "two-stage"], default="two-stage",
                       help="Model mode: single-stage or two-stage")
    parser.add_argument("--confidence", choices=["none", "source"], default="none",
                       help="Confidence features: none or source")
    parser.add_argument("--no-interactions", action="store_true",
                       help="Disable manual interaction features")
    parser.add_argument("--split", choices=["val", "test"], default="test",
                       help="Evaluation split")
    parser.add_argument("--param-sweep", action="store_true",
                       help="Run a small hyperparameter sweep (single-stage)")
    parser.add_argument("--pos-weight", type=float, default=None,
                       help="scale_pos_weight for LightGBM (overrides class_weight)")
    parser.add_argument("--decision-threshold", type=float, default=0.5,
                       help="Decision threshold for class 1")
    parser.add_argument("--stage-breakdown", action="store_true",
                       help="Print stage-1 vs stage-2 contribution breakdown (two-stage only)")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "../../data")
    train_df = pd.read_parquet(os.path.join(data_dir, "train_split.parquet"))
    val_df = pd.read_parquet(os.path.join(data_dir, "val_split.parquet"))
    test_df = pd.read_parquet(os.path.join(data_dir, "test_split.parquet"))

    eval_df = test_df if args.split == "test" else val_df
    use_source_conf = args.confidence == "source"

    model = LightGBMModel(
        mode=args.mode,
        use_source_confidence=use_source_conf,
        use_interactions=not args.no_interactions,
    )

    if args.param_sweep:
        run_param_sweep(train_df, val_df, use_source_conf, not args.no_interactions)
        raise SystemExit(0)
    # override model init to allow scale_pos_weight if provided
    if args.pos_weight is not None:
        def _init_with_weight(self, scale_pos_weight=None):
            return LightGBMModel._init_model(self, scale_pos_weight=args.pos_weight)
        model._init_model = _init_with_weight.__get__(model, LightGBMModel)

    model.fit(train_df, val_df)

    # thresholded predictions
    probs = model.predict_proba(eval_df)[:, 1]
    predictions = (probs >= args.decision_threshold).astype(int)
    evaluate_model(eval_df["open"], predictions, model._get_variant_name())

    if args.stage_breakdown:
        print_stage_breakdown(model, eval_df, decision_threshold=args.decision_threshold)

    print("\nTop 10 Feature Importances:")
    importances = model.get_feature_importances()
    print(importances.head(10).to_string(index=False))

    conf_str = "source_conf" if use_source_conf else "no_conf"
    model_filename = f"lightgbm_{args.mode.replace('-', '_')}_{conf_str}.pkl"
    model_path = os.path.join(os.path.dirname(__file__), model_filename)
    model.save_model(model_path)
