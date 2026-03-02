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
import joblib
try:
    from shared_evaluator import evaluate_predictions
    from shared_featurizer import SharedPlaceFeaturizer
except ImportError:  # pragma: no cover - package import fallback
    from .shared_evaluator import evaluate_predictions
    from .shared_featurizer import SharedPlaceFeaturizer


class LightGBMModel:
    def __init__(
        self,
        mode: str = "two-stage",
        use_source_confidence: bool = False,
        use_interactions: bool = True,
        feature_bundle: str = "low_plus_medium",
    ):
        if mode not in ("single", "two-stage"):
            raise ValueError("mode must be 'single' or 'two-stage'")
        self.mode = mode
        self.use_source_confidence = use_source_confidence
        self.use_interactions = use_interactions
        self.feature_bundle = feature_bundle
        self.model = None
        self._feature_names = None
        self.featurizer = SharedPlaceFeaturizer(
            feature_bundle=feature_bundle,
            use_source_confidence=use_source_confidence,
            use_interactions=use_interactions,
        )

    def _get_variant_name(self) -> str:
        conf_str = "with Source Confidence" if self.use_source_confidence else "No Confidence"
        mode_str = "Two-Stage" if self.mode == "two-stage" else "Single-Stage"
        return f"{mode_str} LightGBM ({conf_str})"

    def stage1_filter(self, df: pd.DataFrame) -> pd.Series:
        """Rule-based filter aligned with LR two-stage policy for fair comparison."""
        has_multi_dataset = df["sources"].apply(
            lambda x: len(set(s.get("dataset", "") for s in x)) >= 2
        )
        has_websites = df["websites"].apply(lambda x: x is not None and len(x) > 0)
        has_phones = df["phones"].apply(lambda x: x is not None and len(x) > 0)
        has_socials = df["socials"].apply(lambda x: x is not None and len(x) > 0)
        has_many_sources = df["sources"].apply(lambda x: len(x) >= 3)

        obviously_open = (
            (has_multi_dataset & has_websites & has_phones & has_socials) |
            (has_many_sources & has_multi_dataset & has_websites & has_phones)
        )
        return obviously_open

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self.featurizer.transform(df)
        self._feature_names = self.featurizer.feature_names
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
        y_val = val_df["open"]
        probs_open = self.predict_proba(val_df)[:, 1]
        predictions = (probs_open >= 0.5).astype(int)
        evaluate_predictions(y_val, predictions, y_score_open=probs_open, model_name="Validation Report")

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


def evaluate_model(y_true, y_pred, model_name: str, y_score_open=None) -> dict:
    return evaluate_predictions(y_true, y_pred, y_score_open=y_score_open, model_name=model_name)


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


def run_param_sweep(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    use_source_confidence: bool,
    use_interactions: bool,
    feature_bundle: str,
) -> None:
    grid = [
        {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20},
        {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20},
        {"n_estimators": 300, "learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 20},
        {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20},
        {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 50},
    ]

    results = []
    for params in grid:
        model = LightGBMModel(
            mode="single",
            use_source_confidence=use_source_confidence,
            use_interactions=use_interactions,
            feature_bundle=feature_bundle,
        )
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
        probs_open = clf.predict_proba(X_val)[:, 1]
        preds = (probs_open >= 0.5).astype(int)
        metrics = evaluate_model(y_val, preds, "sweep", y_score_open=probs_open)
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
    parser.add_argument(
        "--feature-bundle",
        choices=["low_only", "low_plus_medium", "full_schema_native"],
        default="low_plus_medium",
        help="Feature bundle to enforce through shared featurizer",
    )
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
        feature_bundle=args.feature_bundle,
    )

    if args.param_sweep:
        run_param_sweep(
            train_df,
            val_df,
            use_source_conf,
            not args.no_interactions,
            feature_bundle=args.feature_bundle,
        )
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
    evaluate_model(eval_df["open"], predictions, model._get_variant_name(), y_score_open=probs)

    if args.stage_breakdown:
        print_stage_breakdown(model, eval_df, decision_threshold=args.decision_threshold)

    print("\nTop 10 Feature Importances:")
    importances = model.get_feature_importances()
    print(importances.head(10).to_string(index=False))

    conf_str = "source_conf" if use_source_conf else "no_conf"
    model_filename = f"lightgbm_{args.mode.replace('-', '_')}_{conf_str}.pkl"
    model_path = os.path.join(os.path.dirname(__file__), model_filename)
    model.save_model(model_path)
