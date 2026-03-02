"""XGBoost model for open/closed prediction (v2 shared-pipeline version)."""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError(
        "xgboost is required for xgboost_model_v2.py. Install it in your venv (pip install xgboost)."
    ) from exc

try:
    from shared_evaluator import evaluate_predictions
    from shared_featurizer import SharedPlaceFeaturizer
except ImportError:  # pragma: no cover - package import fallback
    from .shared_evaluator import evaluate_predictions
    from .shared_featurizer import SharedPlaceFeaturizer


class XGBoostModel:
    def __init__(
        self,
        mode: str = "single",
        feature_bundle: str = "low_plus_medium",
        use_interactions: bool = True,
    ):
        if mode not in ("single", "two-stage"):
            raise ValueError("mode must be 'single' or 'two-stage'")

        self.mode = mode
        self.feature_bundle = feature_bundle
        self.use_interactions = use_interactions
        self.model = None
        self._feature_names = None
        self.featurizer = SharedPlaceFeaturizer(
            feature_bundle=feature_bundle,
            use_source_confidence=False,
            use_interactions=use_interactions,
        )

    def _get_variant_name(self) -> str:
        mode_str = "Two-Stage" if self.mode == "two-stage" else "Single-Stage"
        return f"{mode_str} XGBoost (Policy)"

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
            (has_multi_dataset & has_websites & has_phones & has_socials)
            | (has_many_sources & has_multi_dataset & has_websites & has_phones)
        )
        return obviously_open

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self.featurizer.transform(df)
        self._feature_names = self.featurizer.feature_names
        return features

    def _init_model(self, scale_pos_weight: float | None = None, **overrides) -> XGBClassifier:
        params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
        if scale_pos_weight is not None:
            params["scale_pos_weight"] = scale_pos_weight
        params.update(overrides)
        return XGBClassifier(**params)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None, scale_pos_weight: float | None = None):
        print(f"\n{'='*60}")
        print(f"Training: {self._get_variant_name()}")
        print(f"{'='*60}")

        if self.mode == "two-stage":
            self._fit_two_stage(train_df, val_df, scale_pos_weight=scale_pos_weight)
        else:
            self._fit_single_stage(train_df, val_df, scale_pos_weight=scale_pos_weight)
        return self

    def _fit_single_stage(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        scale_pos_weight: float | None = None,
    ):
        X_train = self.extract_features(train_df)
        y_train = train_df["open"]

        print(f"Training on {len(train_df)} samples:")
        print(f"  Open: {y_train.sum()} ({y_train.mean():.1%})")
        print(f"  Closed: {(~y_train.astype(bool)).sum()} ({(~y_train.astype(bool)).mean():.1%})")
        print(f"  Features: {X_train.shape[1]}")

        self.model = self._init_model(scale_pos_weight=scale_pos_weight)
        self.model.fit(X_train, y_train)

        if val_df is not None:
            self._print_validation_report(val_df)

    def _fit_two_stage(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        scale_pos_weight: float | None = None,
    ):
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

            self.model = self._init_model(scale_pos_weight=scale_pos_weight)
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
            "feature_names": self._feature_names,
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")


def evaluate_model(y_true, y_pred, model_name: str, y_score_open=None) -> dict:
    return evaluate_predictions(y_true, y_pred, y_score_open=y_score_open, model_name=model_name)


def run_param_sweep(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_bundle: str,
):
    grid = [
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 1},
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 1},
        {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 1},
        {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 4, "min_child_weight": 1},
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 8, "min_child_weight": 2},
    ]

    results = []
    model = XGBoostModel(mode="single", feature_bundle=feature_bundle)
    X_train = model.extract_features(train_df)
    y_train = train_df["open"]
    X_val = model.extract_features(val_df)
    y_val = val_df["open"]

    for params in grid:
        clf = model._init_model(**params)
        clf.fit(X_train, y_train)
        probs_open = clf.predict_proba(X_val)[:, 1]
        preds = (probs_open >= 0.5).astype(int)
        metrics = evaluate_model(y_val, preds, "xgb_sweep", y_score_open=probs_open)
        results.append({**params, **metrics})

    results = pd.DataFrame(results)
    print("\n=== XGBoost Hyperparameter Sweep (sorted by closed F1) ===")
    print(results.sort_values(["closed_f1", "closed_precision"], ascending=False).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost Model v2")
    parser.add_argument("--mode", choices=["single", "two-stage"], default="single", help="Model mode")
    parser.add_argument(
        "--feature-bundle",
        choices=["low_only", "low_plus_medium", "full_schema_native"],
        default="low_plus_medium",
        help="Feature bundle to enforce through shared featurizer",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test", help="Evaluation split")
    parser.add_argument("--decision-threshold", type=float, default=0.5, help="Decision threshold for class 1")
    parser.add_argument("--param-sweep", action="store_true", help="Run a small hyperparameter sweep (single-stage)")
    parser.add_argument("--scale-pos-weight", type=float, default=None, help="XGBoost scale_pos_weight")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "../../data")
    train_df = pd.read_parquet(os.path.join(data_dir, "train_split.parquet"))
    val_df = pd.read_parquet(os.path.join(data_dir, "val_split.parquet"))
    test_df = pd.read_parquet(os.path.join(data_dir, "test_split.parquet"))

    eval_df = test_df if args.split == "test" else val_df

    if args.param_sweep:
        run_param_sweep(train_df, val_df, feature_bundle=args.feature_bundle)
        return

    model = XGBoostModel(mode=args.mode, feature_bundle=args.feature_bundle)
    model.fit(train_df, val_df, scale_pos_weight=args.scale_pos_weight)

    probs_open = model.predict_proba(eval_df)[:, 1]
    predictions = (probs_open >= args.decision_threshold).astype(int)
    evaluate_model(eval_df["open"], predictions, model._get_variant_name(), y_score_open=probs_open)

    print("\nTop 10 Feature Importances:")
    importances = model.get_feature_importances()
    print(importances.head(10).to_string(index=False))

    model_filename = f"xgboost_{args.mode.replace('-', '_')}_policy.pkl"
    model_path = os.path.join(os.path.dirname(__file__), model_filename)
    model.save_model(model_path)


if __name__ == "__main__":
    main()
