"""
Unified Logistic Regression Model for Open/Closed Prediction

Supports 2 policy-compliant variants:
1. Single-stage (confidence-free)
2. Two-stage (confidence-free)

Usage:
    python logistic_regression_unified.py --mode single
    python logistic_regression_unified.py --mode two-stage

obviously open:
1. has_websites AND has_phones AND has_socials AND recency_days <= 180 days
2. has_brand AND num_websites >= 2 AND has_phones AND recency_days <= 730 days
3. num_sources >= 4 AND has_websites AND has_phones AND recency_days <= 180 days
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os
import argparse
try:
    from shared_evaluator import evaluate_predictions
    from shared_featurizer import SharedPlaceFeaturizer
except ImportError:  # pragma: no cover - package import fallback
    from .shared_evaluator import evaluate_predictions
    from .shared_featurizer import SharedPlaceFeaturizer


class UnifiedLogisticRegression:
    """
    Unified logistic regression model supporting single-stage and two-stage modes.
    """
    
    def __init__(
        self,
        mode: str = "two-stage",
        feature_bundle: str = "low_plus_medium",
        use_interactions: bool = True,
    ):
        """
        Initialize the model.
        
        Args:
            mode: "single" or "two-stage"
        """
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
        """Return a human-readable name for this variant."""
        mode_str = "Two-Stage" if self.mode == "two-stage" else "Single-Stage"
        return f"{mode_str} LR (Policy)"

    def stage1_filter(self, df: pd.DataFrame) -> pd.Series:
        """
        Stage 1: Rule-based filter to identify obviously open places.
        Returns: boolean mask where True = obviously open, False = uncertain
        
        Uses dataset diversity as the primary signal: places confirmed across
        multiple independent datasets (e.g. meta + msft) with rich contact info
        are very likely open.
        """
        # Dataset diversity: presence in multiple independent datasets
        has_multi_dataset = df['sources'].apply(
            lambda x: len(set(s.get('dataset', '') for s in x)) >= 2
        )
        
        has_websites = df['websites'].apply(lambda x: x is not None and len(x) > 0)
        has_phones = df['phones'].apply(lambda x: x is not None and len(x) > 0)
        has_socials = df['socials'].apply(lambda x: x is not None and len(x) > 0)
        has_many_sources = df['sources'].apply(lambda x: len(x) >= 3)
        
        obviously_open = (
            # Multi-dataset + triple contacts (strongest signal)
            (has_multi_dataset & has_websites & has_phones & has_socials) |
            
            # Many sources (3+) + multi-dataset + dual contacts
            (has_many_sources & has_multi_dataset & has_websites & has_phones)
        )
        
        return obviously_open

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self.featurizer.transform(df)
        self._feature_names = self.featurizer.feature_names
        return features

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """Train the model."""
        print(f"\n{'='*60}")
        print(f"Training: {self._get_variant_name()}")
        print(f"{'='*60}")
        
        if self.mode == "two-stage":
            self._fit_two_stage(train_df, val_df)
        else:
            self._fit_single_stage(train_df, val_df)
        
        return self
    
    def _fit_single_stage(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """Train single-stage model on all data."""
        X_train = self.extract_features(train_df)
        y_train = train_df['open']
        
        print(f"Training on {len(train_df)} samples:")
        print(f"  Open: {y_train.sum()} ({y_train.mean():.1%})")
        print(f"  Closed: {(~y_train.astype(bool)).sum()} ({(~y_train.astype(bool)).mean():.1%})")
        print(f"  Features: {X_train.shape[1]}")
        
        self.model = LogisticRegression(
            class_weight='balanced',
            random_state=42, 
            max_iter=1000
        )
        self.model.fit(X_train, y_train)
        print("Model trained.")
        
        if val_df is not None:
            self._print_validation_report(val_df)
    
    def _fit_two_stage(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """Train two-stage model: filter obvious cases, then train on uncertain."""
        # Stage 1
        obviously_open_mask = self.stage1_filter(train_df)
        
        print(f"Stage 1 Filter Results:")
        print(f"  Total training samples: {len(train_df)}")
        print(f"  Obviously open (filtered): {obviously_open_mask.sum()} ({obviously_open_mask.mean():.1%})")
        print(f"  Uncertain (to classify): {(~obviously_open_mask).sum()} ({(~obviously_open_mask).mean():.1%})")
        
        # Stage 1 accuracy check
        stage1_correct = train_df[obviously_open_mask]['open'].mean()
        print(f"  Stage 1 accuracy on 'obviously open': {stage1_correct:.3f}")
        print()
        
        # Stage 2 with uncertain cases
        uncertain_train = train_df[~obviously_open_mask].copy()
        
        if len(uncertain_train) > 0:
            X_uncertain = self.extract_features(uncertain_train)
            y_uncertain = uncertain_train['open']
            
            print(f"Stage 2 Training on {len(uncertain_train)} uncertain samples:")
            print(f"  Open: {y_uncertain.sum()} ({y_uncertain.mean():.1%})")
            print(f"  Closed: {(~y_uncertain.astype(bool)).sum()} ({(~y_uncertain.astype(bool)).mean():.1%})")
            print(f"  Features: {X_uncertain.shape[1]}")
            
            # Balanced class weight to improve closed recall while maintaining reasonable precision
            # Changed from {0: 3, 1: 1} to 'balanced' for automatic calculation
            self.model = LogisticRegression(
                class_weight='balanced',
                random_state=42, 
                max_iter=1000
            )
            self.model.fit(X_uncertain, y_uncertain)
            print("Stage 2 model trained.")
        else:
            print("No uncertain cases to train Stage 2 model on.")
        
        if val_df is not None:
            self._print_validation_report(val_df)
    
    def _print_validation_report(self, val_df: pd.DataFrame):
        """Print validation metrics."""
        y_val = val_df["open"]
        probs_open = self.predict_proba(val_df)[:, 1]
        predictions = (probs_open >= 0.5).astype(int)
        evaluate_predictions(y_val, predictions, y_score_open=probs_open, model_name="Validation Report")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        if self.mode == "two-stage":
            return self._predict_two_stage(df)
        else:
            return self._predict_single_stage(df)
    
    def _predict_single_stage(self, df: pd.DataFrame) -> np.ndarray:
        """Single-stage prediction."""
        X = self.extract_features(df)
        return self.model.predict(X)
    
    def _predict_two_stage(self, df: pd.DataFrame) -> np.ndarray:
        """Two-stage prediction."""
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
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        if self.mode == "two-stage":
            return self._predict_proba_two_stage(df)
        else:
            return self._predict_proba_single_stage(df)
    
    def _predict_proba_single_stage(self, df: pd.DataFrame) -> np.ndarray:
        """Single-stage probability prediction."""
        X = self.extract_features(df)
        return self.model.predict_proba(X)
    
    def _predict_proba_two_stage(self, df: pd.DataFrame) -> np.ndarray:
        """Two-stage probability prediction."""
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
        """Get feature importances from the logistic regression coefficients."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        importances = pd.DataFrame({
            'feature': self._feature_names,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': np.abs(self.model.coef_[0])
        })
        return importances.sort_values('abs_coefficient', ascending=False)

    def save_model(self, path: str):
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        model_data = {
            'model': self.model,
            'mode': self.mode,
            'feature_names': self._feature_names
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load the model from disk."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.mode = model_data['mode']
        self._feature_names = model_data['feature_names']
        print(f"Model loaded from {path}")


def evaluate_model(y_true, y_pred, model_name: str, y_score_open=None) -> dict:
    """Evaluate and print model performance."""
    return evaluate_predictions(y_true, y_pred, y_score_open=y_score_open, model_name=model_name)


def run_all_variants(train_df, val_df, test_df, feature_bundle: str = "low_plus_medium"):
    """Run both policy-compliant variants and compare results."""
    variants = [
        ("single", "Single-Stage LR (Policy)"),
        ("two-stage", "Two-Stage LR (Policy)"),
    ]
    
    results = []
    
    for mode, name in variants:
        model = UnifiedLogisticRegression(
            mode=mode,
            feature_bundle=feature_bundle,
        )
        model.fit(train_df, val_df=None)  # Suppress validation output for comparison
        
        probs_open = model.predict_proba(test_df)[:, 1]
        predictions = model.predict(test_df)
        result = evaluate_model(test_df['open'], predictions, name, y_score_open=probs_open)
        results.append(result)
    
    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY COMPARISON - ALL VARIANTS")
    print("=" * 90)
    print(f"{'Model':<40} {'Accuracy':<10} {'Closed Prec':<12} {'Closed Rec':<12} {'Closed F1':<10}")
    print("-" * 90)
    
    for result in results:
        print(f"{result['model']:<40} {result['accuracy']:<10.3f} {result['closed_precision']:<12.3f} {result['closed_recall']:<12.3f} {result['closed_f1']:<10.3f}")
    
    # Find best models
    print("\n" + "=" * 90)
    print("KEY INSIGHTS:")
    print("=" * 90)
    
    best_accuracy = max(results, key=lambda x: x['accuracy'])
    best_closed_precision = max(results, key=lambda x: x['closed_precision'])
    
    print(f"Best Accuracy: {best_accuracy['model']} ({best_accuracy['accuracy']:.3f})")
    print(f"Best Closed Precision: {best_closed_precision['model']} ({best_closed_precision['closed_precision']:.3f})")
    print(f"\nRecommended for production: {best_closed_precision['model']}")
    print("(Minimizes false closed predictions - best for traveler use cases)")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Logistic Regression Model")
    parser.add_argument("--mode", choices=["single", "two-stage"], default="two-stage",
                       help="Model mode: single-stage or two-stage")
    parser.add_argument("--compare-all", action="store_true",
                       help="Run single-stage and two-stage policy variants and compare")
    parser.add_argument("--split", choices=["val", "test"], default="test",
                       help="Evaluation split")
    parser.add_argument(
        "--feature-bundle",
        choices=["low_only", "low_plus_medium", "full_schema_native"],
        default="low_plus_medium",
        help="Feature bundle to enforce through shared featurizer",
    )
    args = parser.parse_args()
    
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    train_df = pd.read_parquet(os.path.join(data_dir, 'train_split.parquet'))
    val_df = pd.read_parquet(os.path.join(data_dir, 'val_split.parquet'))
    test_df = pd.read_parquet(os.path.join(data_dir, 'test_split.parquet'))
    
    eval_df = test_df if args.split == "test" else val_df
    
    if args.compare_all:
        run_all_variants(train_df, val_df, test_df, feature_bundle=args.feature_bundle)
    else:
        # Run single variant
        model = UnifiedLogisticRegression(
            mode=args.mode,
            feature_bundle=args.feature_bundle,
        )
        model.fit(train_df, val_df)
        
        # Evaluate on specified split
        probs_open = model.predict_proba(eval_df)[:, 1]
        predictions = (probs_open >= 0.5).astype(int)
        evaluate_model(eval_df['open'], predictions, model._get_variant_name(), y_score_open=probs_open)
        
        # Show feature importances
        print("\nTop 10 Feature Importances:")
        importances = model.get_feature_importances()
        print(importances.head(10).to_string(index=False))
        
        # Save model
        model_filename = f"lr_{args.mode.replace('-', '_')}_policy.pkl"
        model_path = os.path.join(os.path.dirname(__file__), model_filename)
        model.save_model(model_path)
