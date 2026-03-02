"""
Random forest model
- maybe can handle imbalanced data better?
- captures non-linear relationships and feature interactions
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
import argparse
try:
    from shared_evaluator import evaluate_predictions
    from shared_featurizer import SharedPlaceFeaturizer
except ImportError:  # pragma: no cover - package import fallback
    from .shared_evaluator import evaluate_predictions
    from .shared_featurizer import SharedPlaceFeaturizer

class RandomForestModel:
    def __init__(self, feature_bundle: str = "low_plus_medium"):
        self.model = None
        self._feature_names = None
        self.featurizer = SharedPlaceFeaturizer(
            feature_bundle=feature_bundle,
            use_source_confidence=False,
            use_interactions=True,
        )
    
    def extract_features(self, df):
        features = self.featurizer.transform(df)
        self._feature_names = self.featurizer.feature_names
        return features
    
    def fit(self, train_df, val_df=None):
        X_train = self.extract_features(train_df)
        y_train = train_df['open']
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            max_depth=10
        )
        self.model.fit(X_train, y_train)
        
        if val_df is not None:
            X_val = self.extract_features(val_df)
            y_val = val_df['open']
            y_val_proba_open = self.model.predict_proba(X_val)[:, 1]
            y_val_pred = self.model.predict(X_val)
            evaluate_predictions(y_val, y_val_pred, y_score_open=y_val_proba_open, model_name="Random Forest Validation")
        
        return self
    
    def predict(self, df):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        X = self.extract_features(df)
        return self.model.predict(X)

    def get_feature_importances(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return (
            pd.DataFrame(
                {
                    "feature": self._feature_names,
                    "importance": self.model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Forest Model")
    parser.add_argument(
        "--feature-bundle",
        choices=["low_only", "low_plus_medium", "full_schema_native"],
        default="low_plus_medium",
        help="Feature bundle to enforce through shared featurizer",
    )
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    train_df = pd.read_parquet(os.path.join(data_dir, 'train_split.parquet'))
    val_df = pd.read_parquet(os.path.join(data_dir, 'val_split.parquet'))
    test_df = pd.read_parquet(os.path.join(data_dir, 'test_split.parquet'))
    
    # training
    rf_model = RandomForestModel(feature_bundle=args.feature_bundle)
    rf_model.fit(train_df, val_df)
    
    # testing
    X_test = rf_model.extract_features(test_df)
    y_test_pred = rf_model.predict(test_df)
    y_test_proba_open = rf_model.model.predict_proba(X_test)[:, 1]
    y_test = test_df['open']
    evaluate_predictions(y_test, y_test_pred, y_score_open=y_test_proba_open, model_name="Random Forest Test")
