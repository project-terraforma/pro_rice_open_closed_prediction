"""
Unified Logistic Regression Model for Open/Closed Prediction

Supports 4 variants:
1. Single-stage, no confidence features
2. Single-stage, with source confidence features
3. Two-stage, no confidence features
4. Two-stage, with source confidence features

No global confidence features are used in any variant.

Usage:
    python logistic_regression_unified.py --mode single --confidence none
    python logistic_regression_unified.py --mode single --confidence source
    python logistic_regression_unified.py --mode two-stage --confidence none
    python logistic_regression_unified.py --mode two-stage --confidence source

obviously open:
1. has_websites AND has_phones AND has_socials AND recency_days <= 180 days
2. has_brand AND num_websites >= 2 AND has_phones AND recency_days <= 730 days
3. num_sources >= 4 AND has_websites AND has_phones AND recency_days <= 180 days
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import joblib
import os
import argparse


class UnifiedLogisticRegression:
    """
    Unified logistic regression model supporting single-stage and two-stage modes,
    with optional source confidence features.
    """
    
    def __init__(self, mode: str = "two-stage", use_source_confidence: bool = False):
        """
        Initialize the model.
        
        Args:
            mode: "single" or "two-stage"
            use_source_confidence: If True, include source-level confidence features
        """
        if mode not in ("single", "two-stage"):
            raise ValueError("mode must be 'single' or 'two-stage'")
        
        self.mode = mode
        self.use_source_confidence = use_source_confidence
        self.model = None
        self._feature_names = None
    
    def _get_variant_name(self) -> str:
        """Return a human-readable name for this variant."""
        conf_str = "with Source Confidence" if self.use_source_confidence else "No Confidence"
        mode_str = "Two-Stage" if self.mode == "two-stage" else "Single-Stage"
        return f"{mode_str} LR ({conf_str})"

    def stage1_filter(self, df: pd.DataFrame) -> pd.Series:
        """
        Stage 1: Rule-based filter to identify obviously open places.
        Returns: boolean mask where True = obviously open, False = uncertain
        
        Uses rich signals instead of global confidence scores.
        Balanced approach: stricter than original but not too aggressive.
        """
        # Calculate recency for temporal filtering
        max_update_times = df['sources'].apply(
            lambda x: max([d.get('update_time') for d in x if d.get('update_time')] or [None])
        )
        max_update_times = pd.to_datetime(max_update_times, errors='coerce')
        snapshot_time = max_update_times.max()
        
        if pd.notna(snapshot_time):
            recency_days = (snapshot_time - max_update_times).dt.days.fillna(9999)
        else:
            recency_days = pd.Series([9999] * len(df), index=df.index)
        
        # Data is "very fresh" if updated within 180 days (6 months)
        is_very_fresh = recency_days <= 180
        # Not too stale (within 2 years)
        is_not_stale = recency_days <= 730
        
        obviously_open = (
            # Triple contact methods with very fresh data (strongest signal)
            ((df['websites'].apply(lambda x: x is not None and len(x) > 0)) &
             (df['phones'].apply(lambda x: x is not None and len(x) > 0)) &
             (df['socials'].apply(lambda x: x is not None and len(x) > 0)) &
             is_very_fresh) |
            
            # Brand + multiple websites + phones + not stale
            ((df['brand'].apply(lambda x: x is not None)) &
             (df['websites'].apply(lambda x: x is not None and len(x) >= 2)) &
             (df['phones'].apply(lambda x: x is not None and len(x) > 0)) &
             is_not_stale) |
            
            # Many sources (4+) with dual contacts and fresh data
            ((df['sources'].apply(lambda x: len(x) >= 4)) & 
             (df['websites'].apply(lambda x: x is not None and len(x) > 0)) &
             (df['phones'].apply(lambda x: x is not None and len(x) > 0)) &
             is_very_fresh)
        )
        
        return obviously_open

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from the dataframe.
        
        Features are organized into groups:
        - Basic source & data features
        - Contact information features
        - Brand & category features
        - Address features
        - Name features
        - Temporal features
        - Dataset diversity features
        - Composite/interaction features
        - Source confidence features (optional)
        """
        features = pd.DataFrame(index=df.index)
        
        # === BASIC SOURCE & DATA FEATURES ===
        features['num_sources'] = df['sources'].apply(len)
        features['has_multiple_sources'] = (features['num_sources'] >= 2).astype(int)
        features['has_many_sources'] = (features['num_sources'] >= 3).astype(int)
        
        # === CONTACT INFORMATION FEATURES ===
        # Website features
        features['has_websites'] = df['websites'].apply(lambda x: 1 if x is not None and len(x) > 0 else 0)
        features['num_websites'] = df['websites'].apply(lambda x: len(x) if x is not None else 0)
        features['has_multiple_websites'] = (features['num_websites'] >= 2).astype(int)
        
        # Phone features
        features['has_phones'] = df['phones'].apply(lambda x: 1 if x is not None and len(x) > 0 else 0)
        features['num_phones'] = df['phones'].apply(lambda x: len(x) if x is not None else 0)
        features['has_multiple_phones'] = (features['num_phones'] >= 2).astype(int)
        
        # Social media features
        features['has_socials'] = df['socials'].apply(lambda x: 1 if x is not None and len(x) > 0 else 0)
        features['num_socials'] = df['socials'].apply(lambda x: len(x) if x is not None else 0)
        features['has_multiple_socials'] = (features['num_socials'] >= 2).astype(int)
        
        # === BRAND & CATEGORY FEATURES ===
        features['has_brand'] = df['brand'].apply(lambda x: 1 if x is not None else 0)
        features['has_primary_category'] = df['categories'].apply(lambda x: 1 if x and x.get('primary') else 0)
        
        # Extract category diversity
        def count_categories(categories):
            if not categories:
                return 0
            primary_count = 1 if categories.get('primary') else 0
            alternate_count = len(categories.get('alternate', [])) if categories.get('alternate') is not None else 0
            return primary_count + alternate_count
        
        features['num_categories'] = df['categories'].apply(count_categories)
        features['has_alternate_categories'] = df['categories'].apply(
            lambda x: 1 if x and x.get('alternate') is not None and len(x.get('alternate', [])) > 0 else 0
        )
        
        # === ADDRESS FEATURES ===
        features['num_addresses'] = df['addresses'].apply(lambda x: len(x) if x is not None else 0)
        features['has_addresses'] = (features['num_addresses'] > 0).astype(int)
        features['has_multiple_addresses'] = (features['num_addresses'] >= 2).astype(int)
        
        # Address completeness
        def address_completeness(addresses):
            if not addresses:
                return 0
            total_fields = 0
            filled_fields = 0
            for addr in addresses:
                if isinstance(addr, dict):
                    fields = ['country', 'region', 'locality', 'postcode', 'address']
                    total_fields += len(fields)
                    filled_fields += sum(1 for f in fields if addr.get(f))
            return filled_fields / total_fields if total_fields > 0 else 0
        
        features['address_completeness'] = df['addresses'].apply(address_completeness)
        
        # === NAME FEATURES ===
        features['has_name'] = df['names'].apply(lambda x: 1 if x and x.get('primary') else 0)
        features['name_length'] = df['names'].apply(lambda x: len(x['primary']) if x and x.get('primary') else 0)
        features['has_long_name'] = (features['name_length'] > 20).astype(int)
        features['has_short_name'] = (features['name_length'] <= 5).astype(int)
        
        # Name patterns (chains often have consistent naming)
        def has_chain_pattern(names):
            if not names or not names.get('primary'):
                return 0
            name = names['primary'].lower()
            chain_words = ['mcdonalds', 'starbucks', 'subway', 'pizza hut', 'kfc', 'burger king', 
                          'walmart', 'target', 'cvs', 'walgreens', 'shell', 'bp', 'chevron']
            return 1 if any(word in name for word in chain_words) else 0
        
        features['has_chain_pattern'] = df['names'].apply(has_chain_pattern)
        
        # === TEMPORAL FEATURES ===
        max_update_times = df['sources'].apply(lambda x: max([d.get('update_time') for d in x if d.get('update_time')] or [None]))
        max_update_times = pd.to_datetime(max_update_times, errors='coerce')
        snapshot_time = max_update_times.max()
        if pd.notna(snapshot_time):
            features['recency_days'] = (snapshot_time - max_update_times).dt.days.fillna(-1)
        else:
            features['recency_days'] = -1
        
        # Temporal feature engineering
        features['very_fresh'] = (features['recency_days'] <= 90).astype(int)  # Within 3 months
        features['fresh'] = ((features['recency_days'] > 90) & (features['recency_days'] <= 365)).astype(int)  # 3mo-1yr
        features['stale'] = (features['recency_days'] > 730).astype(int)  # Over 2 years
        features['very_stale'] = (features['recency_days'] > 1825).astype(int)  # Over 5 years
        
        # Source temporal diversity
        def temporal_diversity(sources):
            times = [pd.to_datetime(s.get('update_time'), errors='coerce') for s in sources if s.get('update_time')]
            times = [t for t in times if pd.notna(t)]
            if len(times) <= 1:
                return 0
            time_diffs = [(times[i] - times[0]).days for i in range(1, len(times))]
            return np.std(time_diffs) if time_diffs else 0
        
        features['source_temporal_diversity'] = df['sources'].apply(temporal_diversity)
        
        # === DATASET DIVERSITY FEATURES ===
        def dataset_diversity(sources):
            datasets = set()
            for source in sources:
                if source.get('dataset'):
                    datasets.add(source['dataset'])
            return len(datasets)
        
        features['num_datasets'] = df['sources'].apply(dataset_diversity)
        features['has_multiple_datasets'] = (features['num_datasets'] >= 2).astype(int)
        
        # === COMPOSITE COMPLETENESS FEATURES ===
        features['completeness_score'] = (
            features['has_websites'] + features['has_phones'] + 
            features['has_socials'] + features['has_addresses'] +
            features['has_brand'] + features['has_primary_category'] +
            features['has_name']
        )
        
        features['rich_profile'] = (features['completeness_score'] >= 5).astype(int)
        
        # Contact diversity
        features['contact_diversity'] = (
            features['has_websites'] + features['has_phones'] + features['has_socials']
        )
        features['has_full_contact_info'] = (features['contact_diversity'] == 3).astype(int)
        
        # === INTERACTION FEATURES ===
        features['brand_with_contacts'] = features['has_brand'] * features['contact_diversity']
        features['recent_with_contacts'] = features['very_fresh'] * features['contact_diversity']
        features['multiple_sources_with_contacts'] = features['has_multiple_sources'] * features['contact_diversity']
        
        # === SOURCE CONFIDENCE FEATURES (optional) ===
        if self.use_source_confidence:
            features['mean_source_conf'] = df['sources'].apply(
                lambda x: np.mean([d['confidence'] for d in x]) if len(x) > 0 else 0
            )
            features['max_source_conf'] = df['sources'].apply(
                lambda x: np.max([d['confidence'] for d in x]) if len(x) > 0 else 0
            )
            features['min_source_conf'] = df['sources'].apply(
                lambda x: np.min([d['confidence'] for d in x]) if len(x) > 0 else 0
            )
            features['source_conf_std'] = df['sources'].apply(
                lambda x: np.std([d['confidence'] for d in x]) if len(x) > 1 else 0
            )
            
            # Source confidence bins
            features['high_source_conf'] = (features['max_source_conf'] >= 0.90).astype(int)
            features['low_source_conf'] = (features['max_source_conf'] < 0.70).astype(int)
            
            # Interaction with source confidence
            features['high_conf_with_contacts'] = features['high_source_conf'] * features['contact_diversity']
        
        self._feature_names = list(features.columns)
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
        predictions = self.predict(val_df)
        y_val = val_df['open']
        
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
            'use_source_confidence': self.use_source_confidence,
            'feature_names': self._feature_names
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load the model from disk."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.mode = model_data['mode']
        self.use_source_confidence = model_data['use_source_confidence']
        self._feature_names = model_data['feature_names']
        print(f"Model loaded from {path}")


def evaluate_model(y_true, y_pred, model_name: str) -> dict:
    """Evaluate and print model performance."""
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
        'model': model_name,
        'open_precision': prec_open,
        'open_recall': rec_open,
        'open_f1': f1_open,
        'closed_precision': prec_closed,
        'closed_recall': rec_closed,
        'closed_f1': f1_closed,
        'accuracy': acc
    }


def run_all_variants(train_df, val_df, test_df):
    """Run all 4 variants and compare results."""
    variants = [
        ("single", False, "Single-Stage LR (No Confidence)"),
        ("single", True, "Single-Stage LR (Source Confidence)"),
        ("two-stage", False, "Two-Stage LR (No Confidence)"),
        ("two-stage", True, "Two-Stage LR (Source Confidence)"),
    ]
    
    results = []
    
    for mode, use_conf, name in variants:
        model = UnifiedLogisticRegression(mode=mode, use_source_confidence=use_conf)
        model.fit(train_df, val_df=None)  # Suppress validation output for comparison
        
        predictions = model.predict(test_df)
        result = evaluate_model(test_df['open'], predictions, name)
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
    parser.add_argument("--confidence", choices=["none", "source"], default="none",
                       help="Confidence features: none or source")
    parser.add_argument("--compare-all", action="store_true",
                       help="Run all 4 variants and compare")
    parser.add_argument("--split", choices=["val", "test"], default="test",
                       help="Evaluation split")
    args = parser.parse_args()
    
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    train_df = pd.read_parquet(os.path.join(data_dir, 'train_split.parquet'))
    val_df = pd.read_parquet(os.path.join(data_dir, 'val_split.parquet'))
    test_df = pd.read_parquet(os.path.join(data_dir, 'test_split.parquet'))
    
    eval_df = test_df if args.split == "test" else val_df
    
    if args.compare_all:
        run_all_variants(train_df, val_df, test_df)
    else:
        # Run single variant
        use_source_conf = args.confidence == "source"
        model = UnifiedLogisticRegression(mode=args.mode, use_source_confidence=use_source_conf)
        model.fit(train_df, val_df)
        
        # Evaluate on specified split
        predictions = model.predict(eval_df)
        evaluate_model(eval_df['open'], predictions, model._get_variant_name())
        
        # Show feature importances
        print("\nTop 10 Feature Importances:")
        importances = model.get_feature_importances()
        print(importances.head(10).to_string(index=False))
        
        # Save model
        conf_str = "source_conf" if use_source_conf else "no_conf"
        model_filename = f"lr_{args.mode.replace('-', '_')}_{conf_str}.pkl"
        model_path = os.path.join(os.path.dirname(__file__), model_filename)
        model.save_model(model_path)
