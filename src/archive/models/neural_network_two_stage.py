"""
Two-Stage Neural Network Model for Open/Closed Prediction

This model uses a two-stage approach:
1. Stage 1: Rule-based filter to identify obviously open places
2. Stage 2: Neural network classifier for uncertain cases

No confidence features are used.
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import joblib
import os
import argparse


class TwoStageNeuralNetwork:
    """
    Two-stage neural network model for open/closed prediction.
    Stage 1: Rule-based filter for obviously open places
    Stage 2: Neural network for uncertain cases
    """
    
    def __init__(self, hidden_layer_sizes=(100, 50), random_state=42, decision_threshold=0.5):
        """
        Initialize the model.
        
        Args:
            hidden_layer_sizes: Tuple of hidden layer sizes for the neural network
            random_state: Random state for reproducibility
            decision_threshold: Threshold for classifying as open (default 0.5)
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.decision_threshold = decision_threshold
        self.model = None
        self.scaler = StandardScaler()
        self._feature_names = None
    
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
        
        No confidence features are used.
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
        max_update_times = pd.to_datetime(max_update_times, errors='coerce', utc=True)
        # Convert to timezone-naive
        max_update_times = max_update_times.dt.tz_localize(None)
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
            times = [pd.to_datetime(s.get('update_time'), errors='coerce', utc=True) for s in sources if s.get('update_time')]
            times = [t.tz_localize(None) if pd.notna(t) and t.tz is not None else t for t in times]
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
        features['single_source'] = (features['num_sources'] == 1).astype(int)
        
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
        
        # === MULTI-DATASET INTERACTION FEATURES ===
        features['multi_dataset_with_contacts'] = features['has_multiple_datasets'] * features['contact_diversity']
        features['single_source_no_socials'] = (features['single_source'] * (1 - features['has_socials'])).astype(int)
        
        self._feature_names = list(features.columns)
        return features

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """Train the model using two-stage approach."""
        print(f"\n{'='*60}")
        print(f"Training: Two-Stage Neural Network (No Confidence)")
        print(f"{'='*60}")
        
        # Stage 1
        obviously_open_mask = self.stage1_filter(train_df)
        
        print(f"Stage 1 Filter Results:")
        print(f"  Total training samples: {len(train_df)}")
        print(f"  Obviously open (filtered): {obviously_open_mask.sum()} ({obviously_open_mask.mean():.1%})")
        print(f"  Uncertain (to classify): {(~obviously_open_mask).sum()} ({(~obviously_open_mask).mean():.1%})")
        
        # Stage 1 accuracy check
        if obviously_open_mask.sum() > 0:
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
            
            # Scale features for neural network
            X_scaled = self.scaler.fit_transform(X_uncertain)
            
            # Train neural network with balanced class weights
            self.model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=self.random_state,
                early_stopping=False,  # Disable early stopping to ensure full training
                verbose=False
            )
            
            self.model.fit(X_scaled, y_uncertain)
            print("Stage 2 neural network trained.")
        else:
            print("No uncertain cases to train Stage 2 model on.")
        
        if val_df is not None:
            self._print_validation_report(val_df)
        
        return self
    
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
        """Make predictions using two-stage approach."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        predictions = np.ones(len(df))
        
        obviously_open_mask = self.stage1_filter(df)
        predictions[obviously_open_mask] = 1
        
        uncertain_mask = ~obviously_open_mask
        if uncertain_mask.sum() > 0:
            X_uncertain = self.extract_features(df[uncertain_mask])
            X_scaled = self.scaler.transform(X_uncertain)
            # Use probability threshold for better control
            proba = self.model.predict_proba(X_scaled)
            uncertain_predictions = (proba[:, 1] >= self.decision_threshold).astype(int)
            predictions[uncertain_mask] = uncertain_predictions
        
        return predictions.astype(int)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        proba = np.ones((len(df), 2))
        proba[:, 0] = 0.01
        proba[:, 1] = 0.99
        
        obviously_open_mask = self.stage1_filter(df)
        proba[obviously_open_mask, 0] = 0.01
        proba[obviously_open_mask, 1] = 0.99
        
        uncertain_mask = ~obviously_open_mask
        if uncertain_mask.sum() > 0:
            X_uncertain = self.extract_features(df[uncertain_mask])
            X_scaled = self.scaler.transform(X_uncertain)
            uncertain_proba = self.model.predict_proba(X_scaled)
            proba[uncertain_mask] = uncertain_proba
        
        return proba

    def save_model(self, path: str):
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'random_state': self.random_state,
            'decision_threshold': self.decision_threshold,
            'feature_names': self._feature_names
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load the model from disk."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.hidden_layer_sizes = model_data['hidden_layer_sizes']
        self.random_state = model_data['random_state']
        self.decision_threshold = model_data.get('decision_threshold', 0.5)
        self._feature_names = model_data['feature_names']
        print(f"Model loaded from {path}")


def evaluate_model(y_true, y_pred, model_name: str = "Two-Stage Neural Network") -> dict:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-Stage Neural Network Model")
    parser.add_argument("--hidden-layers", type=str, default="100,50",
                       help="Hidden layer sizes (comma-separated, e.g., '100,50')")
    parser.add_argument("--threshold", type=float, default=0.7,
                       help="Decision threshold for open classification (default 0.7 for better closed recall)")
    parser.add_argument("--split", choices=["val", "test"], default="test",
                       help="Evaluation split")
    args = parser.parse_args()
    
    # Parse hidden layers
    hidden_layers = tuple(map(int, args.hidden_layers.split(',')))
    
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), '../../data/project_c_samples')
    train_df = pd.read_parquet(os.path.join(data_dir, 'train_split.parquet'))
    val_df = pd.read_parquet(os.path.join(data_dir, 'val_split.parquet'))
    test_df = pd.read_parquet(os.path.join(data_dir, 'test_split.parquet'))
    
    eval_df = test_df if args.split == "test" else val_df
    
    # Train model
    model = TwoStageNeuralNetwork(hidden_layer_sizes=hidden_layers)
    model.fit(train_df, val_df)
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    predictions = model.predict(eval_df)
    evaluate_model(eval_df['open'], predictions, "Two-Stage Neural Network (No Confidence)")
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), "nn_two_stage_no_conf.pkl")
    model.save_model(model_path)
