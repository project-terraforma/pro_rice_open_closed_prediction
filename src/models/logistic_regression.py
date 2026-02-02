"""
Two-Stage logistic regression model

- stage 1: filter obviously open places using high-confidence rules
- stage 2: apply logistic regression to uncertain cases
- goal is to reduce false closed predictions by being conservative
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import joblib
import os

class LogisticRegressionModel:
    def __init__(self):
        self.model = None
        self.features = None

    def stage1_filter(self, df):
        """
        Stage 1: Rule-based filter to identify obviously open places
        Returns: boolean mask where True = obviously open, False = uncertain
        """
        obviously_open = (
            # high confidence w/ multiple positive
            ((df['confidence'] >= 0.95) & 
             (df['websites'].apply(lambda x: x is not None and len(x) > 0)) &
             (df['phones'].apply(lambda x: x is not None and len(x) > 0))) |
            
            ((df['confidence'] >= 0.98) & 
             (df['sources'].apply(lambda x: len(x) >= 2))) |
            
            ((df['confidence'] >= 0.90) & 
             (df['brand'].apply(lambda x: x is not None))) |
            
            ((df['confidence'] >= 0.85) &
             (df['websites'].apply(lambda x: x is not None and len(x) > 0)) &
             (df['phones'].apply(lambda x: x is not None and len(x) > 0)) &
             (df['socials'].apply(lambda x: x is not None and len(x) > 0)))
        )
        
        return obviously_open

    def extract_features(self, df):
        features = pd.DataFrame()
        
        features['num_sources'] = df['sources'].apply(len)
        features['mean_source_conf'] = df['sources'].apply(lambda x: np.mean([d['confidence'] for d in x]) if len(x) > 0 else 0)
        features['max_source_conf'] = df['sources'].apply(lambda x: np.max([d['confidence'] for d in x]) if len(x) > 0 else 0)
        features['overall_confidence'] = df['confidence']
        features['has_websites'] = df['websites'].apply(lambda x: 1 if x is not None and len(x) > 0 else 0)
        features['has_phones'] = df['phones'].apply(lambda x: 1 if x is not None and len(x) > 0 else 0)
        features['has_socials'] = df['socials'].apply(lambda x: 1 if x is not None and len(x) > 0 else 0)
        features['has_brand'] = df['brand'].apply(lambda x: 1 if x is not None else 0)
        features['num_addresses'] = df['addresses'].apply(lambda x: len(x) if x is not None else 0)
        features['has_primary_category'] = df['categories'].apply(lambda x: 1 if x and x.get('primary') else 0)
        features['name_length'] = df['names'].apply(lambda x: len(x['primary']) if x and x.get('primary') else 0)
        
        max_update_times = df['sources'].apply(lambda x: max([d.get('update_time') for d in x if d.get('update_time')] or [None]))
        max_update_times = pd.to_datetime(max_update_times, errors='coerce')
        snapshot_time = max_update_times.max()
        if pd.notna(snapshot_time):
            features['recency_days'] = (snapshot_time - max_update_times).dt.days.fillna(-1)
        else:
            features['recency_days'] = -1
                
        conf = df['confidence']
        features['conf_very_high'] = (conf >= 0.95).astype(int)
        features['conf_high'] = ((conf >= 0.90) & (conf < 0.95)).astype(int)
        features['conf_medium'] = ((conf >= 0.80) & (conf < 0.90)).astype(int)
        features['conf_low'] = ((conf >= 0.65) & (conf < 0.80)).astype(int)
        features['conf_very_low'] = (conf < 0.65).astype(int)
        
        features['completeness_score'] = (
            features['has_websites'] + features['has_phones'] + 
            features['has_socials'] + (features['num_addresses'] > 0).astype(int)
        )
        
        features['very_fresh'] = (features['recency_days'] <= 180).astype(int)
        features['stale'] = (features['recency_days'] >= 1500).astype(int)
        
        features['min_source_conf'] = df['sources'].apply(lambda x: np.min([d['confidence'] for d in x]) if len(x) > 0 else 0)
        features['source_conf_std'] = df['sources'].apply(lambda x: np.std([d['confidence'] for d in x]) if len(x) > 1 else 0)
        
        return features

    def fit(self, train_df, val_df=None):
        X_train = self.extract_features(train_df)
        y_train = train_df['open']
        
        # stage 1
        obviously_open_mask = self.stage1_filter(train_df)
        
        print(f"Stage 1 Filter Results:")
        print(f"Total training samples: {len(train_df)}")
        print(f"Obviously open (filtered): {obviously_open_mask.sum()} ({obviously_open_mask.mean():.1%})")
        print(f"Uncertain (to classify): {(~obviously_open_mask).sum()} ({(~obviously_open_mask).mean():.1%})")
        
        # stage 1 accuracy check
        stage1_correct = train_df[obviously_open_mask]['open'].mean()
        print(f"Stage 1 accuracy on 'obviously open': {stage1_correct:.3f}")
        print()
        
        # stage 2 w/ uncertain cases
        uncertain_train = train_df[~obviously_open_mask].copy()
        
        if len(uncertain_train) > 0:
            X_uncertain = self.extract_features(uncertain_train)
            y_uncertain = uncertain_train['open']
            
            print(f"Stage 2 Training on {len(uncertain_train)} uncertain samples:")
            print(f"Open: {y_uncertain.sum()} ({y_uncertain.mean():.1%})")
            print(f"Closed: {(~y_uncertain.astype(bool)).sum()} ({(~y_uncertain.astype(bool)).mean():.1%})")
            
            # higher weight for closed class b/c we're using uncertain cases
            self.model = LogisticRegression(
                class_weight={0: 3, 1: 1},
                random_state=42, 
                max_iter=1000
            )
            self.model.fit(X_uncertain, y_uncertain)
            print("Stage 2 model trained.")
        else:
            print("No uncertain cases to train Stage 2 model on.")
        
        if val_df is not None:
            predictions = self.predict(val_df)
            y_val = val_df['open']
            print("Validation Report:")
            prec_open = precision_score(y_val, predictions, pos_label=1)
            rec_open = recall_score(y_val, predictions, pos_label=1)
            f1_open = f1_score(y_val, predictions, pos_label=1)
            prec_closed = precision_score(y_val, predictions, pos_label=0, zero_division=0)
            rec_closed = recall_score(y_val, predictions, pos_label=0, zero_division=0)
            f1_closed = f1_score(y_val, predictions, pos_label=0, zero_division=0)
            acc = accuracy_score(y_val, predictions)
            print("Open: Precision: {:.3f}    Recall: {:.3f}    F1: {:.3f}".format(prec_open, rec_open, f1_open))
            print("Closed: Precision: {:.3f}    Recall: {:.3f}    F1: {:.3f}".format(prec_closed, rec_closed, f1_closed))
            print("Accuracy: {:.3f}".format(acc))
            print()
        
        return self

    # predictions
    def predict(self, df):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        predictions = np.ones(len(df))
        
        obviously_open_mask = self.stage1_filter(df)
        predictions[obviously_open_mask] = 1
        
        uncertain_mask = ~obviously_open_mask
        if uncertain_mask.sum() > 0:
            X_uncertain = self.extract_features(df[uncertain_mask])
            uncertain_predictions = self.model.predict(X_uncertain)
            predictions[uncertain_mask] = uncertain_predictions
        
        return predictions.astype(int)

    def predict_proba(self, df):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
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

    def save_model(self, path):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    train_df = pd.read_parquet(os.path.join(data_dir, 'train_split.parquet'))
    val_df = pd.read_parquet(os.path.join(data_dir, 'val_split.parquet'))
    test_df = pd.read_parquet(os.path.join(data_dir, 'test_split.parquet'))
    
    # train
    model = LogisticRegressionModel()
    model.fit(train_df, val_df)
    
    # eval test
    test_predictions = model.predict(test_df)
    y_test = test_df['open']
    print("Test Report:")
    prec_open = precision_score(y_test, test_predictions, pos_label=1)
    rec_open = recall_score(y_test, test_predictions, pos_label=1)
    f1_open = f1_score(y_test, test_predictions, pos_label=1)
    prec_closed = precision_score(y_test, test_predictions, pos_label=0, zero_division=0)
    rec_closed = recall_score(y_test, test_predictions, pos_label=0, zero_division=0)
    f1_closed = f1_score(y_test, test_predictions, pos_label=0, zero_division=0)
    acc = accuracy_score(y_test, test_predictions)
    print("Open: Precision: {:.3f}    Recall: {:.3f}    F1: {:.3f}".format(prec_open, rec_open, f1_open))
    print("Closed: Precision: {:.3f}    Recall: {:.3f}    F1: {:.3f}".format(prec_closed, rec_closed, f1_closed))
    print("Accuracy: {:.3f}".format(acc))
    
    model_path = os.path.join(os.path.dirname(__file__), 'logistic_regression_model.pkl')
    model.save_model(model_path)
    print(f"Model saved to {model_path}")