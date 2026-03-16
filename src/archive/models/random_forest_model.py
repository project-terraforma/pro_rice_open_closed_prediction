"""
Random forest model
- maybe can handle imbalanced data better?
- captures non-linear relationships and feature interactions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import joblib
import os

class RandomForestModel:
    def __init__(self):
        self.model = None
    
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
            y_val_pred = self.model.predict(X_val)
            print("Random Forest Validation Report:")
            prec_open = precision_score(y_val, y_val_pred, pos_label=1)
            rec_open = recall_score(y_val, y_val_pred, pos_label=1)
            f1_open = f1_score(y_val, y_val_pred, pos_label=1)
            prec_closed = precision_score(y_val, y_val_pred, pos_label=0)
            rec_closed = recall_score(y_val, y_val_pred, pos_label=0)
            f1_closed = f1_score(y_val, y_val_pred, pos_label=0)
            acc = accuracy_score(y_val, y_val_pred)
            print("Open: Precision: {:.3f}    Recall: {:.3f}    F1: {:.3f}".format(prec_open, rec_open, f1_open))
            print("Closed: Precision: {:.3f}    Recall: {:.3f}    F1: {:.3f}".format(prec_closed, rec_closed, f1_closed))
            print("Accuracy: {:.3f}".format(acc))
            print()
        
        return self
    
    def predict(self, df):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        X = self.extract_features(df)
        return self.model.predict(X)

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    train_df = pd.read_parquet(os.path.join(data_dir, 'train_split.parquet'))
    val_df = pd.read_parquet(os.path.join(data_dir, 'val_split.parquet'))
    test_df = pd.read_parquet(os.path.join(data_dir, 'test_split.parquet'))
    
    # training
    rf_model = RandomForestModel()
    rf_model.fit(train_df, val_df)
    
    # testing
    y_test_pred = rf_model.predict(test_df)
    y_test = test_df['open']
    print("Random Forest Test Report:")
    prec_open = precision_score(y_test, y_test_pred, pos_label=1)
    rec_open = recall_score(y_test, y_test_pred, pos_label=1)
    f1_open = f1_score(y_test, y_test_pred, pos_label=1)
    prec_closed = precision_score(y_test, y_test_pred, pos_label=0)
    rec_closed = recall_score(y_test, y_test_pred, pos_label=0)
    f1_closed = f1_score(y_test, y_test_pred, pos_label=0)
    acc = accuracy_score(y_test, y_test_pred)
    print("Open: Precision: {:.3f}    Recall: {:.3f}    F1: {:.3f}".format(prec_open, rec_open, f1_open))
    print("Closed: Precision: {:.3f}    Recall: {:.3f}    F1: {:.3f}".format(prec_closed, rec_closed, f1_closed))
    print("Accuracy: {:.3f}".format(acc))
