#!/usr/bin/env python3
"""Compare closed predictions across all models for each batch"""

import sys
import ast
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

ROOT = Path.cwd()
BATCHES_DIR = ROOT / "data" / "project_c_samples" / "batches"
SRC_V2 = ROOT / "src" / "models_v2"
sys.path.insert(0, str(SRC_V2))

from shared_featurizer import SharedPlaceFeaturizer

RANDOM_SEED = 42
FEATURE_BUNDLE = "low_plus_medium"

# HPO params
HPO_LR = {
    "C": 0.031066049133505004,
    "class_weight": {0: 4.5, 1: 1.0},
    "max_iter": 2000,
    "solver": "lbfgs",
}

HPO_XGB = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
}

HPO_LGB = {
    "class_weight": "balanced",
    "colsample_bytree": 0.9032350960341495,
    "learning_rate": 0.012106766445987193,
    "min_child_samples": 68,
    "n_estimators": 263,
    "num_leaves": 111,
    "reg_lambda": 0.5117430300313628,
    "subsample": 0.852665759648826,
}

HPO_RF = {
    "class_weight": "balanced",
    "max_depth": 8,
    "max_features": "log2",
    "min_samples_leaf": 7,
    "min_samples_split": 6,
    "n_estimators": 375,
}

# CSV parsing
_PARSE_COLS = ["sources", "names", "categories", "websites",
               "socials", "emails", "phones", "brand", "addresses"]
_LIST_COLS  = ["sources", "names", "websites", "socials", "emails", "phones", "addresses"]

def _parse_field(val):
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    if not isinstance(val, str):
        return val
    val = val.strip()
    if val in ("", "None", "null", "nan"):
        return None
    try:
        return ast.literal_eval(val)
    except Exception:
        return None

def parse_csv_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in _PARSE_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_field)
    for col in _LIST_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
    return df

# Load data
test_df = parse_csv_df(pd.read_csv(BATCHES_DIR / "test_set.csv"))
y_test = test_df["open"].values.astype(int)
n_closed_test = int((y_test == 0).sum())

print(f"Test set: {len(test_df)} rows | {n_closed_test} closed")
print("\n" + "="*100)
print("MODEL COMPARISON: Closed Predictions per Batch")
print("="*100)
print(f"{'Batch':<8} {'LR':<15} {'XGBoost':<15} {'LightGBM':<15} {'RF':<15}")
print("-" * 100)

cumulative = pd.DataFrame()
for bi in range(1, 6):
    batch = parse_csv_df(pd.read_csv(BATCHES_DIR / f"batch_{bi}.csv"))
    cumulative = pd.concat([cumulative, batch], ignore_index=True)
    
    # Featurize
    feat = SharedPlaceFeaturizer(
        feature_bundle=FEATURE_BUNDLE,
        use_source_confidence=False,
        use_interactions=True,
        auto_fit_on_transform=False,
    )
    feat.fit(cumulative, label_col="open")
    X_tr = feat.transform(cumulative)
    X_te = feat.transform(test_df)
    
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    
    y_cum = cumulative["open"].values.astype(int)
    
    # LR
    clf_lr = LogisticRegression(**HPO_LR, penalty="l2", random_state=RANDOM_SEED, n_jobs=-1)
    clf_lr.fit(X_tr, y_cum)
    pred_lr = clf_lr.predict(X_te)
    n_closed_lr = int((pred_lr == 0).sum())
    
    # XGBoost
    n_open, n_closed = int(y_cum.sum()), int((y_cum == 0).sum())
    spw = n_open / n_closed if n_closed > 0 else 1.0
    clf_xgb = XGBClassifier(**HPO_XGB, scale_pos_weight=spw, random_state=RANDOM_SEED, n_jobs=-1)
    clf_xgb.fit(X_tr, y_cum)
    pred_xgb = clf_xgb.predict(X_te)
    n_closed_xgb = int((pred_xgb == 0).sum())
    
    # LightGBM
    clf_lgb = LGBMClassifier(**HPO_LGB, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
    clf_lgb.fit(X_tr, y_cum)
    pred_lgb = clf_lgb.predict(X_te)
    n_closed_lgb = int((pred_lgb == 0).sum())
    
    # Random Forest
    clf_rf = RandomForestClassifier(**HPO_RF, random_state=RANDOM_SEED, n_jobs=-1)
    clf_rf.fit(X_tr, y_cum)
    pred_rf = clf_rf.predict(X_te)
    n_closed_rf = int((pred_rf == 0).sum())
    
    print(f"Batch {bi:<2} {n_closed_lr:<15} {n_closed_xgb:<15} {n_closed_lgb:<15} {n_closed_rf:<15}")

print("="*100)
print(f"Test set has {n_closed_test} true closed samples")
