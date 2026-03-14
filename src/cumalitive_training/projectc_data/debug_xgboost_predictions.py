#!/usr/bin/env python3
"""Debug script to understand XGBoost's closed-class predictions"""

import sys
import ast
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / "data" / "project_c_samples"
BATCHES_DIR = DATA_DIR / "batches"
SRC_V2      = ROOT / "src" / "models_v2"
sys.path.insert(0, str(SRC_V2))

from shared_featurizer import SharedPlaceFeaturizer

RANDOM_SEED = 42
FEATURE_BUNDLE = "low_plus_medium"

HPO_XGB = {
    "n_estimators":     300,
    "learning_rate":    0.05,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_lambda":       1.0,
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
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
batch_5_df = parse_csv_df(pd.read_csv(BATCHES_DIR / "batch_5.csv"))

print(f"Test set: {len(test_df)} rows | open={test_df['open'].sum()} | closed={(test_df['open']==0).sum()}")
print(f"Batch 5:  {len(batch_5_df)} rows | open={batch_5_df['open'].sum()} | closed={(batch_5_df['open']==0).sum()}")

# Featurize
feat = SharedPlaceFeaturizer(
    feature_bundle=FEATURE_BUNDLE,
    use_source_confidence=False,
    use_interactions=True,
    auto_fit_on_transform=False,
)
feat.fit(batch_5_df, label_col="open")
X_batch5 = feat.transform(batch_5_df)
X_test = feat.transform(test_df)

scaler = StandardScaler()
X_batch5 = scaler.fit_transform(X_batch5)
X_test = scaler.transform(X_test)

y_batch5 = batch_5_df["open"].values.astype(int)
y_test = test_df["open"].values.astype(int)

# Train XGBoost
n_open, n_closed = int(y_batch5.sum()), int((y_batch5 == 0).sum())
spw = n_open / n_closed if n_closed > 0 else 1.0

print(f"\nTraining XGBoost with scale_pos_weight={spw:.4f}")
clf = XGBClassifier(
    **HPO_XGB, scale_pos_weight=spw,
    random_state=RANDOM_SEED, n_jobs=-1,
    verbose=0,
)
clf.fit(X_batch5, y_batch5)

# Make predictions
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]  # P(open=1)
y_prob_closed = 1.0 - y_prob  # P(closed=0)

# Count predictions
n_pred_open = int(y_pred.sum())
n_pred_closed = int((y_pred == 0).sum())
n_true_closed = int((y_test == 0).sum())
n_true_open = int(y_test.sum())

print(f"\nTest set true labels: {n_true_open} open, {n_true_closed} closed")
print(f"Predictions: {n_pred_open} open, {n_pred_closed} closed")

# Closed class metrics
tp_closed = int(((y_test == 0) & (y_pred == 0)).sum())
fp_closed = int(((y_test == 1) & (y_pred == 0)).sum())
fn_closed = int(((y_test == 0) & (y_pred == 1)).sum())

print(f"\nClosed class (pos_label=0):")
print(f"  TP (correctly predicted closed): {tp_closed}")
print(f"  FP (incorrectly predicted closed): {fp_closed}")
print(f"  FN (closed but predicted open): {fn_closed}")

closed_prec = tp_closed / (tp_closed + fp_closed) if (tp_closed + fp_closed) > 0 else 0
closed_rec = tp_closed / (tp_closed + fn_closed) if (tp_closed + fn_closed) > 0 else 0

print(f"  Precision (TP / (TP+FP)): {closed_prec:.4f}")
print(f"  Recall (TP / (TP+FN)):    {closed_rec:.4f}")

# Verify with sklearn
prec_sklearn = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
rec_sklearn = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
print(f"  Sklearn precision: {prec_sklearn:.4f}")
print(f"  Sklearn recall:    {rec_sklearn:.4f}")

# Examine closed samples
closed_mask = y_test == 0
closed_indices = np.where(closed_mask)[0]

print(f"\nDetailed breakdown of {n_true_closed} closed test samples:")
pred_correct = (y_pred[closed_indices] == 0).sum()
pred_wrong = (y_pred[closed_indices] == 1).sum()
print(f"  Correctly predicted closed: {pred_correct} ({100*pred_correct/n_true_closed:.1f}%)")
print(f"  Incorrectly predicted open:  {pred_wrong} ({100*pred_wrong/n_true_closed:.1f}%)")

# Probability distributions
closed_probs_open = y_prob[closed_indices]
closed_probs_closed = y_prob_closed[closed_indices]

print(f"  P(open=1) stats for closed samples: min={closed_probs_open.min():.4f}, max={closed_probs_open.max():.4f}, mean={closed_probs_open.mean():.4f}")
print(f"  P(closed=0) stats for closed samples: min={closed_probs_closed.min():.4f}, max={closed_probs_closed.max():.4f}, mean={closed_probs_closed.mean():.4f}")

# Decision boundary
threshold = 0.5
print(f"\nDecision boundary at P(open) = {threshold}")
print(f"  Samples with P(open) >= {threshold}: {(y_prob >= threshold).sum()} → predicted open")
print(f"  Samples with P(open) < {threshold}: {(y_prob < threshold).sum()} → predicted closed")

# Show examples
wrong_mask = (y_pred == 1) & (y_test == 0)
if wrong_mask.sum() > 0:
    wrong_indices = np.where(wrong_mask)[0]
    print(f"\nExamples of closed samples mispredicted as open (n={len(wrong_indices)}):")
    for idx in wrong_indices[:5]:
        print(f"  Sample {idx}: P(open)={y_prob[idx]:.4f}, true=closed, pred=open")
else:
    print("\nNo closed samples mispredicted as open")

correct_mask = (y_pred == 0) & (y_test == 0)
if correct_mask.sum() > 0:
    correct_indices = np.where(correct_mask)[0]
    print(f"\nExamples of closed samples correctly predicted as closed (n={len(correct_indices)}):")
    for idx in correct_indices[:5]:
        print(f"  Sample {idx}: P(open)={y_prob[idx]:.4f}, true=closed, pred=closed")
