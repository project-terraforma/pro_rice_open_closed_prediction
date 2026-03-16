"""
Incremental Training Benchmark – Alex Filtered SF/NY Dataset
==============================================================================
Compares warm-start incremental training vs single-run training on alex-filtered
SF/NY dataset (combined NYC + SF with balanced open/closed labels).

Key differences from project_c:
  - Uses alex_nyc_filtered.csv + alex_sf_filtered.csv
  - Label: 'open' column (1=open, 0=closed)
  - Excludes: confidence, fsq_label, operating_status
  - Ensures balanced distribution: uses alex_city to split NYC/SF evenly across batches
  - Data already has nested structure (names, categories, addresses, etc.)

Training strategy:
  • Incremental: featurizer fitted once on full pool; models warm-start each batch
  • Single-run: fresh model trained on all batches at once
  • Models saved to disk after each batch (joblib.dump) for persistence

Output files:
  - data/sf_ny/batches/ — test_set.csv, batch_1.csv, ..., batch_5.csv, all_batches_combined.csv
  - src/cumulative_training/sf_ny_data/models_persistence/ — lr_batch_1.pkl, ..., lgb_batch_5.pkl
  - src/cumulative_training/sf_ny_data/sf_ny_alex_incremental_results.md
"""
from __future__ import annotations

import ast
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError("xgboost required: pip install xgboost") from e

try:
    from lightgbm import LGBMClassifier
except ImportError as e:
    raise ImportError("lightgbm required: pip install lightgbm") from e

try:
    import joblib
except ImportError as e:
    raise ImportError("joblib required: pip install joblib") from e

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT                  = Path(__file__).resolve().parents[3]
DATA_DIR              = ROOT / "data" / "sf_ny"
BATCHES_DIR           = DATA_DIR / "batches"
RESULTS_DIR           = Path(__file__).resolve().parent
MODELS_PERSISTENCE_DIR = RESULTS_DIR / "models_persistence"
SRC_V2                = ROOT / "src" / "models_v2"
sys.path.insert(0, str(SRC_V2))

from shared_featurizer import SharedPlaceFeaturizer  # noqa: E402

BATCHES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_PERSISTENCE_DIR.mkdir(parents=True, exist_ok=True)

# ── Global config ─────────────────────────────────────────────────────────────
RANDOM_SEED    = 42
TEST_FRACTION  = 0.20
N_BATCHES      = 5
FEATURE_BUNDLE = "low_plus_medium"

# ── HPO-sourced hyperparameters ───────────────────────────────────────────────
HPO_LR = {
    "C":            0.031066049133505004,
    "class_weight": {0: 4.5, 1: 1.0},
    "max_iter":     2000,
    "solver":       "lbfgs",
}

HPO_RF = {
    "class_weight":     "balanced",
    "max_depth":        8,
    "max_features":     "log2",
    "min_samples_leaf": 7,
    "min_samples_split": 6,
    "n_estimators":     375,
}

HPO_LGB = {
    "class_weight":    "balanced",
    "colsample_bytree": 0.9032350960341495,
    "learning_rate":   0.012106766445987193,
    "min_child_samples": 68,
    "n_estimators":    263,
    "num_leaves":      111,
    "reg_lambda":      0.5117430300313628,
    "subsample":       0.852665759648826,
}

HPO_XGB = {
    "n_estimators":    300,
    "learning_rate":   0.05,
    "max_depth":       6,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_lambda":      1.0,
    "objective":       "binary:logistic",
    "eval_metric":     "logloss",
}

# Number of boosting rounds / trees to add per incremental batch
TREES_PER_BATCH_XGB = HPO_XGB["n_estimators"]
TREES_PER_BATCH_LGB = HPO_LGB["n_estimators"]
TREES_PER_BATCH_RF  = HPO_RF["n_estimators"]


# ── CSV parsing / schema handling ─────────────────────────────────────────────
_PARSE_COLS = ["sources", "names", "categories", "websites",
               "socials", "emails", "phones", "brand", "addresses"]
_LIST_COLS  = ["sources", "names", "websites", "socials", "emails", "phones", "addresses"]


def _parse_field(val):
    """Parse nested dict/list from string representation."""
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
    """Parse nested structures from CSV columns."""
    df = df.copy()
    for col in _PARSE_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_field)
    return df


def load_and_combine_alex_sf_ny() -> pd.DataFrame:
    """Load NYC and SF alex-filtered data, combine with shuffled rows."""
    print("Loading alex-filtered NYC and SF data...")
    nyc_file = DATA_DIR / "alex_nyc_filtered.csv"
    sf_file = DATA_DIR / "alex_sf_filtered.csv"
    
    if not nyc_file.exists() or not sf_file.exists():
        raise FileNotFoundError(f"Missing files: {nyc_file} or {sf_file}")
    
    nyc_df = pd.read_csv(nyc_file)
    sf_df = pd.read_csv(sf_file)
    
    # Parse nested structures
    nyc_df = parse_csv_df(nyc_df)
    sf_df = parse_csv_df(sf_df)
    
    print(f"  NYC: {len(nyc_df):,} rows")
    print(f"  SF:  {len(sf_df):,} rows")
    
    # Combine and shuffle
    combined_df = pd.concat([nyc_df, sf_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print(f"  Combined: {len(combined_df):,} rows (shuffled)")
    
    # Check label distribution
    open_count = (combined_df["open"] == 1).sum()
    closed_count = (combined_df["open"] == 0).sum()
    print(f"  Label distribution: {open_count:,} open, {closed_count:,} closed")
    
    return combined_df


def clean_df_for_featurizer(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns not expected by featurizer, keep only necessary ones."""
    df = df.copy()
    
    # Columns to keep (expected by SharedPlaceFeaturizer)
    keep_cols = ["names", "categories", "addresses", "phones", "websites", "socials", "emails", "brand", "sources", "open", "geometry"]
    
    # Only keep columns that exist in the dataframe
    keep_cols = [c for c in keep_cols if c in df.columns]
    
    # If geometry is missing but we have lat/lon, create it
    if "geometry" not in df.columns:
        if "lon" in df.columns and "lat" in df.columns:
            df["geometry"] = df.apply(
                lambda row: f"POINT ({row['lon']} {row['lat']})" 
                if pd.notna(row.get('lon')) and pd.notna(row.get('lat')) 
                else None,
                axis=1
            )
            keep_cols.append("geometry")
    
    # Verify we have the essential columns
    if "open" not in keep_cols:
        raise ValueError("Missing 'open' label column")
    
    # Ensure all list/dict columns have proper defaults (empty list/dict instead of None)
    list_cols = ["names", "categories", "addresses", "phones", "websites", "socials", "emails", "brand", "sources"]
    for col in list_cols:
        if col in df.columns:
            # Replace None with appropriate default
            if col == "sources":
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
            elif col in ["names", "categories"]:
                df[col] = df[col].apply(lambda x: x if isinstance(x, dict) else {})
            elif col in ["addresses", "phones", "websites", "socials", "emails"]:
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
            elif col == "brand":
                df[col] = df[col].apply(lambda x: x if isinstance(x, dict) else {})
    
    return df[keep_cols]


# ── Batch Creation ────────────────────────────────────────────────────────────
def create_batches(
    df: pd.DataFrame,
    test_size: float = 0.2,
    num_batches: int = 5,
    label_col: str = "open",
    city_col: str = "alex_city",
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """
    Split df into test set and training batches.
    Ensures balanced distribution of NYC/SF across batches and test set.
    """
    np.random.seed(random_state)
    
    # Stratified split on both label AND city for balanced distribution
    print(f"Creating stratified train/test split (stratifying on {label_col} and {city_col})...")
    
    # Create a composite stratification key
    if city_col in df.columns:
        strat_col = df[label_col].astype(str) + "_" + df[city_col].astype(str)
    else:
        strat_col = df[label_col]
    
    try:
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=test_size,
            stratify=strat_col,
            random_state=random_state,
        )
        print("✓ Using stratified train/test split")
    except ValueError as e:
        print(f"⚠ Stratification failed ({str(e)}), falling back to random split...")
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=test_size,
            random_state=random_state,
        )
    
    test_df = df.loc[test_idx].reset_index(drop=True)
    train_df = df.loc[train_idx].reset_index(drop=True)
    
    print(f"\nTest set: {len(test_df):,} rows")
    if city_col in test_df.columns:
        nyc_test = (test_df[city_col].str.lower() == "nyc").sum()
        sf_test = (test_df[city_col].str.lower() == "sf").sum()
        print(f"  NYC: {nyc_test:,}, SF: {sf_test:,}")
    
    print(f"Training pool: {len(train_df):,} rows")
    label_counts = train_df[label_col].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = "open" if label == 1 else "closed"
        print(f"  {label_name}: {count:,} rows")
    
    # Create stratified batches with round-robin distribution
    print(f"\nCreating {num_batches} stratified batches (round-robin by label and city)...")
    
    # Separate by label and city
    if city_col in train_df.columns:
        closed_nyc_idx = train_df[(train_df[label_col] == 0) & (train_df[city_col].str.lower() == "nyc")].index.tolist()
        closed_sf_idx = train_df[(train_df[label_col] == 0) & (train_df[city_col].str.lower() == "sf")].index.tolist()
        open_nyc_idx = train_df[(train_df[label_col] == 1) & (train_df[city_col].str.lower() == "nyc")].index.tolist()
        open_sf_idx = train_df[(train_df[label_col] == 1) & (train_df[city_col].str.lower() == "sf")].index.tolist()
        
        print(f"  Closed-NYC: {len(closed_nyc_idx)}, Closed-SF: {len(closed_sf_idx)}")
        print(f"  Open-NYC: {len(open_nyc_idx)}, Open-SF: {len(open_sf_idx)}")
        
        # Distribute all four groups round-robin across batches
        batch_lists = [[] for _ in range(num_batches)]
        
        for i, idx in enumerate(closed_nyc_idx):
            batch_lists[i % num_batches].append(idx)
        for i, idx in enumerate(closed_sf_idx):
            batch_lists[i % num_batches].append(idx)
        for i, idx in enumerate(open_nyc_idx):
            batch_lists[i % num_batches].append(idx)
        for i, idx in enumerate(open_sf_idx):
            batch_lists[i % num_batches].append(idx)
    else:
        # Fallback: just use label-based stratification
        closed_idx = train_df[train_df[label_col] == 0].index.tolist()
        open_idx = train_df[train_df[label_col] == 1].index.tolist()
        
        batch_lists = [[] for _ in range(num_batches)]
        for i, idx in enumerate(closed_idx):
            batch_lists[i % num_batches].append(idx)
        for i, idx in enumerate(open_idx):
            batch_lists[i % num_batches].append(idx)
    
    batches = []
    for batch_idx, batch_indices in enumerate(batch_lists, 1):
        batch_df = train_df.loc[batch_indices].reset_index(drop=True)
        batches.append(batch_df)
        open_count = (batch_df[label_col] == 1).sum()
        closed_count = (batch_df[label_col] == 0).sum()
        print(f"  Batch {batch_idx}: {len(batch_df):,} rows ({open_count:,} open, {closed_count:,} closed)")
    
    return test_df, batches


def save_batches(test_df: pd.DataFrame, batches: list[pd.DataFrame]) -> None:
    """Save test and batch CSVs to BATCHES_DIR."""
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    
    test_df.to_csv(BATCHES_DIR / "test_set.csv", index=False)
    print(f"\n✓ Saved test_set.csv: {len(test_df):,} rows")
    
    for i, batch_df in enumerate(batches, 1):
        batch_df.to_csv(BATCHES_DIR / f"batch_{i}.csv", index=False)
        print(f"✓ Saved batch_{i}.csv: {len(batch_df):,} rows")
    
    # Combined for single-run
    combined = pd.concat(batches, ignore_index=True)
    combined.to_csv(BATCHES_DIR / "all_batches_combined.csv", index=False)
    print(f"✓ Saved all_batches_combined.csv: {len(combined):,} rows")


# ── Featurization ─────────────────────────────────────────────────────────────
def fit_featurizer_and_scaler(
    train_pool: pd.DataFrame,
) -> tuple[SharedPlaceFeaturizer, StandardScaler]:
    """Fit featurizer and scaler on full training pool."""
    print(f"\nFitting featurizer on full training pool ({len(train_pool):,} rows)...")
    feat = SharedPlaceFeaturizer(feature_bundle=FEATURE_BUNDLE)
    feat.fit(train_pool, label_col="open")
    
    X_pool = feat.transform(train_pool)
    print(f"  Transformed to {X_pool.shape[1]} features")
    
    print(f"Fitting scaler on full training pool...")
    scaler = StandardScaler()
    scaler.fit(X_pool)
    
    return feat, scaler


def transform_batch(
    batch_df: pd.DataFrame,
    feat: SharedPlaceFeaturizer,
    scaler: StandardScaler,
) -> np.ndarray:
    """Transform a single batch using the pre-fitted featurizer + scaler."""
    X = feat.transform(batch_df)
    return scaler.transform(X)


# ── Metrics ───────────────────────────────────────────────────────────────────
def get_metrics(y_true, y_pred, y_prob, label: str, model: str,
                n_train: int, total_rows_seen: int) -> dict:
    """
    y_prob is P(open=1).  All closed-class metrics use pos_label=0.
    """
    y_prob_closed = 1.0 - y_prob
    y_true_closed = (np.asarray(y_true) == 0).astype(int)
    return dict(
        model=model,
        label=label,
        n_train=n_train,
        total_rows_seen=total_rows_seen,
        accuracy=accuracy_score(y_true, y_pred),
        closed_precision=precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        closed_recall=recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        f1_closed=f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        pr_auc=average_precision_score(y_true_closed, y_prob_closed),
    )


def _evaluate(model_tag: str, clf, X_test: np.ndarray, y_test: np.ndarray,
              run_label: str, n_train: int, total_rows_seen: int) -> dict:
    """Predict on frozen test set and compute metrics."""
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    m = get_metrics(y_test, y_pred, y_prob, run_label, model_tag,
                    n_train, total_rows_seen)
    print(f"    ⤷ acc={m['accuracy']:.4f}  closed_prec={m['closed_precision']:.4f}  "
          f"closed_rec={m['closed_recall']:.4f}  f1_closed={m['f1_closed']:.4f}  "
          f"pr_auc={m['pr_auc']:.4f}")
    return m


# ═════════════════════════════════════════════════════════════════════════════
# INCREMENTAL RUNNERS  (warm-start / continued training)
# ═════════════════════════════════════════════════════════════════════════════

def run_incremental_lr(batches, X_batches, y_batches, X_test, y_test):
    """LR: warm_start=True — lbfgs resumes from previous coefficients."""
    results = []
    timings = []
    clf = LogisticRegression(
        C=HPO_LR["C"], class_weight=HPO_LR["class_weight"],
        max_iter=HPO_LR["max_iter"], solver=HPO_LR["solver"],
        penalty="l2", random_state=RANDOM_SEED, n_jobs=-1,
        warm_start=True,
    )
    total_seen = 0
    for bi in range(len(batches)):
        total_seen += len(batches[bi])
        n_open   = int(y_batches[bi].sum())
        n_closed = int((y_batches[bi] == 0).sum())
        print(f"  [LR] Training on batch {bi+1}: {len(batches[bi]):,} rows "
              f"(open={n_open:,}, closed={n_closed:,})  |  total seen so far: {total_seen:,}")
        
        start_time = time.time()
        clf.fit(X_batches[bi], y_batches[bi])
        elapsed = time.time() - start_time
        timings.append(elapsed)
        
        m = _evaluate("LR", clf, X_test, y_test, f"After batch {bi+1}",
                       len(batches[bi]), total_seen)
        results.append(m)
        
        # Save model after batch
        model_path = MODELS_PERSISTENCE_DIR / f"lr_batch_{bi+1}.pkl"
        joblib.dump(clf, model_path)
        print(f"      Saved: {model_path.name} ({elapsed:.2f}s)")
    
    return results, timings


def run_incremental_rf(batches, X_batches, y_batches, X_test, y_test):
    """RF: warm_start=True — adds trees each batch."""
    results = []
    timings = []
    clf = RandomForestClassifier(
        n_estimators=TREES_PER_BATCH_RF,
        class_weight=HPO_RF["class_weight"],
        max_depth=HPO_RF["max_depth"],
        max_features=HPO_RF["max_features"],
        min_samples_leaf=HPO_RF["min_samples_leaf"],
        min_samples_split=HPO_RF["min_samples_split"],
        random_state=RANDOM_SEED, n_jobs=-1,
        warm_start=True,
    )
    total_seen = 0
    for bi in range(len(batches)):
        total_seen += len(batches[bi])
        n_open   = int(y_batches[bi].sum())
        n_closed = int((y_batches[bi] == 0).sum())
        print(f"  [RF] Training on batch {bi+1}: {len(batches[bi]):,} rows "
              f"(open={n_open:,}, closed={n_closed:,})  |  total seen so far: {total_seen:,}")
        
        start_time = time.time()
        # Increase trees for each batch
        clf.n_estimators = TREES_PER_BATCH_RF * (bi + 1)
        clf.fit(X_batches[bi], y_batches[bi])
        elapsed = time.time() - start_time
        timings.append(elapsed)
        
        m = _evaluate("RF", clf, X_test, y_test, f"After batch {bi+1}",
                       len(batches[bi]), total_seen)
        results.append(m)
        
        # Save model after batch
        model_path = MODELS_PERSISTENCE_DIR / f"rf_batch_{bi+1}.pkl"
        joblib.dump(clf, model_path)
        print(f"      Saved: {model_path.name} ({elapsed:.2f}s)")
    
    return results, timings


def run_incremental_xgb(batches, X_batches, y_batches, X_test, y_test):
    """XGBoost: xgb_model= continues boosting from prior booster."""
    results = []
    timings = []
    prev_model = None
    total_seen = 0
    for bi in range(len(batches)):
        total_seen += len(batches[bi])
        n_open   = int(y_batches[bi].sum())
        n_closed = int((y_batches[bi] == 0).sum())
        spw = n_open / n_closed if n_closed > 0 else 1.0
        print(f"  [XGBoost] Training on batch {bi+1}: {len(batches[bi]):,} rows "
              f"(open={n_open:,}, closed={n_closed:,}, spw={spw:.2f})  |  "
              f"total seen so far: {total_seen:,}")

        start_time = time.time()
        clf = XGBClassifier(
            n_estimators=TREES_PER_BATCH_XGB,
            learning_rate=HPO_XGB["learning_rate"],
            max_depth=HPO_XGB["max_depth"],
            subsample=HPO_XGB["subsample"],
            colsample_bytree=HPO_XGB["colsample_bytree"],
            min_child_weight=HPO_XGB["min_child_weight"],
            reg_lambda=HPO_XGB["reg_lambda"],
            objective=HPO_XGB["objective"],
            eval_metric=HPO_XGB["eval_metric"],
            scale_pos_weight=spw,
            random_state=RANDOM_SEED, n_jobs=-1,
        )
        clf.fit(X_batches[bi], y_batches[bi], xgb_model=prev_model)
        elapsed = time.time() - start_time
        timings.append(elapsed)
        prev_model = clf
        
        m = _evaluate("XGBoost", clf, X_test, y_test, f"After batch {bi+1}",
                       len(batches[bi]), total_seen)
        results.append(m)
        
        # Save model after batch
        model_path = MODELS_PERSISTENCE_DIR / f"xgb_batch_{bi+1}.pkl"
        joblib.dump(clf, model_path)
        print(f"      Saved: {model_path.name} ({elapsed:.2f}s)")
    
    return results, timings


def run_incremental_lgb(batches, X_batches, y_batches, X_test, y_test):
    """LightGBM: continued boosting with init_model."""
    results = []
    timings = []
    prev_model = None
    total_seen = 0
    for bi in range(len(batches)):
        total_seen += len(batches[bi])
        n_open   = int(y_batches[bi].sum())
        n_closed = int((y_batches[bi] == 0).sum())
        spw = n_open / n_closed if n_closed > 0 else 1.0
        print(f"  [LightGBM] Training on batch {bi+1}: {len(batches[bi]):,} rows "
              f"(open={n_open:,}, closed={n_closed:,}, spw={spw:.2f})  |  "
              f"total seen so far: {total_seen:,}")

        start_time = time.time()
        clf = LGBMClassifier(
            n_estimators=TREES_PER_BATCH_LGB,
            class_weight="balanced",
            colsample_bytree=HPO_LGB["colsample_bytree"],
            learning_rate=HPO_LGB["learning_rate"],
            min_child_samples=HPO_LGB["min_child_samples"],
            num_leaves=HPO_LGB["num_leaves"],
            reg_lambda=HPO_LGB["reg_lambda"],
            subsample=HPO_LGB["subsample"],
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
        )
        clf.fit(X_batches[bi], y_batches[bi], init_model=prev_model)
        elapsed = time.time() - start_time
        timings.append(elapsed)
        prev_model = clf
        
        m = _evaluate("LightGBM", clf, X_test, y_test, f"After batch {bi+1}",
                       len(batches[bi]), total_seen)
        results.append(m)
        
        # Save model after batch
        model_path = MODELS_PERSISTENCE_DIR / f"lgb_batch_{bi+1}.pkl"
        joblib.dump(clf, model_path)
        print(f"      Saved: {model_path.name} ({elapsed:.2f}s)")
    
    return results, timings


# ═════════════════════════════════════════════════════════════════════════════
# SINGLE-RUN TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def run_single_lr(X_train, y_train, X_test, y_test):
    """LR: Train on full training set."""
    print(f"  [LR] Training on full set: {len(y_train):,} rows")
    start_time = time.time()
    clf = LogisticRegression(
        C=HPO_LR["C"], class_weight=HPO_LR["class_weight"],
        max_iter=HPO_LR["max_iter"], solver=HPO_LR["solver"],
        penalty="l2", random_state=RANDOM_SEED, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    elapsed = time.time() - start_time
    m = _evaluate("LR", clf, X_test, y_test, "Single-run",
                   len(y_train), len(y_train))
    print(f"      Completed in {elapsed:.2f}s")
    return [m], elapsed


def run_single_rf(X_train, y_train, X_test, y_test):
    """RF: Train on full training set."""
    print(f"  [RF] Training on full set: {len(y_train):,} rows")
    start_time = time.time()
    clf = RandomForestClassifier(
        n_estimators=HPO_RF["n_estimators"],
        class_weight=HPO_RF["class_weight"],
        max_depth=HPO_RF["max_depth"],
        max_features=HPO_RF["max_features"],
        min_samples_leaf=HPO_RF["min_samples_leaf"],
        min_samples_split=HPO_RF["min_samples_split"],
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    elapsed = time.time() - start_time
    m = _evaluate("RF", clf, X_test, y_test, "Single-run",
                   len(y_train), len(y_train))
    print(f"      Completed in {elapsed:.2f}s")
    return [m], elapsed


def run_single_xgb(X_train, y_train, X_test, y_test):
    """XGBoost: Train on full training set."""
    print(f"  [XGBoost] Training on full set: {len(y_train):,} rows")
    n_open   = int(y_train.sum())
    n_closed = int((y_train == 0).sum())
    spw = n_open / n_closed if n_closed > 0 else 1.0
    
    start_time = time.time()
    clf = XGBClassifier(
        n_estimators=HPO_XGB["n_estimators"],
        learning_rate=HPO_XGB["learning_rate"],
        max_depth=HPO_XGB["max_depth"],
        subsample=HPO_XGB["subsample"],
        colsample_bytree=HPO_XGB["colsample_bytree"],
        min_child_weight=HPO_XGB["min_child_weight"],
        reg_lambda=HPO_XGB["reg_lambda"],
        objective=HPO_XGB["objective"],
        eval_metric=HPO_XGB["eval_metric"],
        scale_pos_weight=spw,
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    elapsed = time.time() - start_time
    m = _evaluate("XGBoost", clf, X_test, y_test, "Single-run",
                   len(y_train), len(y_train))
    print(f"      Completed in {elapsed:.2f}s")
    return [m], elapsed


def run_single_lgb(X_train, y_train, X_test, y_test):
    """LightGBM: Train on full training set."""
    print(f"  [LightGBM] Training on full set: {len(y_train):,} rows")
    n_open   = int(y_train.sum())
    n_closed = int((y_train == 0).sum())
    spw = n_open / n_closed if n_closed > 0 else 1.0
    
    start_time = time.time()
    clf = LGBMClassifier(
        n_estimators=HPO_LGB["n_estimators"],
        class_weight="balanced",
        colsample_bytree=HPO_LGB["colsample_bytree"],
        learning_rate=HPO_LGB["learning_rate"],
        min_child_samples=HPO_LGB["min_child_samples"],
        num_leaves=HPO_LGB["num_leaves"],
        reg_lambda=HPO_LGB["reg_lambda"],
        subsample=HPO_LGB["subsample"],
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
    )
    clf.fit(X_train, y_train)
    elapsed = time.time() - start_time
    m = _evaluate("LightGBM", clf, X_test, y_test, "Single-run",
                   len(y_train), len(y_train))
    print(f"      Completed in {elapsed:.2f}s")
    return [m], elapsed


# ═════════════════════════════════════════════════════════════════════════════
# MARKDOWN GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_markdown(
    inc_results, inc_timings,
    single_results, single_timings,
    batches, test_df
):
    """Generate markdown report of results with timing information."""
    md = []
    
    md.append("# Alex-Filtered SF/NY Incremental Benchmark Results\n")
    md.append("## Setup\n")
    md.append(f"- **Dataset**: `data/sf_ny/alex_nyc_filtered.csv` + `data/sf_ny/alex_sf_filtered.csv`")
    md.append(f"- **Combined rows**: {sum(len(b) for b in batches) + len(test_df):,}")
    md.append(f"- **Test set**: {len(test_df):,} rows (20% stratified hold-out)")
    md.append(f"- **Training pool**: {sum(len(b) for b in batches):,} rows (5 stratified batches)")
    md.append(f"- **Feature bundle**: `{FEATURE_BUNDLE}`")
    md.append(f"- **Label column**: `open` (1=open, 0=closed)")
    md.append(f"- **Excluded**: confidence, fsq_label, operating_status")
    md.append(f"- **Balanced distribution**: stratified by open/closed and alex_city (NYC/SF)")
    md.append("")
    
    md.append("## Batch Sizes (Stratified)\n")
    md.append("| Batch | Rows | Open | Closed |")
    md.append("|-------|-----:|-----:|-------:|")
    for bi, batch in enumerate(batches, 1):
        open_cnt = (batch["open"] == 1).sum()
        closed_cnt = (batch["open"] == 0).sum()
        md.append(f"| {bi} | {len(batch):,} | {open_cnt:,} | {closed_cnt:,} |")
    
    open_test = (test_df["open"] == 1).sum()
    closed_test = (test_df["open"] == 0).sum()
    md.append(f"| **Test** | **{len(test_df):,}** | **{open_test:,}** | **{closed_test:,}** |")
    md.append("")
    
    md.append("## Incremental Training Results (Warm-start across batches)\n")
    
    for model_name in ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM"]:
        md.append(f"\n### {model_name}\n")
        md.append("| Batch | Accuracy | Closed Prec | Closed Rec | F1 (Closed) | PR-AUC | Time (s) |")
        md.append("|-------|----------|-------------|------------|-------------|--------|----------|")
        
        model_results = inc_results.get(model_name, [])
        model_timings = inc_timings.get(model_name, [])
        
        for bi, (r, t) in enumerate(zip(model_results, model_timings), 1):
            md.append(
                f"| After {bi} | "
                f"{r['accuracy']:.4f} | "
                f"{r['closed_precision']:.4f} | "
                f"{r['closed_recall']:.4f} | "
                f"{r['f1_closed']:.4f} | "
                f"{r['pr_auc']:.4f} | "
                f"{t:.2f} |"
            )
    
    md.append("\n")
    md.append("## Single-Run Training Results (All batches at once)\n")
    md.append("| Model | Accuracy | Closed Prec | Closed Rec | F1 (Closed) | PR-AUC | Time (s) |")
    md.append("|-------|----------|-------------|------------|-------------|--------|----------|")
    
    for model_name in ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM"]:
        r = single_results.get(model_name, {})
        t = single_timings.get(model_name, 0.0)
        md.append(
            f"| {model_name} | "
            f"{r.get('accuracy', 0):.4f} | "
            f"{r.get('closed_precision', 0):.4f} | "
            f"{r.get('closed_recall', 0):.4f} | "
            f"{r.get('f1_closed', 0):.4f} | "
            f"{r.get('pr_auc', 0):.4f} | "
            f"{t:.2f} |"
        )
    
    md.append("")
    md.append("## Timing Comparison (per-batch vs single-run)\n")
    md.append("| Model | Incremental (Total) | Single-run | Overhead Ratio |")
    md.append("|-------|-------------------:|----------:|---------------:|")
    
    for model_name in ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM"]:
        model_timings = inc_timings.get(model_name, [])
        single_time = single_timings.get(model_name, 0.0)
        total_inc = sum(model_timings)
        ratio = total_inc / single_time if single_time > 0 else 0
        md.append(
            f"| {model_name} | "
            f"{total_inc:.2f} | "
            f"{single_time:.2f} | "
            f"{ratio:.2f}x |"
        )
    
    md.append("")
    md.append("## Data Files\n")
    md.append("| File | Description |")
    md.append("|------|-------------|")
    md.append("| `data/sf_ny/batches/test_set.csv` | Fixed 20% hold-out test set |")
    md.append("| `data/sf_ny/batches/batch_1.csv` … `batch_5.csv` | Stratified training batches |")
    md.append("| `data/sf_ny/batches/all_batches_combined.csv` | All batches combined (single-run input) |")
    
    md.append("")
    md.append("## Model Persistence\n")
    md.append("Models saved after each batch for warm-start continuation:")
    md.append("- `lr_batch_1.pkl` … `lr_batch_5.pkl` (Logistic Regression)")
    md.append("- `rf_batch_1.pkl` … `rf_batch_5.pkl` (Random Forest)")
    md.append("- `xgb_batch_1.pkl` … `xgb_batch_5.pkl` (XGBoost)")
    md.append("- `lgb_batch_1.pkl` … `lgb_batch_5.pkl` (LightGBM)")
    md.append("")
    md.append("Load with: `model = joblib.load('lr_batch_5.pkl')` and continue training.")
    
    return "\n".join(md)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("ALEX-FILTERED SF/NY INCREMENTAL TRAINING BENCHMARK")
    print("=" * 80)
    
    # Load and combine data
    df = load_and_combine_alex_sf_ny()
    
    # Clean for featurizer (remove extra columns)
    df = clean_df_for_featurizer(df)
    
    # Create batches
    test_df, batches = create_batches(
        df,
        test_size=TEST_FRACTION,
        num_batches=N_BATCHES,
        label_col="open",
        city_col="alex_city" if "alex_city" in df.columns else None,
        random_state=RANDOM_SEED,
    )
    
    # Save batches
    save_batches(test_df, batches)
    
    # Fit featurizer and scaler once
    full_train_df = pd.concat(batches, ignore_index=True)
    feat, scaler = fit_featurizer_and_scaler(full_train_df)
    
    # Transform all data
    X_pool = feat.transform(full_train_df)
    X_test_raw = feat.transform(test_df)
    X_pool_scaled = scaler.transform(X_pool)
    X_test = scaler.transform(X_test_raw)
    
    # Transform batches
    X_batches = [transform_batch(b, feat, scaler) for b in batches]
    y_batches = [b["open"].values for b in batches]
    y_test = test_df["open"].values
    
    # Run incremental training
    print("\n" + "=" * 80)
    print("INCREMENTAL TRAINING (Warm-start)")
    print("=" * 80)
    
    inc_lr, time_lr = run_incremental_lr(batches, X_batches, y_batches, X_test, y_test)
    inc_rf, time_rf = run_incremental_rf(batches, X_batches, y_batches, X_test, y_test)
    inc_xgb, time_xgb = run_incremental_xgb(batches, X_batches, y_batches, X_test, y_test)
    inc_lgb, time_lgb = run_incremental_lgb(batches, X_batches, y_batches, X_test, y_test)
    
    inc_results = {
        "LogisticRegression": inc_lr,
        "RandomForest": inc_rf,
        "XGBoost": inc_xgb,
        "LightGBM": inc_lgb,
    }
    
    inc_timings = {
        "LogisticRegression": time_lr,
        "RandomForest": time_rf,
        "XGBoost": time_xgb,
        "LightGBM": time_lgb,
    }
    
    # Run single-run training
    print("\n" + "=" * 80)
    print("SINGLE-RUN TRAINING (Full training pool)")
    print("=" * 80)
    
    single_lr, time_single_lr = run_single_lr(X_pool_scaled, full_train_df["open"].values, X_test, y_test)
    single_rf, time_single_rf = run_single_rf(X_pool_scaled, full_train_df["open"].values, X_test, y_test)
    single_xgb, time_single_xgb = run_single_xgb(X_pool_scaled, full_train_df["open"].values, X_test, y_test)
    single_lgb, time_single_lgb = run_single_lgb(X_pool_scaled, full_train_df["open"].values, X_test, y_test)
    
    single_results = {
        "LogisticRegression": single_lr[0],
        "RandomForest": single_rf[0],
        "XGBoost": single_xgb[0],
        "LightGBM": single_lgb[0],
    }
    
    single_timings = {
        "LogisticRegression": time_single_lr,
        "RandomForest": time_single_rf,
        "XGBoost": time_single_xgb,
        "LightGBM": time_single_lgb,
    }
    
    # Generate markdown
    markdown_text = generate_markdown(
        inc_results, inc_timings,
        single_results, single_timings,
        batches, test_df
    )
    
    results_path = RESULTS_DIR / "sf_ny_alex_incremental_results.md"
    with open(results_path, "w") as f:
        f.write(markdown_text)
    print(f"\n✓ Results saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"✓ Models saved in: {MODELS_PERSISTENCE_DIR}")
    print(f"✓ Batches saved in: {BATCHES_DIR}")
    print(f"✓ Results in: {results_path}")


if __name__ == "__main__":
    main()
