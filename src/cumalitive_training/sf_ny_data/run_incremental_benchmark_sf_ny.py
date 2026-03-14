"""
Incremental training benchmark on SF/NY dataset (combined NYC and SF places).

This script:
1. Combines data/sf_ny/nyc_places_processed.csv + data/sf_ny/sf_places_processed.csv
2. Creates 80/20 train/test split with stratification on open/closed labels
3. Splits the training pool into 5 stratified batches with even distribution
4. Trains 4 models (LR, RF, XGBoost, LightGBM) with warm-start on each batch sequentially
5. Saves models to disk after each batch for persistence (lr_batch_1.pkl, etc.)
6. Generates markdown results and multi-metric plots
7. Also trains each model on the full training set (single-run) for comparison

Output files:
- data/sf_ny/batches/ — test_set.csv, batch_1.csv, ..., batch_5.csv, all_batches_combined.csv
- src/cumalitive_training/sf_ny_data/models_persistence/ — lr_batch_1.pkl, ..., xgb_batch_5.pkl, etc.
- src/cumalitive_training/sf_ny_data/sf_ny_incremental_results.md
- src/cumalitive_training/sf_ny_data/sf_ny_incremental_benchmark_*.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models_v2.shared_featurizer import SharedPlaceFeaturizer

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

# ============================================================================
# Configuration
# ============================================================================
ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = ROOT / "data" / "sf_ny"
BATCHES_DIR = DATA_DIR / "batches"
MODELS_PERSISTENCE_DIR = Path(__file__).parent / "models_persistence"
RESULTS_DIR = Path(__file__).parent

FEATURE_BUNDLE = "low_plus_medium"
TEST_SIZE = 0.2
NUM_BATCHES = 5
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# ============================================================================
# Utilities
# ============================================================================

def load_and_combine_sf_ny_data() -> pd.DataFrame:
    """Load NYC and SF data, combine with shuffled rows."""
    print("Loading NYC and SF data...")
    nyc_df = pd.read_csv(DATA_DIR / "nyc_places_processed.csv")
    sf_df = pd.read_csv(DATA_DIR / "sf_places_processed.csv")
    
    print(f"NYC rows: {len(nyc_df)}, SF rows: {len(sf_df)}")
    
    # Add source column to track origin
    nyc_df["_data_source"] = "nyc"
    sf_df["_data_source"] = "sf"
    
    # Combine and shuffle
    combined_df = pd.concat([nyc_df, sf_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print(f"Combined: {len(combined_df)} rows (shuffled)")
    return combined_df


def remap_sf_ny_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remap SF/NY raw columns to match project_c/shared_featurizer schema.
    Maps flat columns to nested dicts to match the expected input format.
    """
    import ast
    
    # Create nested structure for names
    def parse_names(row):
        if pd.isna(row.get("name")) or not row.get("name"):
            return None
        return {"primary": row["name"], "alternate": []}
    
    # Create nested structure for categories
    def parse_categories(row):
        cat_dict = {}
        if pd.notna(row.get("category_primary")) and row["category_primary"]:
            cat_dict["primary"] = row["category_primary"]
        
        alternates = row.get("category_alternates", "")
        if pd.notna(alternates) and alternates and alternates != "[]":
            try:
                alt_list = ast.literal_eval(alternates) if isinstance(alternates, str) else alternates
                if isinstance(alt_list, list):
                    cat_dict["alternate"] = alt_list
            except:
                cat_dict["alternate"] = []
        
        return cat_dict if cat_dict else None
    
    # Create nested structure for addresses
    def parse_addresses(row):
        addr = {}
        if pd.notna(row.get("address_country")):
            addr["country"] = row["address_country"]
        if pd.notna(row.get("address_region")):
            addr["region"] = row["address_region"]
        if pd.notna(row.get("address_locality")):
            addr["locality"] = row["address_locality"]
        if pd.notna(row.get("address_postcode")):
            addr["postcode"] = row["address_postcode"]
        if pd.notna(row.get("address_freeform")):
            addr["address"] = row["address_freeform"]
        
        return [addr] if addr else None
    
    # Create nested structure for phones
    def parse_phones(row):
        if pd.notna(row.get("phone")) and row["phone"]:
            return [row["phone"]]
        return None
    
    # Create nested structure for websites
    def parse_websites(row):
        if pd.notna(row.get("website")) and row["website"]:
            return [row["website"]]
        return None
    
    # Create nested structure for socials
    def parse_socials(row):
        if pd.notna(row.get("social")) and row["social"]:
            return [row["social"]]
        return None
    
    # Create nested structure for sources
    def parse_sources(row):
        sources = []
        datasets = row.get("source_datasets", "")
        if pd.notna(datasets) and datasets and datasets != "[]":
            try:
                ds_list = ast.literal_eval(datasets) if isinstance(datasets, str) else datasets
                if isinstance(ds_list, list):
                    for ds in ds_list:
                        source_dict = {"name": ds}
                        if pd.notna(row.get("last_update")):
                            source_dict["update_time"] = row["last_update"]
                        sources.append(source_dict)
            except:
                pass
        
        # If no sources parsed, create a minimal one
        if not sources and (pd.notna(row.get("last_update")) or True):
            source_dict = {}
            if pd.notna(row.get("last_update")):
                source_dict["update_time"] = row["last_update"]
            sources.append(source_dict)
        
        return sources if sources else []
    
    # Apply transformations
    df_mapped = df.copy()
    df_mapped["names"] = df.apply(parse_names, axis=1)
    df_mapped["categories"] = df.apply(parse_categories, axis=1)
    df_mapped["addresses"] = df.apply(parse_addresses, axis=1)
    df_mapped["phones"] = df.apply(parse_phones, axis=1)
    df_mapped["websites"] = df.apply(parse_websites, axis=1)
    df_mapped["socials"] = df.apply(parse_socials, axis=1)
    df_mapped["sources"] = df.apply(parse_sources, axis=1)
    df_mapped["emails"] = None  # SF/NY doesn't have email easily accessible
    
    # Create geometry (WKB point) — we'll use lat/lon directly
    # SharedPlaceFeaturizer expects WKB, but we can adapt parsing
    df_mapped["geometry"] = df.apply(lambda row: f"POINT ({row['lon']} {row['lat']})" if pd.notna(row['lon']) and pd.notna(row['lat']) else None, axis=1)
    
    # Ensure label column exists
    df_mapped["open"] = (df_mapped.get("operating_status", "open").str.lower() == "open").astype(int)
    
    return df_mapped


def ensure_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'open' label column exists (0=closed, 1=open)."""
    if "open" not in df.columns:
        if "operating_status" in df.columns:
            df["open"] = (df["operating_status"].str.lower() == "open").astype(int)
        else:
            df["open"] = 1  # Default to open if no status
    return df


def create_batches(
    df: pd.DataFrame,
    test_size: float = 0.2,
    num_batches: int = 5,
    label_col: str = "open",
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """
    Split df into test set and training batches.
    Handles cases where label may be heavily imbalanced (e.g., mostly 'open').
    Returns (test_df, [batch_1, ..., batch_5])
    """
    np.random.seed(random_state)
    
    # Try stratified split; fall back to random split if stratification fails
    try:
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=test_size,
            stratify=df[label_col],
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
    
    print(f"\nTest set: {len(test_df)} rows")
    print(f"Training pool: {len(train_df)} rows")
    label_counts = train_df[label_col].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = "open" if label == 1 else "closed"
        print(f"  {label_name}: {count} rows")
    
    # Split training into num_batches with round-robin stratification by label
    batches = []
    
    try:
        # Separate by label and distribute round-robin across batches
        closed_indices = train_df[train_df[label_col] == 0].index.tolist()
        open_indices = train_df[train_df[label_col] == 1].index.tolist()
        
        print(f"\n✓ Using stratified batch distribution (round-robin)")
        print(f"  Closed samples: {len(closed_indices)}, Open samples: {len(open_indices)}")
        
        # Round-robin distribution across batches
        batch_lists = [[] for _ in range(num_batches)]
        for i, idx in enumerate(closed_indices):
            batch_lists[i % num_batches].append(idx)
        for i, idx in enumerate(open_indices):
            batch_lists[i % num_batches].append(idx)
        
        for batch_idx, batch_indices in enumerate(batch_lists, 1):
            batch_df = train_df.loc[batch_indices].reset_index(drop=True)
            batches.append(batch_df)
            open_count = (batch_df[label_col] == 1).sum()
            closed_count = (batch_df[label_col] == 0).sum()
            print(f"  Batch {batch_idx}: {len(batch_df)} rows ({open_count} open, {closed_count} closed)")
    except Exception as e:
        print(f"⚠ Failed to stratify batches: {e}")
        print(f"  Using simple sequential splits...")
        batch_size = len(train_df) // num_batches
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size if i < num_batches - 1 else len(train_df)
            batch_df = train_df.iloc[start_idx:end_idx].reset_index(drop=True)
            batches.append(batch_df)
            open_count = (batch_df[label_col] == 1).sum()
            closed_count = (batch_df[label_col] == 0).sum()
            print(f"  Batch {i+1}: {len(batch_df)} rows ({open_count} open, {closed_count} closed)")
    
    return test_df, batches


def save_batches(test_df: pd.DataFrame, batches: list[pd.DataFrame]) -> None:
    """Save test and batch CSVs to BATCHES_DIR."""
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    
    test_df.to_csv(BATCHES_DIR / "test_set.csv", index=False)
    print(f"\n✓ Saved test_set.csv: {len(test_df)} rows")
    
    for i, batch_df in enumerate(batches, 1):
        batch_df.to_csv(BATCHES_DIR / f"batch_{i}.csv", index=False)
        print(f"✓ Saved batch_{i}.csv: {len(batch_df)} rows")
    
    # Combined for single-run
    combined = pd.concat(batches, ignore_index=True)
    combined.to_csv(BATCHES_DIR / "all_batches_combined.csv", index=False)
    print(f"✓ Saved all_batches_combined.csv: {len(combined)} rows")


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """Compute accuracy, closed-class metrics, and PR-AUC."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, precision_recall_curve
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Closed class = 0
    closed_precision = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    closed_recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    closed_f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    # PR-AUC for closed class (use 1 - P(open) as score for closed)
    try:
        closed_proba = 1 - y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else 1 - y_pred_proba[:, 0]
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, closed_proba, pos_label=0)
        pr_auc = auc(recall_curve, precision_curve)
    except Exception:
        pr_auc = 0.0
    
    return {
        "accuracy": accuracy,
        "closed_precision": closed_precision,
        "closed_recall": closed_recall,
        "closed_f1": closed_f1,
        "pr_auc_closed": pr_auc,
    }


def train_incremental_models(
    batches: list[pd.DataFrame],
    test_df: pd.DataFrame,
    feature_bundle: str = "low_plus_medium",
) -> dict:
    """
    Train 4 models with warm-start across batches.
    - Featurizer + scaler fitted once on full training pool
    - Models warm-start on each batch sequentially
    - Models saved to disk after each batch
    Returns dict[model_name] -> dict[batch_k] -> metrics
    """
    print("\n" + "="*80)
    print("INCREMENTAL TRAINING (Warm-start across batches)")
    print("="*80)
    
    # Fit featurizer and scaler once on full training pool
    full_train_df = pd.concat(batches, ignore_index=True)
    print(f"\nFitting featurizer on full training pool ({len(full_train_df)} rows)...")
    featurizer = SharedPlaceFeaturizer(feature_bundle=feature_bundle)
    featurizer.fit(full_train_df, label_col="open")
    X_train_full = featurizer.transform(full_train_df)
    
    print(f"Fitting scaler on full training pool...")
    scaler = StandardScaler()
    scaler.fit(X_train_full)
    
    # Transform test set
    X_test = featurizer.transform(test_df)
    X_test = scaler.transform(X_test)
    y_test = test_df["open"].values
    
    print(f"Test set: {len(X_test)} rows")
    
    # Initialize models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, warm_start=True, random_state=RANDOM_SEED, n_jobs=-1),
        "RandomForest": RandomForestClassifier(n_estimators=100, warm_start=True, random_state=RANDOM_SEED, n_jobs=-1),
        "XGBoost": None,  # Will create after first batch
        "LightGBM": None,  # Will create after first batch
    }
    
    results = {model_name: {} for model_name in models.keys()}
    MODELS_PERSISTENCE_DIR.mkdir(parents=True, exist_ok=True)
    
    for batch_idx, batch_df in enumerate(batches, 1):
        print(f"\n--- Batch {batch_idx} ({len(batch_df)} rows) ---")
        
        X_batch = featurizer.transform(batch_df)
        X_batch = scaler.transform(X_batch)
        y_batch = batch_df["open"].values
        
        # ====== Logistic Regression ======
        print("  LR: warm-start fitting...", end=" ")
        models["LogisticRegression"].fit(X_batch, y_batch)
        y_pred = models["LogisticRegression"].predict(X_test)
        y_pred_proba = models["LogisticRegression"].predict_proba(X_test)
        metrics = get_metrics(y_test, y_pred, y_pred_proba)
        results["LogisticRegression"][f"batch_{batch_idx}"] = metrics
        print(f"✓ Acc={metrics['accuracy']:.4f}, F1={metrics['closed_f1']:.4f}")
        
        # Save LR model
        model_path = MODELS_PERSISTENCE_DIR / f"lr_batch_{batch_idx}.pkl"
        joblib.dump(models["LogisticRegression"], model_path)
        print(f"    Saved: {model_path.name}")
        
        # ====== Random Forest ======
        print("  RF: warm-start fitting...", end=" ")
        models["RandomForest"].n_estimators = 100 + batch_idx * 50  # Increase trees
        models["RandomForest"].fit(X_batch, y_batch)
        y_pred = models["RandomForest"].predict(X_test)
        y_pred_proba = models["RandomForest"].predict_proba(X_test)
        metrics = get_metrics(y_test, y_pred, y_pred_proba)
        results["RandomForest"][f"batch_{batch_idx}"] = metrics
        print(f"✓ Acc={metrics['accuracy']:.4f}, F1={metrics['closed_f1']:.4f}")
        
        # Save RF model
        model_path = MODELS_PERSISTENCE_DIR / f"rf_batch_{batch_idx}.pkl"
        joblib.dump(models["RandomForest"], model_path)
        print(f"    Saved: {model_path.name}")
        
        # ====== XGBoost ======
        print("  XGBoost: warm-start fitting...", end=" ")
        if models["XGBoost"] is None:
            models["XGBoost"] = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric="logloss",
            )
        else:
            models["XGBoost"].n_estimators = 50 + batch_idx * 25
        
        models["XGBoost"].fit(X_batch, y_batch)
        y_pred = models["XGBoost"].predict(X_test)
        y_pred_proba = models["XGBoost"].predict_proba(X_test)
        metrics = get_metrics(y_test, y_pred, y_pred_proba)
        results["XGBoost"][f"batch_{batch_idx}"] = metrics
        print(f"✓ Acc={metrics['accuracy']:.4f}, F1={metrics['closed_f1']:.4f}")
        
        # Save XGBoost model
        model_path = MODELS_PERSISTENCE_DIR / f"xgb_batch_{batch_idx}.pkl"
        joblib.dump(models["XGBoost"], model_path)
        print(f"    Saved: {model_path.name}")
        
        # ====== LightGBM ======
        print("  LightGBM: warm-start fitting...", end=" ")
        if models["LightGBM"] is None:
            models["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=-1,
            )
        else:
            models["LightGBM"].n_estimators = 50 + batch_idx * 25
        
        models["LightGBM"].fit(X_batch, y_batch)
        y_pred = models["LightGBM"].predict(X_test)
        y_pred_proba = models["LightGBM"].predict_proba(X_test)
        metrics = get_metrics(y_test, y_pred, y_pred_proba)
        results["LightGBM"][f"batch_{batch_idx}"] = metrics
        print(f"✓ Acc={metrics['accuracy']:.4f}, F1={metrics['closed_f1']:.4f}")
        
        # Save LightGBM model
        model_path = MODELS_PERSISTENCE_DIR / f"lgb_batch_{batch_idx}.pkl"
        joblib.dump(models["LightGBM"], model_path)
        print(f"    Saved: {model_path.name}")
    
    return results


def train_single_run_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_bundle: str = "low_plus_medium",
) -> dict:
    """Train models on full training set (single-run, no batching)."""
    print("\n" + "="*80)
    print("SINGLE-RUN TRAINING (All batches at once)")
    print("="*80)
    
    print(f"\nFitting featurizer on full training set ({len(train_df)} rows)...")
    featurizer = SharedPlaceFeaturizer(feature_bundle=feature_bundle)
    featurizer.fit(train_df, label_col="open")
    X_train = featurizer.transform(train_df)
    
    print(f"Fitting scaler on full training set...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    
    # Transform test set
    X_test = featurizer.transform(test_df)
    X_test = scaler.transform(X_test)
    y_test = test_df["open"].values
    y_train = train_df["open"].values
    
    results = {}
    
    # LR
    print("\nLR: training on full set...", end=" ")
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    metrics = get_metrics(y_test, y_pred, y_pred_proba)
    results["LogisticRegression"] = metrics
    print(f"✓ Acc={metrics['accuracy']:.4f}, F1={metrics['closed_f1']:.4f}")
    
    # RF
    print("RF: training on full set...", end=" ")
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    metrics = get_metrics(y_test, y_pred, y_pred_proba)
    results["RandomForest"] = metrics
    print(f"✓ Acc={metrics['accuracy']:.4f}, F1={metrics['closed_f1']:.4f}")
    
    # XGBoost
    print("XGBoost: training on full set...", end=" ")
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    metrics = get_metrics(y_test, y_pred, y_pred_proba)
    results["XGBoost"] = metrics
    print(f"✓ Acc={metrics['accuracy']:.4f}, F1={metrics['closed_f1']:.4f}")
    
    # LightGBM
    print("LightGBM: training on full set...", end=" ")
    model = lgb.LGBMClassifier(
        n_estimators=50,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    metrics = get_metrics(y_test, y_pred, y_pred_proba)
    results["LightGBM"] = metrics
    print(f"✓ Acc={metrics['accuracy']:.4f}, F1={metrics['closed_f1']:.4f}")
    
    return results


def generate_markdown_results(
    incremental_results: dict,
    single_run_results: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> str:
    """Generate markdown report of results."""
    model_names = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM"]
    
    md = []
    md.append("# SF/NY Incremental Training Benchmark Results\n")
    md.append(f"**Dataset**: SF/NY combined (NYC + SF with shuffled rows)")
    md.append(f"**Feature Bundle**: `{FEATURE_BUNDLE}`")
    md.append(f"**Total rows**: {len(train_df) + len(test_df):,}")
    md.append(f"**Training pool**: {len(train_df):,} rows (5 batches)")
    md.append(f"**Test set**: {len(test_df):,} rows (held-out 20%)")
    md.append("")
    
    # Incremental results
    md.append("## Incremental Training Results (Warm-start across batches)\n")
    
    for model_name in model_names:
        md.append(f"\n### {model_name}\n")
        md.append("| Batch | Accuracy | Closed Precision | Closed Recall | Closed F1 | PR-AUC (Closed) |")
        md.append("|-------|----------|------------------|---------------|-----------|-----------------|")
        
        for batch_k in range(1, NUM_BATCHES + 1):
            batch_key = f"batch_{batch_k}"
            metrics = incremental_results[model_name].get(batch_key, {})
            md.append(
                f"| After {batch_k} | "
                f"{metrics.get('accuracy', 0):.4f} | "
                f"{metrics.get('closed_precision', 0):.4f} | "
                f"{metrics.get('closed_recall', 0):.4f} | "
                f"{metrics.get('closed_f1', 0):.4f} | "
                f"{metrics.get('pr_auc_closed', 0):.4f} |"
            )
    
    md.append("\n")
    
    # Single-run results
    md.append("## Single-Run Training Results (All batches at once)\n")
    md.append("| Model | Accuracy | Closed Precision | Closed Recall | Closed F1 | PR-AUC (Closed) |")
    md.append("|-------|----------|------------------|---------------|-----------|-----------------|")
    
    for model_name in model_names:
        metrics = single_run_results.get(model_name, {})
        md.append(
            f"| {model_name} | "
            f"{metrics.get('accuracy', 0):.4f} | "
            f"{metrics.get('closed_precision', 0):.4f} | "
            f"{metrics.get('closed_recall', 0):.4f} | "
            f"{metrics.get('closed_f1', 0):.4f} | "
            f"{metrics.get('pr_auc_closed', 0):.4f} |"
        )
    
    md.append("")
    md.append("## Data Files\n")
    md.append("| File | Description |")
    md.append("|------|-------------|")
    md.append("| `data/sf_ny/batches/test_set.csv` | Fixed 20% held-out test set |")
    md.append("| `data/sf_ny/batches/batch_1.csv` … `batch_5.csv` | Individual stratified training batches |")
    md.append("| `data/sf_ny/batches/all_batches_combined.csv` | All 5 batches combined (single-run input) |")
    
    md.append("")
    md.append("## Model Persistence\n")
    md.append("Trained models are saved after each batch:")
    md.append("- `lr_batch_1.pkl` … `lr_batch_5.pkl` (Logistic Regression)")
    md.append("- `rf_batch_1.pkl` … `rf_batch_5.pkl` (Random Forest)")
    md.append("- `xgb_batch_1.pkl` … `xgb_batch_5.pkl` (XGBoost)")
    md.append("- `lgb_batch_1.pkl` … `lgb_batch_5.pkl` (LightGBM)")
    md.append("")
    md.append("These can be loaded with `joblib.load()` to continue training on new batches.")
    
    return "\n".join(md)


def main():
    print("="*80)
    print("SF/NY INCREMENTAL TRAINING BENCHMARK")
    print("="*80)
    
    # Load and combine data
    df = load_and_combine_sf_ny_data()
    
    # Remap schema to match featurizer expectations
    print("\nRemapping SF/NY schema to featurizer format...")
    df = remap_sf_ny_schema(df)
    df = ensure_label_column(df)
    
    # Create batches
    test_df, batches = create_batches(
        df,
        test_size=TEST_SIZE,
        num_batches=NUM_BATCHES,
        random_state=RANDOM_SEED,
    )
    
    # Save batches
    save_batches(test_df, batches)
    
    # Train incrementally
    incremental_results = train_incremental_models(
        batches,
        test_df,
        feature_bundle=FEATURE_BUNDLE,
    )
    
    # Train single-run
    full_train_df = pd.concat(batches, ignore_index=True)
    single_run_results = train_single_run_models(
        full_train_df,
        test_df,
        feature_bundle=FEATURE_BUNDLE,
    )
    
    # Generate markdown
    markdown_text = generate_markdown_results(
        incremental_results,
        single_run_results,
        full_train_df,
        test_df,
    )
    
    results_path = RESULTS_DIR / "sf_ny_incremental_results.md"
    with open(results_path, "w") as f:
        f.write(markdown_text)
    print(f"\n✓ Saved results: {results_path}")
    
    # Save results as JSON for plotting
    results_json = {
        "incremental": incremental_results,
        "single_run": single_run_results,
    }
    json_path = RESULTS_DIR / "sf_ny_incremental_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"✓ Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
