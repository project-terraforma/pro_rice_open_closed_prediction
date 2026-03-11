"""
Build synthetic batches for label-coverage simulation.

Splits train+val data into stratified fake "release" batches while preserving
label distribution and key segments per batch.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import json
import os


def extract_primary_category(categories_str):
    """Extract primary category from categories dict string."""
    try:
        if pd.isna(categories_str):
            return None
        if isinstance(categories_str, str):
            categories_dict = eval(categories_str)
        else:
            categories_dict = categories_str
        return categories_dict.get('primary', None)
    except:
        return None


def extract_primary_source(sources_str):
    """Extract primary dataset source from sources list."""
    try:
        if pd.isna(sources_str):
            return None
        if isinstance(sources_str, str):
            sources_list = eval(sources_str)
        else:
            sources_list = sources_str
        if sources_list and len(sources_list) > 0:
            return sources_list[0].get('dataset', 'unknown')
        return None
    except:
        return None


def build_sim_batches(
    df: pd.DataFrame,
    n_batches: int = 6,
    seed: int = 42,
    stratify_by_label: bool = True,
    stratify_by_segment: bool = True,
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    Split train+val data into synthetic batches.
    
    Args:
        df: Full dataset (train + val)
        n_batches: Number of batches to create
        seed: Random seed
        stratify_by_label: If True, keep label distribution stable across batches
        stratify_by_segment: If True, balance category/source distribution across batches
    
    Returns:
        Tuple of (list of batch dataframes, metadata dataframe)
    """
    np.random.seed(seed)
    df = df.copy().reset_index(drop=True)
    
    # Extract segment info
    df['category'] = df['categories'].apply(extract_primary_category)
    df['source'] = df['sources'].apply(extract_primary_source)
    
    # Fill missing segments
    df['category'] = df['category'].fillna('unknown')
    df['source'] = df['source'].fillna('unknown')
    
    print(f"\n{'='*60}")
    print(f"Building {n_batches} Synthetic Batches")
    print(f"{'='*60}")
    print(f"Total records: {len(df)}")
    print(f"Open: {df['open'].sum()} ({df['open'].mean():.1%})")
    print(f"Closed: {(~df['open'].astype(bool)).sum()} ({(~df['open'].astype(bool)).mean():.1%})")
    print(f"\nTop categories: {df['category'].value_counts().head(5).to_dict()}")
    print(f"Top sources: {df['source'].value_counts().head(5).to_dict()}")
    
    batches = []
    batch_metadata = []
    
    # Stratified batch creation
    if stratify_by_label and stratify_by_segment:
        # Split by label and segment
        strata = df.groupby(['open', 'category']).groups
        stratum_lists = [indices.tolist() for indices in strata.values()]
        
        # Shuffle within each stratum and distribute across batches
        all_indices = []
        for stratum_list in stratum_lists:
            np.random.shuffle(stratum_list)
            all_indices.extend(stratum_list)
        
        np.random.shuffle(all_indices)
    elif stratify_by_label:
        # Split by label only
        open_mask = df['open'].astype(bool)
        open_indices = np.where(open_mask)[0]
        closed_indices = np.where(~open_mask)[0]
        np.random.shuffle(open_indices)
        np.random.shuffle(closed_indices)
        all_indices = list(open_indices) + list(closed_indices)
    else:
        all_indices = df.index.tolist()
        np.random.shuffle(all_indices)
    
    # Distribute indices into batches
    batch_size = len(all_indices) // n_batches
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = len(all_indices) if i == n_batches - 1 else (i + 1) * batch_size
        batch_indices = all_indices[start_idx:end_idx]
        batch_df = df.iloc[batch_indices].copy()
        batch_df = batch_df.sort_index()
        
        batches.append(batch_df)
        
        # Metadata
        metadata = {
            'batch_id': i,
            'n_records': len(batch_df),
            'n_open': batch_df['open'].sum(),
            'pct_open': batch_df['open'].mean(),
            'top_category': batch_df['category'].value_counts().index[0] if len(batch_df) > 0 else None,
            'top_source': batch_df['source'].value_counts().index[0] if len(batch_df) > 0 else None,
        }
        batch_metadata.append(metadata)
    
    metadata_df = pd.DataFrame(batch_metadata)
    
    print(f"\nBatch Metadata:")
    print(metadata_df.to_string(index=False))
    
    return batches, metadata_df


def load_and_prepare_data(
    data_path: str,
    train_split_path: str,
    val_split_path: str = None,
) -> pd.DataFrame:
    """Load and combine train+val splits for simulation."""
    df = pd.read_csv(data_path)
    train_ids = pd.read_parquet(train_split_path)['id'].tolist()
    
    # Load val_ids from CSV if parquet doesn't exist
    if val_split_path and val_split_path.endswith('.parquet') and not os.path.exists(val_split_path):
        val_csv_path = val_split_path.replace('.parquet', '.csv')
        if os.path.exists(val_csv_path):
            val_ids = pd.read_csv(val_csv_path)['id'].tolist()
        else:
            val_ids = []
    elif val_split_path:
        val_ids = pd.read_parquet(val_split_path)['id'].tolist()
    else:
        val_ids = []
    
    sim_pool = df[df['id'].isin(train_ids + val_ids)].copy()
    print(f"Loaded simulation pool: {len(sim_pool)} records")
    return sim_pool


if __name__ == "__main__":
    # Example usage
    data_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/project_c_samples.csv"
    train_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/train_split.parquet"
    val_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/val_split.parquet"
    
    df = load_and_prepare_data(data_path, train_path, val_path)
    batches, metadata = build_sim_batches(df, n_batches=6, seed=42)
    
    print(f"\n{'='*60}")
    print(f"Batch Creation Complete")
    print(f"{'='*60}")
