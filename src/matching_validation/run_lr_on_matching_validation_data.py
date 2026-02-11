"""
WARNING: Matching-validation only. Not valid for open/closed prediction reporting.

This script is preserved for reference from earlier experimentation.
It runs a Two-Stage LR flow on `samples_3k_project_c_updated.csv`, where:
- label=1 means records match
- label=0 means records do not match
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../models"))

from logistic_regression import UnifiedLogisticRegression, evaluate_model


def parse_json_field(field):
    """Parse a JSON string field, return None if empty/invalid."""
    if pd.isna(field) or field == "" or field == "null":
        return None
    try:
        parsed = json.loads(field)
        if isinstance(parsed, (list, dict)) and len(parsed) == 0:
            return None
        return parsed
    except (json.JSONDecodeError, TypeError):
        return None


def load_and_prepare_data(csv_path):
    """Load CSV and prepare data for the model."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # NOTE: This mapping is preserved from earlier experiments, but this dataset is matching-label data.
    df["open"] = df["label"].astype(int)

    json_fields = ["sources", "names", "categories", "websites", "socials", "emails", "phones", "brand", "addresses"]

    print("\nParsing JSON fields...")
    for field in json_fields:
        if field in df.columns:
            df[field] = df[field].apply(parse_json_field)

    print("\nSample parsed data:")
    print(f"  sources: {len(df['sources'].iloc[0]) if df['sources'].iloc[0] else 0} sources")
    print(f"  websites: {df['websites'].iloc[0] if df['websites'].iloc[0] else []}")
    print(f"  phones: {df['phones'].iloc[0] if df['phones'].iloc[0] else []}")
    print(f"  brand: {df['brand'].iloc[0]}")

    print("\nClass distribution:")
    print(f"  Open (1): {df['open'].sum()} ({df['open'].mean():.1%})")
    print(f"  Closed (0): {(~df['open'].astype(bool)).sum()} ({(~df['open'].astype(bool)).mean():.1%})")

    return df


def split_data(df, train_ratio=0.7, val_ratio=0.15, random_state=42):
    """Split data into train/val/test sets."""
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=random_state, stratify=df["open"])
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_ratio_adjusted, random_state=random_state, stratify=temp_df["open"]
    )

    print("\nData split:")
    print(f"  Train: {len(train_df)} samples ({train_df['open'].mean():.1%} open)")
    print(f"  Val: {len(val_df)} samples ({val_df['open'].mean():.1%} open)")
    print(f"  Test: {len(test_df)} samples ({test_df['open'].mean():.1%} open)")

    return train_df, val_df, test_df


if __name__ == "__main__":
    print("WARNING: This script operates on matching-validation labels, not true open/closed labels.")

    csv_path = os.path.join(os.path.dirname(__file__), "../../data/matching_validation/samples_3k_project_c_updated.csv")
    df = load_and_prepare_data(csv_path)

    train_df, val_df, test_df = split_data(df, train_ratio=0.7, val_ratio=0.15, random_state=42)

    print("\n" + "=" * 60)
    print("TRAINING TWO-STAGE LR (NO CONFIDENCE) MODEL")
    print("=" * 60)

    model = UnifiedLogisticRegression(mode="two-stage", use_source_confidence=False)
    model.fit(train_df, val_df)

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    test_predictions = model.predict(test_df)
    results = evaluate_model(test_df["open"], test_predictions, "Two-Stage LR (No Confidence)")

    print("\n" + "=" * 60)
    print("TOP 15 FEATURE IMPORTANCES")
    print("=" * 60)
    importances = model.get_feature_importances()
    print(importances.head(15).to_string(index=False))

    print("\n" + "=" * 60)
    print("PREDICTION BREAKDOWN")
    print("=" * 60)

    stage1_mask = model.stage1_filter(test_df)
    print("\nStage 1 'Obviously Open' filter:")
    print(f"  Filtered as open: {stage1_mask.sum()} / {len(test_df)} ({stage1_mask.mean():.1%})")
    print(f"  Stage 1 accuracy: {test_df[stage1_mask]['open'].mean():.3f}")

    print("\nConfusion breakdown:")
    true_open = test_df["open"] == 1
    pred_open = test_predictions == 1

    tp = (true_open & pred_open).sum()
    tn = (~true_open & ~pred_open).sum()
    fp = (~true_open & pred_open).sum()
    fn = (true_open & ~pred_open).sum()

    print(f"  True Positives (correctly predicted open): {tp}")
    print(f"  True Negatives (correctly predicted closed): {tn}")
    print(f"  False Positives (predicted open, actually closed): {fp}")
    print(f"  False Negatives (predicted closed, actually open): {fn}")

    if fn > 0:
        print("\n⚠️ False Negatives (predicted closed, actually open):")
        fn_mask = true_open & ~pred_open
        fn_df = test_df[fn_mask]
        for _, row in fn_df.iterrows():
            name = row["names"].get("primary", "N/A") if row["names"] else "N/A"
            websites = len(row["websites"]) if row["websites"] else 0
            phones = len(row["phones"]) if row["phones"] else 0
            socials = len(row["socials"]) if row["socials"] else 0
            sources = len(row["sources"]) if row["sources"] else 0
            print(f"  - {name}: {sources} sources, {websites}W {phones}P {socials}S")

    if fp > 0:
        print("\n⚠️ False Positives (predicted open, actually closed):")
        fp_mask = ~true_open & pred_open
        fp_df = test_df[fp_mask]
        for _, row in fp_df.iterrows():
            name = row["names"].get("primary", "N/A") if row["names"] else "N/A"
            websites = len(row["websites"]) if row["websites"] else 0
            phones = len(row["phones"]) if row["phones"] else 0
            socials = len(row["socials"]) if row["socials"] else 0
            sources = len(row["sources"]) if row["sources"] else 0
            print(f"  - {name}: {sources} sources, {websites}W {phones}P {socials}S")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Open Recall: {results['open_recall']:.3f} (minimizes false 'closed' predictions)")
    print(f"Closed Recall: {results['closed_recall']:.3f} (detects actually closed places)")
    print(f"Closed Precision: {results['closed_precision']:.3f} (confidence in 'closed' predictions)")
