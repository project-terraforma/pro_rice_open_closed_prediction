"""
Quick Label-Coverage Comparison: Fast baseline & model comparisons.

This version:
- Runs 2 seeds instead of 3 (faster)
- Reduces model types to LR + XGBoost (skip LightGBM)
- Runs 4 batches instead of 6 (maintains trend validation)
- Shows key results rapidly
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List

sys.path.insert(0, '/Users/claricepark/Desktop/pro_rice_open_closed_prediction/src/models_v2')

from build_sim_batches import build_sim_batches
from triage_policy import TriagePolicy, print_triage_summary
from label_store import LabelStore
from logistic_regression_v2 import UnifiedLogisticRegression
from xgboost_model_v2 import XGBoostModel


class ModelFactory:
    """Factory for creating models."""
    @staticmethod
    def create_model(model_type: str, mode: str = "two-stage"):
        if model_type == "logistic_regression":
            return UnifiedLogisticRegression(mode=mode, feature_bundle="low_plus_medium")
        elif model_type == "xgboost":
            return XGBoostModel(mode=mode, feature_bundle="low_plus_medium")
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class QuickSimulation:
    """Simplified simulation for quick comparisons."""
    
    def __init__(self, test_ids: list, test_df: pd.DataFrame, model_type: str = "logistic_regression"):
        self.test_ids = set(test_ids)
        self.test_df = test_df.copy()
        self.model_type = model_type
        
        self.policy = TriagePolicy(t_closed_high=0.85, t_open_low=0.15)
        self.label_store = LabelStore()
        self.history = []
    
    def run_batch(self, batch_id: int, batch_df: pd.DataFrame, all_data: pd.DataFrame, strategy: str = "uncertainty") -> dict:
        """Run one batch."""
        training_data = self.label_store.get_training_data()
        labeled_ids = self.label_store.get_labeled_ids()
        
        # Build training set
        if len(training_data) > 0:
            train_set = all_data[all_data['id'].isin(training_data['id'])].copy()
            train_set = train_set.merge(training_data[['id', 'label', 'weight']], on='id')
        else:
            train_set = None
        
        # Train model
        model = ModelFactory.create_model(self.model_type, mode="two-stage")
        
        if train_set is not None and len(train_set) > 10:
            model.fit(train_set, val_df=None)
            predictions = model.predict_proba(batch_df)[:, 1]
            p_closed = 1.0 - predictions
        else:
            p_closed = np.random.uniform(0, 1, len(batch_df))
        
        # Triage
        routed_batch = self.policy.route_batch(batch_df.copy(), p_closed)
        
        # Review budget (5% per batch)
        review_budget = max(1, int(len(batch_df) * 0.05))
        
        # Strategy-specific selection
        if strategy == "random":
            candidates = routed_batch[routed_batch['route'].isin(['review', 'defer'])].copy()
            if len(candidates) > 0:
                review_ids_list = np.random.choice(candidates.index, size=min(review_budget, len(candidates)), replace=False)
                review_batch = routed_batch.loc[review_ids_list]
            else:
                review_batch = pd.DataFrame()
        
        elif strategy == "static":
            review_batch = pd.DataFrame()  # No retraining
        
        else:  # uncertainty
            candidates = routed_batch[routed_batch['route'] == 'review'].copy()
            if len(candidates) > 0:
                candidates['review_score'] = candidates['uncertainty'] * candidates['impact_score']
                candidates = candidates.sort_values('review_score', ascending=False)
                review_batch = candidates.head(review_budget)
            else:
                review_batch = pd.DataFrame()
        
        # Add gold labels
        if len(review_batch) > 0:
            review_ids = review_batch['id'].values
            review_labels = review_batch['open'].values
            self.label_store.add_gold_labels(ids=review_ids, labels=review_labels, confidence=1.0, source='manual_review')
        
        # Add silver labels
        auto_accept_mask = routed_batch['route'] == 'auto_accept'
        if auto_accept_mask.sum() > 0:
            auto_ids = routed_batch[auto_accept_mask]['id'].values
            auto_labels = routed_batch[auto_accept_mask]['auto_label'].values
            auto_confidences = routed_batch[auto_accept_mask]['p_closed'].apply(lambda p: max(p, 1 - p)).values
            self.label_store.add_silver_labels(ids=auto_ids, labels=auto_labels, confidence=auto_confidences, source='auto_high_confidence')
        
        # Evaluate on test
        test_eval = self._evaluate_on_test(model)
        
        # Auto-label precision
        auto_labels_df = routed_batch[routed_batch['route'] == 'auto_accept'].copy()
        auto_label_precision = (auto_labels_df['auto_label'] == auto_labels_df['open']).mean() if len(auto_labels_df) > 0 else None
        
        metrics = {
            'batch_id': batch_id,
            'strategy': strategy,
            'model_type': self.model_type,
            'labeled_cumulative': len(labeled_ids),
            'n_gold': len(review_batch),
            'n_silver': auto_accept_mask.sum() if auto_accept_mask.sum() > 0 else 0,
            'auto_precision': auto_label_precision,
            'test_pr_auc': test_eval.get('closed_pr_auc'),
        }
        
        self.history.append(metrics)
        return metrics
    
    def _evaluate_on_test(self, model) -> dict:
        """Evaluate on frozen test set."""
        try:
            test_predictions = model.predict_proba(self.test_df)[:, 1]
            p_closed = 1.0 - test_predictions
            y_true = self.test_df['open'].values
            
            from sklearn.metrics import auc, precision_recall_curve
            precision, recall, _ = precision_recall_curve(1 - y_true, p_closed)
            closed_pr_auc = auc(recall, precision)
            
            return {'closed_pr_auc': closed_pr_auc}
        except:
            return {}
    
    def get_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


def main():
    """Quick comparison run."""
    print("\n" + "="*80)
    print("QUICK LABEL-COVERAGE COMPARISON")
    print("="*80)
    
    # Load data
    data_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/project_c_samples.parquet"
    test_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/test_split.parquet"
    train_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/train_split.parquet"
    
    df = pd.read_parquet(data_path)
    test_df = pd.read_parquet(test_path)
    train_df = pd.read_parquet(train_path)
    
    train_ids = train_df['id'].tolist()
    val_ids_set = set()
    try:
        val_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/val_split.parquet"
        val_df = pd.read_parquet(val_path)
        val_ids_set = set(val_df['id'].tolist())
    except:
        pass
    
    sim_pool = df[df['id'].isin(train_ids + list(val_ids_set))].copy()
    print(f"Simulation pool: {len(sim_pool)} records")
    print(f"Test set: {len(test_df)} records (frozen)")
    
    # Run quick comparisons
    results = []
    strategies = ["uncertainty", "random", "static"]
    models = ["logistic_regression", "xgboost"]
    n_seeds = 2
    n_batches = 4
    
    test_ids = test_df['id'].tolist()
    
    for seed in range(n_seeds):
        print(f"\n{'='*80}")
        print(f"SEED {seed}")
        print(f"{'='*80}")
        
        batches, _ = build_sim_batches(sim_pool, n_batches=n_batches, seed=42 + seed)
        
        for model_type in models:
            for strategy in strategies:
                print(f"\n{model_type} + {strategy}:")
                try:
                    sim = QuickSimulation(test_ids, test_df, model_type)
                    
                    for batch_id, batch_df in enumerate(batches):
                        sim.run_batch(batch_id, batch_df, sim_pool, strategy)
                    
                    history = sim.get_history()
                    for _, row in history.iterrows():
                        row['seed'] = seed
                        results.append(row)
                    
                    # Quick summary
                    init_pr = history.iloc[0]['test_pr_auc']
                    final_pr = history.iloc[-1]['test_pr_auc']
                    improvement = (final_pr / init_pr - 1) * 100 if init_pr > 0 else 0
                    auto_prec = history['auto_precision'].mean()
                    
                    print(f"  Initial PR-AUC: {init_pr:.4f}")
                    print(f"  Final PR-AUC: {final_pr:.4f}")
                    print(f"  Improvement: {improvement:.1f}%")
                    print(f"  Avg Auto-Precision: {auto_prec:.3f}")
                    
                except Exception as e:
                    print(f"  ERROR: {e}")
    
    # Aggregate results
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print(f"AGGREGATED RESULTS")
    print(f"{'='*80}")
    
    # Save detailed results
    output_dir = Path("/Users/claricepark/Desktop/pro_rice_open_closed_prediction/artifacts/quick_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_dir / "quick_results.csv", index=False)
    
    # Summary by strategy & model
    summary = results_df.groupby(['strategy', 'model_type']).agg({
        'test_pr_auc': ['min', 'mean', 'max'],
        'auto_precision': 'mean',
        'n_gold': 'mean',
        'n_silver': 'mean',
    }).round(4)
    
    summary.to_csv(output_dir / "summary.csv")
    print("\nSummary by Strategy & Model:")
    print(summary)
    
    # Best performance per strategy
    best = results_df.loc[results_df.groupby('strategy')['test_pr_auc'].idxmax()]
    print("\n\nBest Model per Strategy (by final PR-AUC):")
    print(best[['strategy', 'model_type', 'seed', 'test_pr_auc']].to_string(index=False))
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
