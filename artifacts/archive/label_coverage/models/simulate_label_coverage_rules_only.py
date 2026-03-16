#!/usr/bin/env python3
"""
Label Coverage Simulation for Rules-Only Baseline

This script runs the iterative active-labeling simulation using the Rules-Only
baseline model to improve closed-place prediction.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Add src to path  
sys.path.insert(0, '/Users/claricepark/Desktop/pro_rice_open_closed_prediction/src/models_v2')
sys.path.insert(0, '/Users/claricepark/Desktop/pro_rice_open_closed_prediction/src/archive/models')

from build_sim_batches import build_sim_batches, load_and_prepare_data
from triage_policy import TriagePolicy, print_triage_summary
from label_store import LabelStore
from shared_metrics import compute_metrics
from rules_only_baseline import score_row, RuleConfig, has_value, count_sources, max_update_time, max_source_conf


class RulesOnlyModel:
    """Wrapper for Rules-Only baseline to match model interface."""
    
    def __init__(self, config: RuleConfig = None):
        self.config = config or RuleConfig()
        self._fitted = False
    
    def fit(self, train_df, val_df=None):
        """Rules don't need training, but we'll mark as fitted."""
        self._fitted = True
        return self
    
    def predict_proba(self, df):
        """Predict probability of being OPEN (not closed)."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare features for rules
        df_features = df.copy()
        
        # Add derived features needed by rules
        df_features['sources_n'] = df_features['sources'].apply(count_sources)
        df_features['max_source_conf'] = df_features['sources'].apply(max_source_conf)
        df_features['max_update_time'] = df_features['sources'].apply(max_update_time)
        
        # Add contact signal features
        df_features['has_websites'] = df_features['websites'].apply(has_value)
        df_features['has_phones'] = df_features['phones'].apply(has_value)
        df_features['has_socials'] = df_features['socials'].apply(has_value)
        df_features['has_addresses'] = df_features['addresses'].apply(has_value)
        
        # Score each row
        scores = []
        for _, row in df_features.iterrows():
            score = score_row(row, self.config)
            scores.append(score)
        
        scores = np.array(scores)
        
        # Convert scores to probabilities
        # Higher score = more likely to be OPEN
        # We'll use sigmoid transformation
        p_open = 1 / (1 + np.exp(-scores))
        
        # Clip to avoid extreme probabilities
        p_open = np.clip(p_open, 0.01, 0.99)
        
        return p_open


class LabelCoverageSimulation:
    """Main simulation orchestrator for Rules-Only baseline."""
    
    def __init__(
        self,
        test_ids: list,
        test_df: pd.DataFrame,
        policy_cfg: dict = None,
    ):
        """
        Initialize simulation.
        
        Args:
            test_ids: List of test IDs (frozen, never touched)
            test_df: Test dataframe
            policy_cfg: Policy configuration dict
        """
        self.test_ids = set(test_ids)
        self.test_df = test_df.copy()
        
        self.policy_cfg = policy_cfg or {
            't_closed_high': 0.85,
            't_open_low': 0.15,
            'review_budget_pct': 0.1,  # 10% of batch
        }
        
        self.policy = TriagePolicy(
            t_closed_high=self.policy_cfg['t_closed_high'],
            t_open_low=self.policy_cfg['t_open_low'],
        )
        
        self.label_store = LabelStore()
        self.history = []  # List of metrics dicts per batch
    
    def run_batch(
        self,
        batch_id: int,
        batch_df: pd.DataFrame,
        all_data: pd.DataFrame,
        use_random_baseline: bool = False,
    ) -> dict:
        """
        Run one batch of the simulation.
        
        Args:
            batch_id: Batch number
            batch_df: Current batch dataframe
            all_data: All available data (for fitting encoders)
            use_random_baseline: If True, random review selection instead of uncertainty
        
        Returns:
            Metrics dict for this batch
        """
        print(f"\n{'='*70}")
        print(f"BATCH {batch_id}")
        print(f"{'='*70}")
        
        # Step 1: Get training data from label store
        training_data = self.label_store.get_training_data()
        labeled_ids = self.label_store.get_labeled_ids()
        
        print(f"\nCumulative labeled so far: {len(labeled_ids)}")
        if len(training_data) > 0:
            print(f"  Gold: {(training_data['source'].str.contains('manual|review')).sum()}")
            print(f"  Silver: {(training_data['source'].str.contains('auto')).sum()}")
        
        # Step 2: Train model (rules don't need training data but we'll initialize)
        print(f"\n{'='*60}")
        print(f"Training: Rules-Only Baseline (No training needed)")
        print(f"{'='*60}")
        
        model = RulesOnlyModel()
        model.fit(pd.DataFrame())  # Rules don't need training data
        
        # Step 3: Score current batch 
        batch_unlabeled = batch_df[~batch_df['id'].isin(labeled_ids)].copy()
        print(f"\nScoring {len(batch_unlabeled)} unlabeled records in batch...")
        
        p_open = model.predict_proba(batch_unlabeled)
        p_closed = 1 - p_open
        
        batch_unlabeled['p_closed'] = p_closed
        
        # Step 4: Apply triage policy
        print(f"\nApplying triage policy (thresholds: {self.policy_cfg['t_closed_high']:.2f}/{self.policy_cfg['t_open_low']:.2f})...")
        
        routed_batch = self.policy.route_batch(
            batch_df=batch_unlabeled,
            p_closed=p_closed,
        )
        
        # Print triage summary
        auto_accept_mask = routed_batch['route'] == 'auto_accept'
        review_mask = routed_batch['route'] == 'review'
        defer_mask = routed_batch['route'] == 'defer'
        
        auto_closed = (routed_batch.loc[auto_accept_mask, 'p_closed'] >= self.policy_cfg['t_closed_high']).sum()
        auto_open = (routed_batch.loc[auto_accept_mask, 'p_closed'] <= self.policy_cfg['t_open_low']).sum()
        
        print(f"Triage Results:")
        print(f"  Auto-accept: {auto_accept_mask.sum()} ({auto_accept_mask.mean():.1%})")
        print(f"  Review queue: {review_mask.sum()} ({review_mask.mean():.1%})")
        print(f"  Defer: {defer_mask.sum()} ({defer_mask.mean():.1%})")
        print(f"    Auto-closed: {auto_closed}, Auto-open: {auto_open}")
        
        # Step 5: Select review candidates
        review_batch = routed_batch[review_mask].copy()
        review_budget = max(int(len(batch_unlabeled) * self.policy_cfg['review_budget_pct']), 1)
        
        print(f"\nReview Budget: {review_budget}")
        
        if len(review_batch) == 0:
            review_ids = []
        else:
            if use_random_baseline:
                review_ids = review_batch.sample(min(review_budget, len(review_batch)), random_state=batch_id)['id'].tolist()
            else:
                # Select by uncertainty (already ranked by triage policy)
                review_ids = review_batch.head(review_budget)['id'].tolist()
        
        print(f"Sent for review: {len(review_ids)}")
        
        # Step 6: Add oracle labels for reviewed items
        if len(review_ids) > 0:
            review_records = batch_df[batch_df['id'].isin(review_ids)].copy()
            self.label_store.add_gold_labels(
                ids=review_records['id'].values,
                labels=review_records['open'].values,
                confidence=1.0,
                source=f'manual_review_batch_{batch_id}'
            )
            print(f"  Added {len(review_ids)} gold labels")
        
        # Step 7: Add auto-labels for high-confidence cases
        auto_accept_records = routed_batch[auto_accept_mask].copy()
        if len(auto_accept_records) > 0:
            # Predict labels based on thresholds
            predicted_labels = (auto_accept_records['p_closed'] < self.policy_cfg['t_closed_high']).astype(int)
            confidences = np.maximum(auto_accept_records['p_closed'], 1 - auto_accept_records['p_closed'])
            
            self.label_store.add_silver_labels(
                ids=auto_accept_records['id'].values,
                labels=predicted_labels.values,
                confidence=confidences.values,
                source=f'auto_batch_{batch_id}'
            )
            print(f"  Added {len(auto_accept_records)} silver labels")
        
        # Step 8: Evaluate test set performance
        test_eval = self._evaluate_on_test(model, batch_df)
        print(f"\nTest Set Evaluation:")
        print(f"  Closed PR-AUC: {test_eval['closed_pr_auc']:.4f}")
        print(f"  Closed F1: {test_eval['closed_f1']:.4f}")
        print(f"  Closed Precision: {test_eval['closed_precision']:.4f}, Recall: {test_eval['closed_recall']:.4f}")
        
        # Step 9: Evaluate auto-label quality
        if len(auto_accept_records) > 0:
            # Get true labels for auto-accepted records
            true_labels = []
            pred_labels = []
            confidences = []
            
            for _, row in auto_accept_records.iterrows():
                true_label = batch_df[batch_df['id'] == row['id']]['open'].iloc[0]
                pred_label = 0 if row['p_closed'] >= self.policy_cfg['t_closed_high'] else 1
                confidence = max(row['p_closed'], 1 - row['p_closed'])
                
                true_labels.append(true_label)
                pred_labels.append(pred_label)
                confidences.append(confidence)
            
            auto_label_precision = np.mean(np.array(true_labels) == np.array(pred_labels))
            conf_mean = np.mean(confidences)
            conf_median = np.median(confidences)
            conf_10 = np.percentile(confidences, 10)
            conf_90 = np.percentile(confidences, 90)
            
            print(f"  Auto-label precision: {auto_label_precision:.3f}")
            print(f"  Auto-confidence: mean={conf_mean:.3f}, median={conf_median:.3f}, 10-90%=[{conf_10:.3f}, {conf_90:.3f}]")
        else:
            auto_label_precision = None
            conf_mean = conf_median = conf_10 = conf_90 = None
        
        # Step 10: Store metrics
        batch_metrics = {
            'batch_id': batch_id,
            'labeled_cumulative': len(labeled_ids) + len(review_ids) + len(auto_accept_records),
            'labeled_this_round': len(review_ids) + len(auto_accept_records),
            'n_auto_accept': len(auto_accept_records),
            'n_review': len(review_ids),
            'n_defer': defer_mask.sum(),
            'auto_label_precision': auto_label_precision,
            'auto_conf_mean': conf_mean,
            'auto_conf_median': conf_median,
            'auto_conf_10pct': conf_10,
            'auto_conf_90pct': conf_90,
            'test_closed_pr_auc': test_eval.get('closed_pr_auc'),
            'test_closed_f1': test_eval.get('closed_f1'),
            'test_closed_precision': test_eval.get('closed_precision'),
            'test_closed_recall': test_eval.get('closed_recall'),
        }
        
        self.history.append(batch_metrics)
        self.label_store.print_stats()
        
        return batch_metrics
    
    def _evaluate_on_test(self, model, batch_df) -> dict:
        """Evaluate model performance on frozen test set."""
        try:
            p_open = model.predict_proba(self.test_df)
            
            y_true = self.test_df['open'].values  # Use original open labels
            y_pred = (p_open >= 0.5).astype(int)
            
            metrics = compute_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_score_open=p_open
            )
            
            return {
                'closed_pr_auc': metrics.get('pr_auc_closed', 0.0),
                'closed_f1': metrics.get('closed_f1', 0.0),
                'closed_precision': metrics.get('closed_precision', 0.0),
                'closed_recall': metrics.get('closed_recall', 0.0),
            }
        except Exception as e:
            print(f"Error evaluating on test set: {e}")
            return {}
    
    def get_history(self) -> pd.DataFrame:
        """Return simulation history as DataFrame."""
        return pd.DataFrame(self.history)


def main():
    """Run the Rules-Only baseline label coverage simulation."""
    print(f"\n{'='*70}")
    print(f"RULES-ONLY BASELINE LABEL-COVERAGE SIMULATION")
    print(f"{'='*70}")
    
    # Load data
    data_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/project_c_samples.parquet"
    train_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/train_split.parquet"
    test_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/test_split.parquet"
    
    df = pd.read_parquet(data_path)
    test_df = pd.read_parquet(test_path)
    train_df = pd.read_parquet(train_path)
    
    train_ids = train_df['id'].tolist()
    test_ids = test_df['id'].tolist()
    
    # Load val (if exists)
    val_ids_set = set()
    try:
        val_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/val_split.parquet"
        val_df = pd.read_parquet(val_path)
        val_ids_set = set(val_df['id'].tolist())
    except:
        pass
    
    sim_pool = df[df['id'].isin(train_ids + list(val_ids_set))].copy()
    print(f"Loaded simulation pool: {len(sim_pool)} records")
    print(f"Test set: {len(test_df)} records")
    
    # Build synthetic batches
    print(f"\n{'='*60}")
    print(f"Building 6 Synthetic Batches")
    print(f"{'='*60}")
    batches, metadata = build_sim_batches(sim_pool, n_batches=6, seed=42)
    
    # Display batch metadata
    print(f"Total records: {len(sim_pool)}")
    open_pct = (sim_pool['open'] == 1).mean() * 100
    print(f"Open: {(sim_pool['open'] == 1).sum()} ({open_pct:.1f}%)")
    print(f"Closed: {(sim_pool['open'] == 0).sum()} ({100-open_pct:.1f}%)")
    
    print(f"\nBatch Metadata:")
    print(metadata.to_string(index=False))
    
    # Initialize simulation
    sim = LabelCoverageSimulation(
        test_ids=test_ids,
        test_df=test_df,
    )
    
    # Run simulation
    for batch_id, batch_df in enumerate(batches):
        sim.run_batch(
            batch_id=batch_id,
            batch_df=batch_df,
            all_data=sim_pool,
        )
    
    print(f"\n{'='*70}")
    print(f"SIMULATION COMPLETE")
    print(f"{'='*70}")
    
    # Save results
    history = sim.get_history()
    print("\nSimulation History:")
    print(history.to_string(index=False))
    
    output_dir = Path("/Users/claricepark/Desktop/pro_rice_open_closed_prediction/artifacts/archive/label_coverage")
    output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_dir / "simulation_history_rules_only.csv", index=False)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
