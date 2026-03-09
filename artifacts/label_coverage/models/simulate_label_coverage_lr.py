"""
Label-Coverage Simulation: Run iterative active-labeling loop on synthetic batches.

Protocol:
1. Freeze a final untouched test set.
2. Build simulation pool from train+val.
3. Create synthetic batches.
4. For each batch:
   - Train model on cumulative labeled data (gold + weighted silver).
   - Score batch with current model.
   - Triage batch into auto-accept, review, defer.
   - Reveal oracle labels for review queue.
   - Add new labels to store.
   - Evaluate on frozen test set.
5. Compare against baselines.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import ast

# Add src to path
sys.path.insert(0, '/Users/claricepark/Desktop/pro_rice_open_closed_prediction/src/models_v2')

from build_sim_batches import build_sim_batches, load_and_prepare_data, extract_primary_category, extract_primary_source
from triage_policy import TriagePolicy, print_triage_summary
from label_store import LabelStore
from logistic_regression_v2 import UnifiedLogisticRegression
from shared_metrics import compute_metrics


class LabelCoverageSimulation:
    """Main simulation orchestrator."""
    
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
            't_closed_high': 0.95,  # Very conservative for highest precision
            't_open_low': 0.05,     # Very conservative for highest precision  
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
        
        # Step 2: Build training set
        if len(training_data) > 0:
            # Filter all_data to only labeled records
            train_set = all_data[all_data['id'].isin(training_data['id'])].copy()
            train_set = train_set.merge(training_data[['id', 'label', 'weight']], on='id')
            
            print(f"\nTraining on {len(train_set)} records")
            print(f"  Open: {train_set['label'].sum()} ({train_set['label'].mean():.1%})")
            print(f"  Closed: {(~train_set['label'].astype(bool)).sum()} ({(~train_set['label'].astype(bool)).mean():.1%})")
        else:
            print("No labeled data yet; using seed rules only.")
            train_set = None
        
        # Step 3: Train model
        model = UnifiedLogisticRegression(mode="two-stage", feature_bundle="low_plus_medium")
        
        if train_set is not None and len(train_set) > 10:
            # Weighted training
            model.fit(train_set, val_df=None)
            predictions = model.predict_proba(batch_df)[:, 1]  # P(open)
            p_closed = 1.0 - predictions
        else:
            # No model yet; use uniform random
            print("Insufficient training data; using random predictions.")
            p_closed = np.random.uniform(0, 1, len(batch_df))
        
        # Step 4: Triage batch
        routed_batch = self.policy.route_batch(batch_df.copy(), p_closed)
        print_triage_summary(routed_batch)
        
        # Step 5: Select for review
        review_budget = max(1, int(len(batch_df) * self.policy_cfg['review_budget_pct']))
        
        if use_random_baseline:
            # Random selection from uncertain + review candidates
            candidates = routed_batch[routed_batch['route'].isin(['review', 'defer'])].copy()
            if len(candidates) > 0:
                review_ids = np.random.choice(candidates.index, size=min(review_budget, len(candidates)), replace=False)
                review_batch = routed_batch.loc[review_ids]
            else:
                review_batch = pd.DataFrame()
        else:
            # Uncertainty + impact selection
            candidates = routed_batch[routed_batch['route'] == 'review'].copy()
            if len(candidates) > 0:
                # Score by uncertainty * impact
                candidates['review_score'] = candidates['uncertainty'] * candidates['impact_score']
                candidates = candidates.sort_values('review_score', ascending=False)
                review_batch = candidates.head(review_budget)
            else:
                review_batch = pd.DataFrame()
        
        print(f"\nReview Budget: {review_budget}")
        print(f"Sent for review: {len(review_batch)}")
        
        # Step 6: Reveal oracle labels for reviewed records
        if len(review_batch) > 0:
            review_ids = review_batch['id'].values
            review_labels = review_batch['open'].values
            self.label_store.add_gold_labels(
                ids=review_ids,
                labels=review_labels,
                confidence=1.0,
                source='manual_review',
            )
            print(f"  Added {len(review_ids)} gold labels")
        
        # Step 7: Add auto-labels
        auto_accept_mask = routed_batch['route'] == 'auto_accept'
        if auto_accept_mask.sum() > 0:
            auto_ids = routed_batch[auto_accept_mask]['id'].values
            auto_labels = routed_batch[auto_accept_mask]['auto_label'].values
            auto_confidences = routed_batch[auto_accept_mask]['p_closed'].apply(
                lambda p: max(p, 1 - p)  # Confidence = max(p_closed, p_open)
            ).values
            
            self.label_store.add_silver_labels(
                ids=auto_ids,
                labels=auto_labels,
                confidence=auto_confidences,
                source='auto_high_confidence',
            )
            print(f"  Added {len(auto_ids)} silver labels")
        
        # Step 8: Evaluate on frozen test set
        test_eval = self._evaluate_on_test(model, routed_batch)
        
        # Step 9: Compute auto-label precision (audit sample)
        auto_labels_df = routed_batch[routed_batch['route'] == 'auto_accept'].copy()
        if len(auto_labels_df) > 0:
            auto_label_precision = (auto_labels_df['auto_label'] == auto_labels_df['open']).mean()
            
            # Compute confidence stats for auto-accepted items
            auto_confidences = auto_labels_df['p_closed'].apply(lambda p: max(p, 1 - p)).values
            conf_mean = np.mean(auto_confidences)
            conf_median = np.median(auto_confidences)
            conf_10 = np.percentile(auto_confidences, 10)
            conf_90 = np.percentile(auto_confidences, 90)
            
            print(f"  Auto-label precision: {auto_label_precision:.3f}")
            print(f"  Auto-confidence: mean={conf_mean:.3f}, median={conf_median:.3f}, 10-90%=[{conf_10:.3f}, {conf_90:.3f}]")
        else:
            auto_label_precision = None
            conf_mean = conf_median = conf_10 = conf_90 = None
        
        # Step 10: Store metrics
        batch_metrics = {
            'batch_id': batch_id,
            'labeled_cumulative': len(labeled_ids),
            'labeled_this_round': len(review_ids) + (auto_accept_mask.sum() if auto_accept_mask.sum() > 0 else 0) if 'review_ids' in locals() else 0,
            'n_auto_accept': auto_accept_mask.sum() if auto_accept_mask.sum() > 0 else 0,
            'n_review': len(review_batch),
            'n_defer': (routed_batch['route'] == 'defer').sum(),
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
        """
        Evaluate model on frozen test set.
        
        Args:
            model: Trained model
            batch_df: Current batch (for featurizer fit context)
        
        Returns:
            Metrics dict
        """
        try:
            # Predict on test
            test_predictions = model.predict_proba(self.test_df)[:, 1]  # P(open)
            p_closed = 1.0 - test_predictions  # P(closed)
            test_pred_closed = (p_closed >= 0.5).astype(int)  # Predict closed class
            
            # Compute metrics
            y_true = self.test_df['open'].values
            y_true_closed = (1 - y_true).astype(int)  # Convert to closed labels
            
            # Precision, recall, F1 for closed class at threshold 0.5
            tp = ((test_pred_closed == 1) & (y_true_closed == 1)).sum()  # closed predicted as closed
            fp = ((test_pred_closed == 1) & (y_true_closed == 0)).sum()  # open predicted as closed
            fn = ((test_pred_closed == 0) & (y_true_closed == 1)).sum()  # closed predicted as open
            
            closed_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            closed_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            closed_f1 = 2 * (closed_precision * closed_recall) / (closed_precision + closed_recall) if (closed_precision + closed_recall) > 0 else 0
            
            # PR-AUC for closed class
            from sklearn.metrics import auc, precision_recall_curve
            p_closed = 1.0 - test_predictions
            precision, recall, _ = precision_recall_curve(1 - y_true, p_closed)
            closed_pr_auc = auc(recall, precision)
            
            metrics = {
                'closed_precision': closed_precision,
                'closed_recall': closed_recall,
                'closed_f1': closed_f1,
                'closed_pr_auc': closed_pr_auc,
            }
            
            print(f"\nTest Set Evaluation:")
            print(f"  Closed PR-AUC: {closed_pr_auc:.4f}")
            print(f"  Closed F1: {closed_f1:.4f}")
            print(f"  Closed Precision: {closed_precision:.4f}, Recall: {closed_recall:.4f}")
            
            return metrics
        except Exception as e:
            print(f"  Error evaluating on test set: {e}")
            return {}
    
    def get_history(self) -> pd.DataFrame:
        """Get simulation history as dataframe."""
        return pd.DataFrame(self.history)


def main():
    """Run the full simulation."""
    
    print(f"\n{'='*70}")
    print(f"LABEL-COVERAGE SIMULATION")
    print(f"{'='*70}")
    
    # Load data (use parquet for proper types)
    data_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/project_c_samples.parquet"
    train_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/train_split.parquet"
    val_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/val_split.parquet"
    test_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/test_split.parquet"
    
    # Load full data from parquet
    df = pd.read_parquet(data_path)
    test_df = pd.read_parquet(test_path)
    test_ids = test_df['id'].tolist()
    
    # Load simulation pool (train + val) from parquet
    train_df = pd.read_parquet(train_path)
    train_ids = train_df['id'].tolist()
    
    # Try to load val from parquet, fallback to creating from remaining data
    val_ids_set = set()
    try:
        val_df = pd.read_parquet(val_path)
        val_ids = val_df['id'].tolist()
        val_ids_set = set(val_ids)
    except:
        pass
    
    sim_pool = df[df['id'].isin(train_ids + list(val_ids_set))].copy()
    print(f"Loaded simulation pool: {len(sim_pool)} records")
    
    # Build batches
    batches, batch_metadata = build_sim_batches(sim_pool, n_batches=6, seed=42)
    
    # Policy config
    policy_cfg = {
        't_closed_high': 0.85,
        't_open_low': 0.15,
        'review_budget_pct': 0.15,  # 15% of batch per round (focus on more closed examples)
    }
    
    # Run simulation
    sim = LabelCoverageSimulation(
        test_ids=test_ids,
        test_df=test_df,
        policy_cfg=policy_cfg,
    )
    
    for batch_id, batch_df in enumerate(batches):
        try:
            sim.run_batch(
                batch_id=batch_id,
                batch_df=batch_df,
                all_data=sim_pool,
                use_random_baseline=False,
            )
        except Exception as e:
            print(f"Error in batch {batch_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print final results
    print(f"\n{'='*70}")
    print(f"SIMULATION COMPLETE")
    print(f"{'='*70}")
    
    history = sim.get_history()
    print(f"\nSimulation History:")
    print(history.to_string(index=False))
    
    # Save results
    output_dir = Path("/Users/claricepark/Desktop/pro_rice_open_closed_prediction/artifacts/label_coverage")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history.to_csv(output_dir / "simulation_history.csv", index=False)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
