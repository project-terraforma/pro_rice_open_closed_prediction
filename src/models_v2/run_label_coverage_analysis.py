"""
Comprehensive Label-Coverage Analysis:
- Run baseline comparisons (random-review, static-model)
- Test stronger model classes (XGBoost, LightGBM)
- Multi-seed robustness validation
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, '/Users/claricepark/Desktop/pro_rice_open_closed_prediction/src/models_v2')

from build_sim_batches import build_sim_batches, load_and_prepare_data
from triage_policy import TriagePolicy, print_triage_summary
from label_store import LabelStore
from logistic_regression_v2 import UnifiedLogisticRegression
from xgboost_model_v2 import XGBoostModel
from shared_metrics import compute_metrics

# Try to import LightGBM (may fail due to numpy compatibility)
try:
    from lightgbm_model_v2 import LightGBMModel
    LIGHTGBM_AVAILABLE = True
except (ImportError, AttributeError):
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available due to dependency issues, will skip LightGBM tests")


class ModelFactory:
    """Factory for creating models."""
    
    @staticmethod
    def create_model(model_type: str, mode: str = "two-stage"):
        """Create a model instance."""
        if model_type == "logistic_regression":
            return UnifiedLogisticRegression(mode=mode, feature_bundle="low_plus_medium")
        elif model_type == "xgboost":
            return XGBoostModel(mode=mode, feature_bundle="low_plus_medium")
        elif model_type == "lightgbm":
            if not LIGHTGBM_AVAILABLE:
                raise ValueError("LightGBM not available")
            return LightGBMModel(mode=mode, feature_bundle="low_plus_medium")
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class LabelCoverageSimulation:
    """Main simulation orchestrator with multiple strategies."""
    
    def __init__(
        self,
        test_ids: list,
        test_df: pd.DataFrame,
        model_type: str = "logistic_regression",
        policy_cfg: dict = None,
    ):
        """Initialize simulation."""
        self.test_ids = set(test_ids)
        self.test_df = test_df.copy()
        self.model_type = model_type
        
        self.policy_cfg = policy_cfg or {
            't_closed_high': 0.85,
            't_open_low': 0.15,
            'review_budget_pct': 0.05,
        }
        
        self.policy = TriagePolicy(
            t_closed_high=self.policy_cfg['t_closed_high'],
            t_open_low=self.policy_cfg['t_open_low'],
        )
        
        self.label_store = LabelStore()
        self.history = []
    
    def run_batch(
        self,
        batch_id: int,
        batch_df: pd.DataFrame,
        all_data: pd.DataFrame,
        strategy: str = "uncertainty",  # uncertainty, random, static
    ) -> dict:
        """
        Run one batch with specified strategy.
        
        Args:
            batch_id: Batch number
            batch_df: Current batch dataframe
            all_data: All available data (for fitting encoders)
            strategy: Review strategy ('uncertainty', 'random', 'static')
        
        Returns:
            Metrics dict for this batch
        """
        print(f"\n{'='*70}")
        print(f"BATCH {batch_id} | Strategy: {strategy.upper()}")
        print(f"{'='*70}")
        
        # Step 1: Get training data from label store
        training_data = self.label_store.get_training_data()
        labeled_ids = self.label_store.get_labeled_ids()
        
        print(f"Cumulative labeled so far: {len(labeled_ids)}")
        
        # Step 2: Build training set
        if len(training_data) > 0:
            train_set = all_data[all_data['id'].isin(training_data['id'])].copy()
            train_set = train_set.merge(training_data[['id', 'label', 'weight']], on='id')
            print(f"Training on {len(train_set)} records")
        else:
            train_set = None
        
        # Step 3: Train model
        model = ModelFactory.create_model(self.model_type, mode="two-stage")
        
        if train_set is not None and len(train_set) > 10:
            model.fit(train_set, val_df=None)
            predictions = model.predict_proba(batch_df)[:, 1]  # P(open)
            p_closed = 1.0 - predictions
        else:
            # Use random predictions
            p_closed = np.random.uniform(0, 1, len(batch_df))
        
        # Step 4: Triage batch
        routed_batch = self.policy.route_batch(batch_df.copy(), p_closed)
        print_triage_summary(routed_batch)
        
        # Step 5: Select for review based on strategy
        review_budget = max(1, int(len(batch_df) * self.policy_cfg['review_budget_pct']))
        
        if strategy == "random":
            # Random selection from uncertain + review candidates
            candidates = routed_batch[routed_batch['route'].isin(['review', 'defer'])].copy()
            if len(candidates) > 0:
                review_ids = np.random.choice(candidates.index, size=min(review_budget, len(candidates)), replace=False)
                review_batch = routed_batch.loc[review_ids]
            else:
                review_batch = pd.DataFrame()
        
        elif strategy == "static":
            # Static model: use seed rules only, never retrain (just add auto-labels)
            review_batch = pd.DataFrame()  # No manual review
        
        else:  # uncertainty
            # Uncertainty + impact selection
            candidates = routed_batch[routed_batch['route'] == 'review'].copy()
            if len(candidates) > 0:
                candidates['review_score'] = candidates['uncertainty'] * candidates['impact_score']
                candidates = candidates.sort_values('review_score', ascending=False)
                review_batch = candidates.head(review_budget)
            else:
                review_batch = pd.DataFrame()
        
        print(f"Review Budget: {review_budget}")
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
            print(f"Added {len(review_ids)} gold labels")
        
        # Step 7: Add auto-labels
        auto_accept_mask = routed_batch['route'] == 'auto_accept'
        if auto_accept_mask.sum() > 0:
            auto_ids = routed_batch[auto_accept_mask]['id'].values
            auto_labels = routed_batch[auto_accept_mask]['auto_label'].values
            auto_confidences = routed_batch[auto_accept_mask]['p_closed'].apply(
                lambda p: max(p, 1 - p)
            ).values
            
            self.label_store.add_silver_labels(
                ids=auto_ids,
                labels=auto_labels,
                confidence=auto_confidences,
                source='auto_high_confidence',
            )
            print(f"Added {len(auto_ids)} silver labels")
        
        # Step 8: Evaluate on frozen test set
        test_eval = self._evaluate_on_test(model)
        
        # Step 9: Compute auto-label precision
        auto_labels_df = routed_batch[routed_batch['route'] == 'auto_accept'].copy()
        auto_label_precision = (auto_labels_df['auto_label'] == auto_labels_df['open']).mean() if len(auto_labels_df) > 0 else None
        
        # Step 10: Store metrics
        batch_metrics = {
            'batch_id': batch_id,
            'strategy': strategy,
            'model_type': self.model_type,
            'labeled_cumulative': len(labeled_ids),
            'n_gold_this_round': len(review_batch),
            'n_silver_this_round': auto_accept_mask.sum() if auto_accept_mask.sum() > 0 else 0,
            'n_auto_accept': auto_accept_mask.sum() if auto_accept_mask.sum() > 0 else 0,
            'n_review': len(review_batch),
            'n_defer': (routed_batch['route'] == 'defer').sum(),
            'auto_label_precision': auto_label_precision,
            'test_closed_pr_auc': test_eval.get('closed_pr_auc'),
            'test_closed_f1': test_eval.get('closed_f1'),
            'test_closed_precision': test_eval.get('closed_precision'),
            'test_closed_recall': test_eval.get('closed_recall'),
        }
        
        self.history.append(batch_metrics)
        self.label_store.print_stats()
        
        return batch_metrics
    
    def _evaluate_on_test(self, model) -> dict:
        """Evaluate model on frozen test set."""
        try:
            test_predictions = model.predict_proba(self.test_df)[:, 1]  # P(open)
            p_closed = 1.0 - test_predictions  # P(closed)
            test_pred_closed = (p_closed >= 0.5).astype(int)  # Predict closed class
            
            y_true = self.test_df['open'].values
            y_true_closed = (1 - y_true).astype(int)  # Convert to closed labels
            
            tp = ((test_pred_closed == 1) & (y_true_closed == 1)).sum()  # closed predicted as closed
            fp = ((test_pred_closed == 1) & (y_true_closed == 0)).sum()  # open predicted as closed
            fn = ((test_pred_closed == 0) & (y_true_closed == 1)).sum()  # closed predicted as open
            
            closed_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            closed_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            closed_f1 = 2 * (closed_precision * closed_recall) / (closed_precision + closed_recall) if (closed_precision + closed_recall) > 0 else 0
            
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
            
            print(f"Test Closed PR-AUC: {closed_pr_auc:.4f}")
            
            return metrics
        except Exception as e:
            print(f"Error evaluating on test set: {e}")
            return {}
    
    def get_history(self) -> pd.DataFrame:
        """Get simulation history as dataframe."""
        return pd.DataFrame(self.history)


def run_strategy_comparison(
    sim_pool: pd.DataFrame,
    test_df: pd.DataFrame,
    batches: List[pd.DataFrame],
    n_seeds: int = 3,
) -> Dict[str, pd.DataFrame]:
    """
    Run comparison of strategies and model types across multiple seeds.
    
    Returns:
        Dict mapping (strategy, model_type) -> history DataFrame
    """
    
    strategies = ["uncertainty", "random", "static"]
    model_types = ["logistic_regression", "xgboost"]
    if LIGHTGBM_AVAILABLE:
        model_types.append("lightgbm")
    
    results = {}
    
    test_ids = test_df['id'].tolist()
    
    for seed in range(n_seeds):
        print(f"\n{'='*80}")
        print(f"SEED {seed}")
        print(f"{'='*80}")
        
        # Re-create batches with different seed for ordering variability
        seed_batches, _ = build_sim_batches(sim_pool, n_batches=6, seed=42 + seed)
        
        for model_type in model_types:
            print(f"\n{'='*80}")
            print(f"MODEL: {model_type.upper()}")
            print(f"{'='*80}")
            
            for strategy in strategies:
                print(f"\n{'='*80}")
                print(f"STRATEGY: {strategy.upper()}")
                print(f"{'='*80}")
                
                try:
                    sim = LabelCoverageSimulation(
                        test_ids=test_ids,
                        test_df=test_df,
                        model_type=model_type,
                    )
                    
                    for batch_id, batch_df in enumerate(seed_batches):
                        sim.run_batch(
                            batch_id=batch_id,
                            batch_df=batch_df,
                            all_data=sim_pool,
                            strategy=strategy,
                        )
                    
                    history = sim.get_history()
                    key = (seed, strategy, model_type)
                    results[key] = history
                    
                    # Print summary
                    print(f"\n{'='*70}")
                    print(f"SUMMARY (Seed {seed}, {strategy}, {model_type})")
                    print(f"{'='*70}")
                    print(f"Initial Closed PR-AUC: {history.iloc[0]['test_closed_pr_auc']:.4f}")
                    print(f"Final Closed PR-AUC: {history.iloc[-1]['test_closed_pr_auc']:.4f}")
                    print(f"Improvement: {(history.iloc[-1]['test_closed_pr_auc'] / history.iloc[0]['test_closed_pr_auc'] - 1) * 100:.1f}%")
                    print(f"Auto-label Precision: {history['auto_label_precision'].mean():.3f}")
                    
                except Exception as e:
                    print(f"ERROR in seed {seed}, {strategy}, {model_type}: {e}")
                    import traceback
                    traceback.print_exc()
    
    return results


def aggregate_results(results: Dict, output_dir: Path) -> None:
    """Aggregate and save results across all runs."""
    
    print(f"\n{'='*80}")
    print(f"AGGREGATING RESULTS")
    print(f"{'='*80}")
    
    # Combine all results
    all_history = pd.concat(results.values(), ignore_index=True)
    all_history.to_csv(output_dir / "all_runs_history.csv", index=False)
    
    # Aggregate by strategy and model type
    summary_stats = []
    
    for (seed, strategy, model_type), history in results.items():
        if len(history) > 0:
            summary_stats.append({
                'seed': seed,
                'strategy': strategy,
                'model_type': model_type,
                'initial_pr_auc': history.iloc[0]['test_closed_pr_auc'],
                'final_pr_auc': history.iloc[-1]['test_closed_pr_auc'],
                'pr_auc_improvement_pct': (history.iloc[-1]['test_closed_pr_auc'] / history.iloc[0]['test_closed_pr_auc'] - 1) * 100,
                'avg_auto_label_precision': history['auto_label_precision'].mean(),
                'total_gold_labels': history['n_gold_this_round'].sum(),
                'total_silver_labels': history['n_silver_this_round'].sum(),
                'total_labels': history['n_gold_this_round'].sum() + history['n_silver_this_round'].sum(),
                'leverage_ratio': (history['n_silver_this_round'].sum() / history['n_gold_this_round'].sum()) if history['n_gold_this_round'].sum() > 0 else 0,
            })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / "summary_stats.csv", index=False)
    
    # Aggregate across seeds by strategy and model
    agg_by_strategy_model = summary_df.groupby(['strategy', 'model_type']).agg({
        'final_pr_auc': ['mean', 'std'],
        'pr_auc_improvement_pct': ['mean', 'std'],
        'avg_auto_label_precision': ['mean', 'std'],
        'total_labels': ['mean', 'std'],
        'leverage_ratio': ['mean', 'std'],
    }).round(4)
    
    agg_by_strategy_model.to_csv(output_dir / "aggregated_by_strategy_model.csv")
    print("\nAggregated Results by Strategy & Model:")
    print(agg_by_strategy_model)
    
    # Aggregate across seeds by strategy (pooling all models)
    agg_by_strategy = summary_df.groupby('strategy').agg({
        'final_pr_auc': ['mean', 'std'],
        'pr_auc_improvement_pct': ['mean', 'std'],
        'avg_auto_label_precision': ['mean', 'std'],
    }).round(4)
    
    agg_by_strategy.to_csv(output_dir / "aggregated_by_strategy.csv")
    print("\nAggregated Results by Strategy:")
    print(agg_by_strategy)
    
    # Best model per strategy
    best_by_strategy = summary_df.loc[summary_df.groupby('strategy')['final_pr_auc'].idxmax()]
    best_by_strategy.to_csv(output_dir / "best_by_strategy.csv", index=False)
    print("\nBest Model per Strategy:")
    print(best_by_strategy)


def main():
    """Run the full analysis."""
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE LABEL-COVERAGE ANALYSIS")
    print(f"{'='*80}")
    
    # Load data
    data_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/project_c_samples.parquet"
    train_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/train_split.parquet"
    test_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/test_split.parquet"
    
    df = pd.read_parquet(data_path)
    test_df = pd.read_parquet(test_path)
    train_df = pd.read_parquet(train_path)
    
    train_ids = train_df['id'].tolist()
    
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
    
    # Build batches (initial seed)
    batches, _ = build_sim_batches(sim_pool, n_batches=6, seed=42)
    
    # Run comparisons
    results = run_strategy_comparison(
        sim_pool=sim_pool,
        test_df=test_df,
        batches=batches,
        n_seeds=3,
    )
    
    # Aggregate and save results
    output_dir = Path("/Users/claricepark/Desktop/pro_rice_open_closed_prediction/artifacts/label_coverage_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    aggregate_results(results, output_dir)
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
