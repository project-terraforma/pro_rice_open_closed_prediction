#!/usr/bin/env python3
"""
Run label coverage simulation with higher confidence thresholds
to improve auto-label precision.
"""

import sys
import os
sys.path.insert(0, '/Users/claricepark/Desktop/pro_rice_open_closed_prediction/src/models_v2')

from simulate_label_coverage import LabelCoverageSimulation
from build_sim_batches import load_and_prepare_data, build_sim_batches
import pandas as pd
from pathlib import Path

def main():
    print("=== Running Improved Auto-Label Precision Simulation ===")
    
    # Load data
    print("Loading data...")
    data_path = Path('/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/project_c_samples.parquet')
    train_df, val_df, test_df = load_and_prepare_data(data_path)
    
    # More conservative thresholds for higher precision
    policy_configs = [
        {
            'name': 'Conservative',
            'config': {
                't_closed_high': 0.90,  # Increased from 0.85
                't_open_low': 0.10,     # Decreased from 0.15
                'review_budget_pct': 0.1,
            }
        },
        {
            'name': 'Very Conservative', 
            'config': {
                't_closed_high': 0.95,  # Very high confidence required
                't_open_low': 0.05,     # Very low for open
                'review_budget_pct': 0.1,
            }
        }
    ]
    
    results = []
    
    for policy_setup in policy_configs:
        print(f"\n--- Testing {policy_setup['name']} Policy ---")
        print(f"Thresholds: closed >= {policy_setup['config']['t_closed_high']}, open <= {policy_setup['config']['t_open_low']}")
        
        # Create simulator
        simulator = LabelCoverageSimulation(
            test_ids=test_df.index.tolist(),
            test_df=test_df,
            policy_cfg=policy_setup['config']
        )
        
        # Run 3 batches for quick comparison
        for batch_id in range(3):
            print(f"\n  Batch {batch_id}:")
            metrics = simulator.run_batch(batch_id)
            
            if metrics['auto_label_precision'] is not None:
                print(f"    Auto-label precision: {metrics['auto_label_precision']:.1%}")
                print(f"    Auto-accepted: {metrics['n_auto_accept']}")
                print(f"    For review: {metrics['n_review']}")
        
        # Save results for this policy
        history_df = pd.DataFrame(simulator.history)
        avg_precision = history_df[history_df['auto_label_precision'].notna()]['auto_label_precision'].mean()
        
        results.append({
            'policy': policy_setup['name'],
            't_closed_high': policy_setup['config']['t_closed_high'],
            't_open_low': policy_setup['config']['t_open_low'],
            'avg_auto_precision': avg_precision,
            'final_pr_auc': history_df['test_closed_pr_auc'].iloc[-1] if len(history_df) > 0 else None
        })
        
        print(f"\n  Average auto-label precision: {avg_precision:.1%}")
    
    # Compare results
    print("\n=== COMPARISON RESULTS ===")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    print(f"\nOriginal simulation (0.85/0.15 thresholds) had precision dropping to ~67-76%")
    print(f"Conservative approach should maintain higher precision!")

if __name__ == "__main__":
    main()
