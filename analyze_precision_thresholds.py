#!/usr/bin/env python3
"""
Quick test to show the effect of different confidence thresholds
on auto-label precision.
"""

import sys
import os
import pandas as pd
sys.path.insert(0, '/Users/claricepark/Desktop/pro_rice_open_closed_prediction/src/models_v2')

from triage_policy import TriagePolicy
import numpy as np

def simulate_triage_precision(p_closed_values, true_labels, threshold_configs):
    """
    Simulate how different thresholds affect auto-label precision.
    
    Args:
        p_closed_values: Array of predicted probabilities
        true_labels: Array of true labels (0=open, 1=closed)
        threshold_configs: List of (t_closed_high, t_open_low) tuples
    """
    
    results = []
    
    for t_closed_high, t_open_low in threshold_configs:
        policy = TriagePolicy(t_closed_high=t_closed_high, t_open_low=t_open_low)
        
        # Apply triage policy
        auto_accept_closed = p_closed_values >= t_closed_high
        auto_accept_open = p_closed_values <= t_open_low
        auto_accept_mask = auto_accept_closed | auto_accept_open
        
        if auto_accept_mask.sum() == 0:
            precision = None
            n_auto = 0
        else:
            # Calculate auto-label precision
            auto_predictions = np.where(p_closed_values >= t_closed_high, 1, 0)
            auto_true = true_labels[auto_accept_mask]
            auto_pred = auto_predictions[auto_accept_mask]
            precision = (auto_true == auto_pred).mean()
            n_auto = auto_accept_mask.sum()
        
        results.append({
            't_closed_high': t_closed_high,
            't_open_low': t_open_low,
            'auto_precision': precision,
            'n_auto_accepted': n_auto,
            'pct_auto_accepted': n_auto / len(p_closed_values) * 100
        })
    
    return pd.DataFrame(results)

def main():
    print("=== Auto-Label Precision Analysis ===\n")
    
    # Simulate realistic model predictions
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic distribution (90% open, 10% closed)
    true_labels = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    # Create realistic model probabilities with some noise
    # Good model should predict high prob for closed businesses, low for open
    p_closed_base = true_labels * 0.7 + (1 - true_labels) * 0.1  
    p_closed_noise = np.random.normal(0, 0.15, n_samples)
    p_closed_values = np.clip(p_closed_base + p_closed_noise, 0.01, 0.99)
    
    print(f"Simulated dataset:")
    print(f"  Total samples: {n_samples}")
    print(f"  Closed businesses: {true_labels.sum()} ({true_labels.mean():.1%})")
    print(f"  Mean predicted prob(closed): {p_closed_values.mean():.3f}")
    print()
    
    # Test different threshold configurations
    threshold_configs = [
        (0.85, 0.15),  # Original (current simulation)
        (0.90, 0.10),  # Conservative  
        (0.95, 0.05),  # Very conservative
        (0.80, 0.20),  # Aggressive (more auto-labels, lower precision)
    ]
    
    results = simulate_triage_precision(p_closed_values, true_labels, threshold_configs)
    
    print("THRESHOLD ANALYSIS:")
    print("=" * 80)
    print(f"{'Thresholds':<15} {'Auto Precision':<15} {'# Auto-Accept':<15} {'% Auto-Accept':<15}")
    print("-" * 80)
    
    for _, row in results.iterrows():
        thresholds = f"{row['t_closed_high']:.2f}/{row['t_open_low']:.2f}"
        precision = f"{row['auto_precision']:.1%}" if row['auto_precision'] else "N/A"
        n_auto = f"{row['n_auto_accepted']:,}"
        pct_auto = f"{row['pct_auto_accepted']:.1f}%"
        
        print(f"{thresholds:<15} {precision:<15} {n_auto:<15} {pct_auto:<15}")
    
    print("\nKEY INSIGHTS:")
    print("- Higher thresholds (0.90/0.10, 0.95/0.05) = Higher precision, fewer auto-labels")
    print("- Lower thresholds (0.80/0.20) = Lower precision, more auto-labels") 
    print("- Current simulation uses 0.85/0.15")
    print("\nTO IMPROVE AUTO-LABEL PRECISION:")
    print("1. Increase t_closed_high from 0.85 to 0.90 or 0.95")
    print("2. Decrease t_open_low from 0.15 to 0.10 or 0.05") 
    print("3. Trade-off: Fewer auto-labels but higher accuracy")

if __name__ == "__main__":
    main()
