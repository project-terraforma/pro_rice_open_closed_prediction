"""
Generate visualization charts for quick comparison results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_path = Path("/Users/claricepark/Desktop/pro_rice_open_closed_prediction/artifacts/quick_comparison")
results_df = pd.read_csv(results_path / "quick_results.csv")

# Filter out batch 0 (no training data yet) for cleaner visualization
results_df = results_df[results_df['batch_id'] > 0].copy()

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Label-Coverage Active-Labeling Loop: Strategy & Model Comparison', fontsize=16, fontweight='bold')

# Plot 1: PR-AUC by Batch (LR only)
ax = axes[0, 0]
lr_data = results_df[results_df['model_type'] == 'logistic_regression']
for strategy in ['uncertainty', 'random', 'static']:
    strategy_data = lr_data[lr_data['strategy'] == strategy]
    grouped = strategy_data.groupby('batch_id')['test_pr_auc'].mean()
    ax.plot(grouped.index, grouped.values, marker='o', label=strategy, linewidth=2, markersize=8)
ax.set_xlabel('Batch Number', fontsize=11)
ax.set_ylabel('Closed PR-AUC', fontsize=11)
ax.set_title('PR-AUC Trend: Logistic Regression', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Plot 2: PR-AUC Comparison (Batch 4)
ax = axes[0, 1]
batch4_data = results_df[results_df['batch_id'] == 3]
x_pos = np.arange(len(['uncertainty', 'random', 'static']))
width = 0.35

lr_values = [batch4_data[(batch4_data['model_type'] == 'logistic_regression') & (batch4_data['strategy'] == s)]['test_pr_auc'].mean() for s in ['uncertainty', 'random', 'static']]
xgb_values = [batch4_data[(batch4_data['model_type'] == 'xgboost') & (batch4_data['strategy'] == s)]['test_pr_auc'].mean() for s in ['uncertainty', 'random', 'static']]

ax.bar(x_pos - width/2, lr_values, width, label='LR', color='steelblue', alpha=0.8)
ax.bar(x_pos + width/2, xgb_values, width, label='XGBoost', color='coral', alpha=0.8)
ax.set_ylabel('Closed PR-AUC', fontsize=11)
ax.set_title('Final PR-AUC Comparison (Batch 4)', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['Uncertainty', 'Random', 'Static'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Auto-Label Precision
ax = axes[1, 0]
for strategy in ['uncertainty', 'random', 'static']:
    strategy_data = lr_data[lr_data['strategy'] == strategy]
    grouped = strategy_data.groupby('batch_id')['auto_precision'].mean()
    ax.plot(grouped.index, grouped.values * 100, marker='s', label=strategy, linewidth=2, markersize=8)
ax.set_xlabel('Batch Number', fontsize=11)
ax.set_ylabel('Auto-Label Precision (%)', fontsize=11)
ax.set_title('Auto-Label Quality: Logistic Regression', fontsize=12, fontweight='bold')
ax.axhline(y=85, color='red', linestyle='--', label='Target (85%)', linewidth=1.5)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 110])

# Plot 4: Labeled Data Accumulation
ax = axes[1, 1]
for strategy in ['uncertainty', 'random', 'static']:
    strategy_data = lr_data[lr_data['strategy'] == strategy].groupby('batch_id')['labeled_cumulative'].mean()
    ax.plot(strategy_data.index, strategy_data.values, marker='^', label=strategy, linewidth=2, markersize=8)
ax.set_xlabel('Batch Number', fontsize=11)
ax.set_ylabel('Cumulative Labeled Records', fontsize=11)
ax.set_title('Label Accumulation: Logistic Regression', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_path / "comparison_charts.png", dpi=300, bbox_inches='tight')
print(f"Chart saved to {results_path / 'comparison_charts.png'}")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\n1. Best Model per Strategy (by final PR-AUC):")
batch4_data = results_df[results_df['batch_id'] == 3]
for strategy in ['uncertainty', 'random', 'static']:
    strategy_data = batch4_data[batch4_data['strategy'] == strategy]
    best_model = strategy_data.loc[strategy_data['test_pr_auc'].idxmax()]
    print(f"   {strategy.upper():15} -> {best_model['model_type']:20} PR-AUC={best_model['test_pr_auc']:.4f}")

print("\n2. Average Performance (Batches 2-4):")
valid_data = results_df[results_df['batch_id'] >= 2]
for strategy in ['uncertainty', 'random', 'static']:
    for model in ['logistic_regression', 'xgboost']:
        subset = valid_data[(valid_data['strategy'] == strategy) & (valid_data['model_type'] == model)]
        if len(subset) > 0:
            avg_pr_auc = subset['test_pr_auc'].mean()
            avg_precision = subset['auto_precision'].mean() * 100
            print(f"   {strategy:15} + {model:20} -> PR-AUC={avg_pr_auc:.4f}, Precision={avg_precision:.1f}%")

print("\n3. Robustness (Std Dev across seeds, final batch):")
batch4_data = results_df[results_df['batch_id'] == 3]
for strategy in ['uncertainty', 'random', 'static']:
    for model in ['logistic_regression', 'xgboost']:
        subset = batch4_data[(batch4_data['strategy'] == strategy) & (batch4_data['model_type'] == model)]
        if len(subset) > 0:
            std_pr_auc = subset['test_pr_auc'].std()
            print(f"   {strategy:15} + {model:20} -> Std Dev={std_pr_auc:.4f}")

print("\n" + "="*80)
