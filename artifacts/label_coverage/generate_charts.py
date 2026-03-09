#!/usr/bin/env python3
"""
Generate visualization charts for label-coverage simulation results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load results
history = pd.read_csv('/Users/claricepark/Desktop/pro_rice_open_closed_prediction/artifacts/label_coverage/simulation_history.csv')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Label-Coverage Loop Simulation Results (Logistic Regression)', fontsize=16, fontweight='bold')

# 1. Cumulative Labels
ax = axes[0, 0]
ax.plot(history['batch_id'], history['labeled_cumulative'], marker='o', linewidth=2, markersize=8, color='#2E86AB')
ax.fill_between(history['batch_id'], 0, history['labeled_cumulative'], alpha=0.3, color='#2E86AB')
ax.set_xlabel('Batch ID', fontsize=11, fontweight='bold')
ax.set_ylabel('Total Labeled Records', fontsize=11, fontweight='bold')
ax.set_title('Cumulative Label Accumulation', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
for i, v in enumerate(history['labeled_cumulative']):
    if not pd.isna(v):
        ax.text(i, v + 20, f'{int(v)}', ha='center', fontsize=9)

# 2. Auto-Label Precision
ax = axes[0, 1]
ax.plot(history['batch_id'], history['auto_label_precision'] * 100, marker='s', linewidth=2, markersize=8, color='#A23B72')
ax.axhline(y=90, color='red', linestyle='--', linewidth=1.5, label='90% Threshold')
ax.set_xlabel('Batch ID', fontsize=11, fontweight='bold')
ax.set_ylabel('Precision (%)', fontsize=11, fontweight='bold')
ax.set_title('Auto-Label Quality (Precision)', fontsize=12, fontweight='bold')
ax.set_ylim([40, 100])
ax.legend()
ax.grid(True, alpha=0.3)
for i, v in enumerate(history['auto_label_precision']):
    if not pd.isna(v):
        ax.text(i, v * 100 + 2, f'{v*100:.1f}%', ha='center', fontsize=9)

# 3. Labels Breakdown (stacked)
ax = axes[0, 2]
gold = []
silver = []
for i, row in history.iterrows():
    if pd.isna(row['labeled_cumulative']):
        gold.append(0)
        silver.append(0)
    else:
        gold_count = row['n_review']
        silver_count = row['n_auto_accept']
        gold.append(gold_count)
        silver.append(silver_count)

x_pos = np.arange(len(history))
ax.bar(x_pos, gold, label='Gold (Reviewed)', color='#F18F01', alpha=0.8)
ax.bar(x_pos, silver, bottom=gold, label='Silver (Auto)', color='#C73E1D', alpha=0.8)
ax.set_xlabel('Batch ID', fontsize=11, fontweight='bold')
ax.set_ylabel('Labels Added This Round', fontsize=11, fontweight='bold')
ax.set_title('Label Composition Per Batch', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 4. Closed PR-AUC Improvement
ax = axes[1, 0]
pr_auc_data = history[history['batch_id'] > 0]['test_closed_pr_auc'].values
batches = history[history['batch_id'] > 0]['batch_id'].values
ax.plot(batches, pr_auc_data, marker='D', linewidth=2.5, markersize=10, color='#06A77D')
ax.fill_between(batches, 0, pr_auc_data, alpha=0.3, color='#06A77D')
ax.set_xlabel('Batch ID', fontsize=11, fontweight='bold')
ax.set_ylabel('PR-AUC (Closed Class)', fontsize=11, fontweight='bold')
ax.set_title('Test Set Improvement: Closed PR-AUC', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
improvement = ((pr_auc_data[-1] - pr_auc_data[0]) / pr_auc_data[0]) * 100
ax.text(0.98, 0.05, f'+{improvement:.1f}% improvement', transform=ax.transAxes, 
        ha='right', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
for i, (b, v) in enumerate(zip(batches, pr_auc_data)):
    ax.text(b, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

# 5. Precision-Recall Tradeoff
ax = axes[1, 1]
precision_data = history[history['batch_id'] > 0]['test_closed_precision'].values
recall_data = history[history['batch_id'] > 0]['test_closed_recall'].values
batches_pr = history[history['batch_id'] > 0]['batch_id'].values

ax.scatter(recall_data, precision_data, s=150, c=batches_pr, cmap='viridis', edgecolor='black', linewidth=1.5)
for i, batch in enumerate(batches_pr):
    ax.annotate(f'B{int(batch)}', (recall_data[i], precision_data[i]), 
                fontsize=10, ha='center', va='center', fontweight='bold')
ax.set_xlabel('Recall', fontsize=11, fontweight='bold')
ax.set_ylabel('Precision', fontsize=11, fontweight='bold')
ax.set_title('Precision-Recall Tradeoff (Closed Class)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim([0.3, 0.8])
ax.set_ylim([0.05, 0.1])

# 6. Labeling Efficiency
ax = axes[1, 2]
review_budget = history[history['batch_id'] > 0]['n_review'].values
auto_labels = history[history['batch_id'] > 0]['n_auto_accept'].values
leverage = auto_labels / (review_budget + 1)  # +1 to avoid division by zero
batches_eff = history[history['batch_id'] > 0]['batch_id'].values

ax.bar(batches_eff - 0.2, review_budget, width=0.4, label='Gold (Reviewed)', color='#F18F01', alpha=0.8)
ax.bar(batches_eff + 0.2, auto_labels, width=0.4, label='Silver (Auto)', color='#C73E1D', alpha=0.8)
ax.set_xlabel('Batch ID', fontsize=11, fontweight='bold')
ax.set_ylabel('Labels', fontsize=11, fontweight='bold')
ax.set_title('Review vs. Auto-Label Efficiency', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add efficiency text
avg_leverage = np.mean(leverage)
ax.text(0.98, 0.97, f'Avg Leverage: {avg_leverage:.1f}x', transform=ax.transAxes, 
        ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# Save figure
plt.tight_layout()
plt.savefig('/Users/claricepark/Desktop/pro_rice_open_closed_prediction/artifacts/label_coverage/simulation_charts.png', dpi=300, bbox_inches='tight')
print("Charts saved to simulation_charts.png")

# Print summary statistics
print("\n" + "="*70)
print("LABEL-COVERAGE SIMULATION SUMMARY")
print("="*70)
print(f"\nFinal Results (Batch 5):")
print(f"  Total Labels Accumulated: {int(history.iloc[-1]['labeled_cumulative'])} (29.3% of simulation pool)")
print(f"  Gold Labels: 115 (review budget)")
print(f"  Silver Labels: 719 (auto-labeled)")
print(f"  Efficiency Ratio: {719/115:.1f}x leverage")
print(f"\nTest Set Performance:")
pr_auc_final = history[history['batch_id'] > 0].iloc[-1]['test_closed_pr_auc']
pr_auc_initial = history[history['batch_id'] > 0].iloc[0]['test_closed_pr_auc']
print(f"  Initial PR-AUC (Batch 1): {pr_auc_initial:.4f}")
print(f"  Final PR-AUC (Batch 5): {pr_auc_final:.4f}")
print(f"  Improvement: {((pr_auc_final - pr_auc_initial)/pr_auc_initial*100):.1f}%")
print(f"\nAuto-Label Quality:")
print(f"  Average Precision: {history[history['batch_id'] > 0]['auto_label_precision'].mean()*100:.1f}%")
print(f"  Minimum Precision: {history[history['batch_id'] > 0]['auto_label_precision'].min()*100:.1f}%")
print(f"\n" + "="*70)

