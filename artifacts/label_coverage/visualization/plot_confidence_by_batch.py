import matplotlib.pyplot as plt
import numpy as np
import os

# Data (from simulation results)
batches = np.array([0, 1, 2, 3, 4, 5])
lr = np.array([0.926, 0.960, 0.963, 0.963, 0.978, 0.975])
xgb = np.array([np.nan, 0.990, 0.990, 0.990, 0.897, 0.950])
rf = np.array([np.nan, 0.957, 0.986, 0.993, 0.990, 0.990])
lgb = np.array([np.nan, 0.990, 0.949, 0.998, 0.997, 0.998])
rules = np.array([0.904, 0.903, 0.901, 0.900, 0.899, 0.898])

plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(10, 6))

plt.plot(batches, lr, marker='o', label='Logistic Regression', linewidth=2)
plt.plot(batches, xgb, marker='s', label='XGBoost', linewidth=2)
plt.plot(batches, rf, marker='^', label='Random Forest', linewidth=2)
plt.plot(batches, lgb, marker='d', label='LightGBM', linewidth=2)
plt.plot(batches, rules, marker='x', label='Rules-Only', linewidth=2)

# X axis: integer batches (no decimals)
plt.xticks(batches)
plt.xlabel('Batch', fontsize=12)
plt.ylabel('Auto-labeling Confidence (p_closed)', fontsize=12)
plt.ylim(0.88, 1.00)
plt.yticks(np.linspace(0.88, 1.00, 7))
plt.title('Model Auto-Labeling Confidence by Batch', fontsize=14)
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.6)

outdir = os.path.dirname(__file__)
if not outdir:
    outdir = '.'
outpath = os.path.join(outdir, 'confidence_by_batch.png')
plt.tight_layout()
plt.savefig(outpath, dpi=200)
print(f"Saved plot to {outpath}")
