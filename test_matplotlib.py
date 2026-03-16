#!/usr/bin/env python3
import sys
import matplotlib
print(f"Python: {sys.version}", file=sys.stderr)
print(f"Matplotlib: {matplotlib.__version__}", file=sys.stderr)

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

print("Creating test plot...", file=sys.stderr)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot([1, 2, 3], [1, 2, 3])
out_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/test_plot.png"
fig.savefig(out_path, dpi=100)
print(f"Saved to {out_path}", file=sys.stderr)
plt.close(fig)
print("Done!", file=sys.stderr)
