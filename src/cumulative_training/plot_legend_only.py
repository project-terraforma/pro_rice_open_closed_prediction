#!/usr/bin/env python3
"""Generates a standalone legend PNG for the incremental benchmark plots."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

COLORS = {
    "LR":            "#4878CF",
    "XGBoost":       "#6ACC65",
    "LightGBM":      "#E53935",
    "Random Forest": "#FF8C00",
}

OUT = Path(__file__).resolve().parent / "incremental_benchmark_legend.png"

fig, ax = plt.subplots(figsize=(4.2, 1.0))
ax.set_axis_off()

handles = [
    mpatches.Patch(facecolor=color, edgecolor="white", linewidth=0.5, label=model)
    for model, color in COLORS.items()
]

legend = ax.legend(
    handles=handles,
    loc="center",
    ncol=4,
    fontsize=12,
    frameon=True,
    framealpha=1.0,
    edgecolor="#CCCCCC",
    handlelength=1.4,
    handleheight=1.0,
    columnspacing=1.2,
    handletextpad=0.6,
)
legend.get_frame().set_linewidth(0.8)

plt.tight_layout(pad=0.3)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
plt.close(fig)
