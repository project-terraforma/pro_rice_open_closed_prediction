"""
Plot incremental benchmark accuracy across batches + single run.

Reads accuracy values directly from the benchmark results (hardcoded from
incremental_results.md).  The single-run point is placed to the right of
batch 5 and labelled "Single".

Output: src/incremental_benchmarking/incremental_benchmark_accuracy.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

# ── Data from incremental_results.md ─────────────────────────────────────────
# Accuracy per batch (incremental / cumulative retrain) + single run
DATA = {
    "LR": {
        "batches": [0.8613, 0.8628, 0.8803, 0.8584, 0.8555],
        "single":   0.8555,
    },
    "XGBoost": {
        "batches": [0.9095, 0.9124, 0.9095, 0.9109, 0.9095],
        "single":   0.9095,
    },
    "LightGBM": {
        "batches": [0.8117, 0.7650, 0.7752, 0.7723, 0.7664],
        "single":   0.7664,
    },
    "Random Forest": {
        "batches": [0.8058, 0.7693, 0.7737, 0.7839, 0.8175],
        "single":   0.8175,
    },
}

COLORS = {
    "LR":            "#4878CF",   # blue
    "XGBoost":       "#6ACC65",   # green
    "LightGBM":      "#E53935",   # brighter red
    "Random Forest": "#FF8C00",   # orange
}

# Darker versions for text boxes
DARK_COLORS = {
    "LR":            "#2E5AA0",
    "XGBoost":       "#4A9D3F",
    "LightGBM":      "#A83838",
    "Random Forest": "#8B5BA3",
}

N_BATCHES   = 5
BATCH_XS    = list(range(1, N_BATCHES + 1))   # 1 … 5
SINGLE_X    = N_BATCHES + 1.5                 # gap between batch 5 and single

# ── Figure with improved styling ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7), facecolor="white")
ax.set_facecolor("#FAFBFC")

for model, vals in DATA.items():
    color = COLORS[model]
    acc   = vals["batches"]
    sing  = vals["single"]

    # Incremental line (batches 1–5) — thicker, smoother
    ax.plot(BATCH_XS, acc, marker="o", color=color, linewidth=2.5,
            markersize=7, label=model, zorder=3, alpha=0.85)

    # Dashed connector from batch 5 to single run
    ax.plot([N_BATCHES, SINGLE_X], [acc[-1], sing],
            linestyle="--", color=color, linewidth=1.5,
            alpha=0.5, zorder=2)

    # Single-run marker: simple filled dot
    ax.scatter([SINGLE_X], [sing], marker="o", s=80,
               color=color, zorder=4, edgecolors="white", linewidths=0.8)

# ── Add accuracy value boxes at each point ────────────────────────────────────
# Collect all points for collision detection/spacing
all_points = {}
for model in DATA:
    all_points[model] = {}
    for i, x in enumerate(BATCH_XS):
        all_points[model][x] = DATA[model]["batches"][i]
    all_points[model][SINGLE_X] = DATA[model]["single"]

# Helper function to check overlap and adjust positions
def add_accuracy_labels(ax, all_points, batch_xs, single_x):
    box_height = 0.012
    box_width = 0.16
    
    for x_pos in batch_xs + [single_x]:
        # Get all values at this x position
        values_at_x = [(model, all_points[model][x_pos]) for model in DATA]
        values_at_x.sort(key=lambda v: v[1])  # Sort by y value
        
        n_values = len(values_at_x)
        
        if n_values == 1:
            # Single value: place directly above
            model, y_val = values_at_x[0]
            add_box(ax, x_pos, y_val, f"{y_val:.4f}", DARK_COLORS[model], box_width, box_height)
        else:
            # Multiple values: spread horizontally
            total_width = box_width * n_values + 0.02 * (n_values - 1)
            start_x = x_pos - total_width / 2
            
            for i, (model, y_val) in enumerate(values_at_x):
                box_x = start_x + i * (box_width + 0.02)
                add_box(ax, box_x, y_val, f"{y_val:.4f}", DARK_COLORS[model], box_width, box_height)

def add_box(ax, x, y, text, color, width, height):
    """Add a fancy box with accuracy text."""
    box = FancyBboxPatch(
        (x - width/2, y + height/2), width, height,
        boxstyle="round,pad=0.003",
        facecolor=color,
        edgecolor="white",
        linewidth=1,
        alpha=0.85,
        zorder=5,
        transform=ax.transData,
    )
    ax.add_patch(box)
    
    ax.text(x, y + height/2, text,
            ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="white",
            zorder=6)

add_accuracy_labels(ax, all_points, BATCH_XS, SINGLE_X)

# ── Vertical divider between batch 5 and single run ──────────────────────────
divider_x = (N_BATCHES + SINGLE_X) / 2   # midpoint ≈ 5.75
ax.axvline(divider_x, color="#D0D5DC", linewidth=1.2,
           linestyle=":", alpha=0.8, zorder=1)

# ── Axes formatting ───────────────────────────────────────────────────────────
# Ticks: 1–5 (integers) then "Single" at SINGLE_X
tick_positions = BATCH_XS + [SINGLE_X]
tick_labels    = [str(i) for i in BATCH_XS] + ["Single"]

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=11, fontweight="500")
ax.set_xlim(0.3, SINGLE_X + 0.7)

# Y-axis: tight range around the data
all_acc = [v for d in DATA.values() for v in d["batches"] + [d["single"]]]
y_lo = max(0.0, min(all_acc) - 0.05)
y_hi = min(1.0, max(all_acc) + 0.06)
ax.set_ylim(y_lo, y_hi)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
ax.tick_params(axis="y", labelsize=10)
ax.tick_params(axis="x", labelsize=10)

ax.set_xlabel("Batch", fontsize=12, fontweight="600", labelpad=8)
ax.set_ylabel("Accuracy (Test Set)", fontsize=12, fontweight="600", labelpad=8)
ax.set_title(
    "Incremental Batch Analysis",
    fontsize=14, fontweight="bold", pad=16, color="#1A1A1A",
)

# ── Grid styling ──────────────────────────────────────────────────────────────
ax.grid(axis="y", linestyle="-", alpha=0.15, linewidth=0.7, zorder=0)
ax.grid(axis="x", linestyle="-", alpha=0.08, linewidth=0.6, zorder=0)

# ── Spine styling ──────────────────────────────────────────────────────────────
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#CCCCCC")
ax.spines["bottom"].set_color("#CCCCCC")
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = Path(__file__).resolve().parent / "incremental_benchmark_accuracy.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {out_path}")
plt.show()
