"""
Random Forest incremental benchmarking: all metrics in one plot.

Creates a single plot showing:
  - Accuracy
  - Closed F1
  - Closed PR-AUC
  - Closed Recall

Each metric shown as a line across batches 1-5 + single-run dot.

Output: src/cumalitive_training/sf_ny_data/plots/rf_incremental_all_metrics.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Color palette for metrics (distinct, readable colors)
METRIC_COLORS = {
    "Accuracy": "#4878CF",      # Blue
    "F1 (Closed)": "#6ACC65",   # Green
    "PR-AUC (Closed)": "#E53935",  # Red
    "Recall (Closed)": "#FF8C00",   # Orange
    "Precision (Closed)": "#8E44AD", # Purple (closed precision)
}

# Random Forest data from sf_ny_alex_incremental_results.md
RF_DATA = {
    "accuracy": {
        "batches": [0.7721, 0.7748, 0.7752, 0.7758, 0.7739],
        "single": 0.7697,
        "y_min": 0.76,
        "y_max": 0.78,
    },
    "precision": {
        "batches": [0.3878, 0.3908, 0.3914, 0.3920, 0.3901],
        "single": 0.3855,
        "y_min": 0.38,
        "y_max": 0.40,
    },
    "f1": {
        "batches": [0.5312, 0.5337, 0.5345, 0.5347, 0.5336],
        "single": 0.5297,
        "y_min": 0.52,
        "y_max": 0.54,
    },
    "pr_auc": {
        "batches": [0.5186, 0.5182, 0.5187, 0.5197, 0.5196],
        "single": 0.5205,
        "y_min": 0.516,
        "y_max": 0.522,
    },
    "recall": {
        "batches": [0.8428, 0.8416, 0.8424, 0.8408, 0.8440],
        "single": 0.8465,
        "y_min": 0.84,
        "y_max": 0.85,
    },
}

METRICS = [
    ("accuracy", "Accuracy", RF_DATA["accuracy"]["y_min"], RF_DATA["accuracy"]["y_max"]),
    ("precision", "Precision (Closed)", RF_DATA["precision"]["y_min"], RF_DATA["precision"]["y_max"]),
    ("f1", "F1 (Closed)", RF_DATA["f1"]["y_min"], RF_DATA["f1"]["y_max"]),
    ("pr_auc", "PR-AUC (Closed)", RF_DATA["pr_auc"]["y_min"], RF_DATA["pr_auc"]["y_max"]),
    ("recall", "Recall (Closed)", RF_DATA["recall"]["y_min"], RF_DATA["recall"]["y_max"]),
]

N_BATCHES = 5
BATCH_XS = list(range(1, N_BATCHES + 1))
SINGLE_X = N_BATCHES + 1.5


def darken_color(hex_color, factor=0.7):
    """Darken a hex color by multiplying RGB values."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def compute_label_positions(values_dict, y_range):
    """
    Compute vertical offsets to avoid label collisions.
    Returns dict of {metric: offset}
    """
    sorted_vals = sorted(values_dict.items(), key=lambda x: x[1])
    offsets = {}
    label_height = y_range * 0.02
    
    for i, (metric, val) in enumerate(sorted_vals):
        # Stack labels vertically based on their value
        offset = (i - len(sorted_vals) / 2) * label_height
        offsets[metric] = offset
    
    return offsets


def create_rf_combined_metrics_plot():
    """Create combined plot with all RF metrics."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Y-axis range (use widest range to fit all metrics on one scale)
    y_min_global = 0.30
    y_max_global = 0.85
    
    # Plot lines for each metric
    for metric_key, metric_label, y_min, y_max in METRICS:
        metric_data = RF_DATA[metric_key]
        batch_vals = metric_data["batches"]
        single_val = metric_data["single"]
        color = METRIC_COLORS[metric_label]
        
        # Plot incremental line (batches 1-5)
        ax.plot(BATCH_XS, batch_vals, 
               color=color, linewidth=2.5, marker='o', markersize=8,
               label=metric_label, alpha=0.85, zorder=3)
        
        # Plot single-run dot
        ax.scatter([SINGLE_X], [single_val], 
                  color=color, s=120, marker='D', alpha=0.85, zorder=4)
    
    # Add value labels on data points
    for metric_key, metric_label, _, _ in METRICS:
        metric_data = RF_DATA[metric_key]
        batch_vals = metric_data["batches"]
        single_val = metric_data["single"]
        color = METRIC_COLORS[metric_label]
        
        # Batch labels
        for x_pos, val in zip(BATCH_XS, batch_vals):
            ax.text(x_pos, val + 0.003, f"{val:.4f}",
                   fontsize=7.5, ha="center", va="bottom",
                   bbox=dict(boxstyle="round,pad=0.25",
                            facecolor=darken_color(color, 0.65),
                            edgecolor="white", linewidth=0.5, alpha=0.9),
                   color="white", fontweight="bold", zorder=5)
        
        # Single-run label
        ax.text(SINGLE_X, single_val + 0.003, f"{single_val:.4f}",
               fontsize=7.5, ha="center", va="bottom",
               bbox=dict(boxstyle="round,pad=0.25",
                        facecolor=darken_color(color, 0.65),
                        edgecolor="white", linewidth=0.5, alpha=0.9),
               color="white", fontweight="bold", zorder=5)
    
    # Add dotted lines connecting batch 5 to single-run for each metric
    for metric_key, metric_label, _, _ in METRICS:
        metric_data = RF_DATA[metric_key]
        batch_5_val = metric_data["batches"][-1]
        single_val = metric_data["single"]
        color = METRIC_COLORS[metric_label]
        
        ax.plot([BATCH_XS[-1], SINGLE_X], [batch_5_val, single_val],
               color=color, linewidth=1.5, linestyle=":", alpha=0.6, zorder=2)
    
    # Divider line
    ax.axvline((N_BATCHES + SINGLE_X) / 2, color="#888888",
              linewidth=1.0, linestyle=":", alpha=0.7, zorder=1)
    
    # Formatting
    ax.set_xticks(BATCH_XS + [SINGLE_X])
    ax.set_xticklabels([str(i) for i in BATCH_XS] + ["Single\nrun"], fontsize=11)
    ax.set_xlim(0.4, SINGLE_X + 0.6)
    ax.set_ylim(y_min_global, y_max_global)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.3f}"))
    ax.set_xlabel("Batch", fontsize=12, fontweight='bold')
    ax.set_ylabel("Metric Value", fontsize=12, fontweight='bold')
    ax.set_title("Random Forest Incremental Benchmarking — All Metrics",
                fontsize=14, fontweight='bold', pad=14)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_facecolor("#FAFBFC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    output_dir = Path(__file__).resolve().parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "rf_incremental_all_metrics.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


def main():
    print("Creating Random Forest incremental benchmarking plot (all metrics)...\n")
    create_rf_combined_metrics_plot()
    create_rf_legend()
    print("Plots generated successfully!")


def create_rf_legend():
    """Create separate vertically stacked legend for RF metrics (one column).
    """
    # Taller figure to accommodate four rows stacked vertically
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.axis('off')

    # Create legend entries vertically (1 column)
    legend_elements = [
        plt.Line2D([0], [0], color=METRIC_COLORS[metric], linewidth=3,
                   marker='o', markersize=8, label=metric)
        for metric in ["Accuracy", "Precision (Closed)", "F1 (Closed)", "PR-AUC (Closed)", "Recall (Closed)"]
    ]

    # Center the legend and arrange in 1 column (vertical stack)
    legend = ax.legend(handles=legend_elements, loc='center',
                       ncol=1, fontsize=11, frameon=False,
                       handlelength=2, columnspacing=1)

    output_dir = Path(__file__).resolve().parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "rf_incremental_all_metrics_legend.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
