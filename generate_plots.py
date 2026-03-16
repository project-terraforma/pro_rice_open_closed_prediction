#!/usr/bin/env python3
"""
Generate incremental benchmark plots for alex-filtered SF/NY dataset.
Standalone script with no imports from src/
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Data from incremental results ────────
DATA = {
    "LR": {
        "accuracy":   [0.8190, 0.8097, 0.8085, 0.8130, 0.8151],
        "precision":  [0.4464, 0.4319, 0.4298, 0.4368, 0.4398],
        "recall":     [0.7548, 0.7675, 0.7659, 0.7618, 0.7560],
        "f1":         [0.5610, 0.5528, 0.5506, 0.5552, 0.5561],
        "pr_auc":     [0.5256, 0.5264, 0.5277, 0.5199, 0.5256],
        "single_accuracy":  0.8129,
        "single_precision": 0.4367,
        "single_recall":    0.7622,
        "single_f1":        0.5552,
        "single_pr_auc":    0.5263,
    },
    "XGBoost": {
        "accuracy":   [0.8541, 0.8543, 0.8539, 0.8546, 0.8565],
        "precision":  [0.7164, 0.6776, 0.6840, 0.6911, 0.7305],
        "recall":     [0.0786, 0.0929, 0.0860, 0.0925, 0.0999],
        "f1":         [0.1416, 0.1634, 0.1527, 0.1632, 0.1757],
        "pr_auc":     [0.5440, 0.5336, 0.5323, 0.5333, 0.5450],
        "single_accuracy":  0.8490,
        "single_precision": 0.8070,
        "single_recall":    0.0188,
        "single_f1":        0.0368,
        "single_pr_auc":    0.5622,
    },
    "LightGBM": {
        "accuracy":   [0.7810, 0.7827, 0.7853, 0.7865, 0.7783],
        "precision":  [0.5108, 0.5153, 0.5252, 0.5301, 0.5201],
        "recall":     [0.6263, 0.6428, 0.6531, 0.6556, 0.6407],
        "f1":         [0.5627, 0.5753, 0.5860, 0.5892, 0.5778],
        "pr_auc":     [0.5366, 0.5400, 0.5430, 0.5442, 0.5420],
        "single_accuracy":  0.7845,
        "single_precision": 0.5187,
        "single_recall":    0.6389,
        "single_f1":        0.5723,
        "single_pr_auc":    0.5411,
    },
    "RF": {
        "accuracy":   [0.7721, 0.7724, 0.7735, 0.7747, 0.7739],
        "precision":  [0.5038, 0.5047, 0.5090, 0.5117, 0.5098],
        "recall":     [0.6191, 0.6198, 0.6264, 0.6308, 0.6288],
        "f1":         [0.5570, 0.5586, 0.5639, 0.5685, 0.5668],
        "pr_auc":     [0.5294, 0.5301, 0.5326, 0.5347, 0.5336],
        "single_accuracy":  0.7697,
        "single_precision": 0.5018,
        "single_recall":    0.6144,
        "single_f1":        0.5535,
        "single_pr_auc":    0.5279,
    }
}

COLORS = {
    "LR": "#4878CF",
    "XGBoost": "#6ACC65",
    "LightGBM": "#E53935",
    "RF": "#FF8C00",
}

METRICS = [
    ("accuracy", "Accuracy", 0.75, 0.87),
    ("precision", "Closed Precision", 0.00, 0.85),
    ("recall", "Closed Recall", 0.00, 0.90),
    ("f1", "F1 Score (Closed)", 0.00, 0.60),
    ("pr_auc", "PR-AUC (Closed)", 0.50, 0.58),
]

N_BATCHES = 5
SINGLE_X = 6.5
BATCH_XS = list(range(1, N_BATCHES + 1))


def darken_color(hex_color, factor=0.7):
    """Darken a hex color by multiplying RGB values."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r = max(0, int(r * factor))
    g = max(0, int(g * factor))
    b = max(0, int(b * factor))
    return f"#{r:02x}{g:02x}{b:02x}"


def compute_label_positions(vdict, y_range):
    """Compute vertical offsets to avoid label collisions."""
    values = sorted(vdict.values())
    offsets = {}
    collision_threshold = y_range * 0.025
    
    for model in vdict:
        offset = 0
        for other_model in vdict:
            if model != other_model:
                if abs(vdict[model] - vdict[other_model]) < collision_threshold:
                    if vdict[model] < vdict[other_model]:
                        offset -= collision_threshold
                    else:
                        offset += collision_threshold
        offsets[model] = offset
    
    return offsets


def create_metric_plot(metric_key, metric_label, y_min, y_max, output_dir):
    """Create a single metric plot with incremental + single-run."""
    y_range = y_max - y_min
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Incremental lines
    for model in ["LR", "RF", "XGBoost", "LightGBM"]:
        values = DATA[model][metric_key]
        ax.plot(BATCH_XS, values, color=COLORS[model], linewidth=2.5,
                marker='o', markersize=6, label=model, zorder=3)
    
    # Single-run dots
    for model in ["LR", "RF", "XGBoost", "LightGBM"]:
        single_key = f"single_{metric_key}"
        val = DATA[model][single_key]
        ax.scatter([SINGLE_X], [val], color=COLORS[model], s=150, zorder=4,
                  edgecolors='white', linewidth=1.5, marker='D')
    
    # Labels with collision avoidance
    for x_pos in BATCH_XS:
        vdict = {m: DATA[m][metric_key][int(x_pos)-1] for m in DATA}
        offsets = compute_label_positions(vdict, y_range)
        for model, val in vdict.items():
            ax.text(x_pos + offsets[model], val, f"{val:.4f}",
                    fontsize=7.5, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor=darken_color(COLORS[model], 0.65),
                              edgecolor="white", linewidth=0.5, alpha=0.9),
                    color="white", fontweight="bold", zorder=5)
    
    # Single-run labels
    vdict = {m: DATA[m][f"single_{metric_key}"] for m in DATA}
    offsets = compute_label_positions(vdict, y_range)
    for model, val in vdict.items():
        ax.text(SINGLE_X + offsets[model], val, f"{val:.4f}",
                fontsize=7.5, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=darken_color(COLORS[model], 0.65),
                          edgecolor="white", linewidth=0.5, alpha=0.9),
                color="white", fontweight="bold", zorder=5)
    
    # Divider
    ax.axvline((N_BATCHES + SINGLE_X) / 2, color="#888888",
               linewidth=1.0, linestyle=":", alpha=0.7, zorder=1)
    
    ax.set_xticks(BATCH_XS + [SINGLE_X])
    ax.set_xticklabels([f"After {i}" for i in BATCH_XS] + ["Single\nrun"], fontsize=10)
    ax.set_xlim(0.4, SINGLE_X + 0.6)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
    ax.set_xlabel("Warm-Start Step  (model accumulates knowledge across batches)",
                  fontsize=11, fontweight="bold")
    ax.set_ylabel(metric_label, fontsize=12, fontweight="bold")
    ax.set_title(f"Incremental Warm-Start Benchmark — {metric_label} (Alex-filtered SF/NY)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_facecolor("#FAFBFC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    out_path = output_dir / f"incremental_benchmark_{metric_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved → {out_path}")
    plt.close(fig)


def main():
    output_dir = Path("/Users/claricepark/Desktop/pro_rice_open_closed_prediction/src/cumulative_training/sf_ny_data/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Creating incremental benchmark plots (warm-start)...")
    print(f"Output directory: {output_dir}")
    
    for metric_key, metric_label, y_min, y_max in METRICS:
        try:
            create_metric_plot(metric_key, metric_label, y_min, y_max, output_dir)
        except Exception as e:
            print(f"✗ Error creating {metric_key} plot: {e}")
    
    # Verify files were created
    created_files = list(output_dir.glob("*.png"))
    print(f"\nGeneration complete! Created {len(created_files)} plot files:")
    for f in sorted(created_files):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
