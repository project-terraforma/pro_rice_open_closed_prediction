"""
Multi-metric incremental benchmark plots (warm-start) for alex-filtered SF/NY dataset.

Creates separate plots for:
  • Accuracy
  • Closed Precision
  • Closed Recall
  • F1 Score (Closed)
  • PR-AUC (Closed)

Each plot shows warm-start incremental (after batches 1–5) + single run dot.
X-axis label: "After batch k" — model has accumulated knowledge from batches 1…k.

Output: src/cumalitive_training/sf_ny_data/plots/incremental_benchmark_*.png
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Data from sf_ny_alex_incremental_results.md (warm-start training) ────────
# Each list entry = metrics after warm-starting through batches 1…k
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
        "precision":  [0.3970, 0.3981, 0.4006, 0.4033, 0.3922],
        "recall":     [0.8281, 0.8183, 0.8088, 0.8215, 0.8142],
        "f1":         [0.5367, 0.5356, 0.5358, 0.5410, 0.5294],
        "pr_auc":     [0.5464, 0.5357, 0.5357, 0.5420, 0.5462],
        "single_accuracy":  0.7845,
        "single_precision": 0.4022,
        "single_recall":    0.8359,
        "single_f1":        0.5431,
        "single_pr_auc":    0.5590,
    },
    "Random Forest": {
        "accuracy":   [0.7721, 0.7748, 0.7752, 0.7758, 0.7739],
        "precision":  [0.3878, 0.3908, 0.3914, 0.3920, 0.3901],
        "recall":     [0.8428, 0.8416, 0.8424, 0.8408, 0.8440],
        "f1":         [0.5312, 0.5337, 0.5345, 0.5347, 0.5336],
        "pr_auc":     [0.5186, 0.5182, 0.5187, 0.5197, 0.5196],
        "single_accuracy":  0.7697,
        "single_precision": 0.3855,
        "single_recall":    0.8465,
        "single_f1":        0.5297,
        "single_pr_auc":    0.5205,
    },
}

COLORS = {
    "LR":            "#4878CF",
    "XGBoost":       "#6ACC65",
    "LightGBM":      "#E53935",
    "Random Forest": "#FF8C00",
}

METRICS = [
    ("accuracy",  "Accuracy",           0.75, 0.87),
    ("precision", "Closed Precision",   0.00, 0.85),
    ("recall",    "Closed Recall",      0.00, 0.90),
    ("f1",        "F1 Score (Closed)",  0.00, 0.60),
    ("pr_auc",    "PR-AUC (Closed)",    0.50, 0.58),
]

N_BATCHES = 5
BATCH_XS  = list(range(1, N_BATCHES + 1))
SINGLE_X  = N_BATCHES + 1.5


def darken_color(hex_color, factor=0.6):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"#{max(0,int(r*factor)):02x}{max(0,int(g*factor)):02x}{max(0,int(b*factor)):02x}"


def compute_label_positions(values_dict, y_range, tolerance_frac=0.035):
    label_h = tolerance_frac * y_range
    ordered = sorted(values_dict.items(), key=lambda kv: kv[1])
    offsets = {}
    placed  = []
    for model, y_val in ordered:
        group = [py for _, py in placed if abs(y_val - py) < label_h * 1.2]
        if group:
            idx     = len(group)
            n_group = idx + 1
            x_off   = (idx - n_group / 2 + 0.5) * 0.13
        else:
            x_off = 0.0
        offsets[model] = x_off
        placed.append((model, y_val))
    return offsets


def create_metric_plot(metric_key, metric_label, y_min, y_max, output_dir):
    fig, ax = plt.subplots(figsize=(13, 7))
    single_key = f"single_{metric_key}"
    y_range = y_max - y_min

    for model in DATA:
        color = COLORS[model]
        vals  = DATA[model][metric_key]
        sing  = DATA[model][single_key]

        ax.plot(BATCH_XS, vals, marker="o", color=color, linewidth=2.5,
                markersize=7, alpha=0.85, zorder=3)
        ax.plot([N_BATCHES, SINGLE_X], [vals[-1], sing],
                linestyle="--", color=color, linewidth=1.2, alpha=0.6, zorder=2)
        ax.scatter([SINGLE_X], [sing], marker="o", s=80, color=color,
                   zorder=4, edgecolors="white", linewidths=0.8)

    # Value labels with collision avoidance
    for x_pos in BATCH_XS + [SINGLE_X]:
        if x_pos == SINGLE_X:
            vdict = {m: DATA[m][single_key]              for m in DATA}
        else:
            vdict = {m: DATA[m][metric_key][int(x_pos)-1] for m in DATA}

        offsets = compute_label_positions(vdict, y_range)
        for model, val in vdict.items():
            ax.text(x_pos + offsets[model], val, f"{val:.4f}",
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
    print(f"Saved → {out_path}")
    plt.close(fig)


def main():
    output_dir = Path(__file__).resolve().parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Creating incremental benchmark plots (warm-start)...")
    for metric_key, metric_label, y_min, y_max in METRICS:
        create_metric_plot(metric_key, metric_label, y_min, y_max, output_dir)
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
