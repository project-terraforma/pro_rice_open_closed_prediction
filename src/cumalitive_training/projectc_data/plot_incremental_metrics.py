"""
Multi-metric incremental benchmark plots (warm-start).

Creates separate plots for:
  • Accuracy
  • Closed Precision
  • Closed Recall
  • F1 Score (Closed)
  • PR-AUC (Closed)

Each plot shows warm-start incremental (after batches 1–5) + single run dot.
X-axis label: "After batch k" — model has accumulated knowledge from batches 1…k.

Output: src/incremental_benchmarking/incremental_benchmark_*.png
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Data from incremental_results.md (warm-start training) ───────────────────
# Each list entry = metrics after warm-starting through batches 1…k
DATA = {
    "LR": {
        "accuracy":   [0.8555, 0.8949, 0.8934, 0.8774, 0.8978],
        "precision":  [0.2429, 0.3200, 0.3214, 0.3158, 0.3158],
        "recall":     [0.2698, 0.1270, 0.1429, 0.2857, 0.0952],
        "f1":         [0.2556, 0.1818, 0.1978, 0.3000, 0.1463],
        "pr_auc":     [0.1893, 0.1792, 0.2093, 0.2012, 0.1669],
        "single_accuracy":  0.8555,
        "single_precision": 0.2632,
        "single_recall":    0.3175,
        "single_f1":        0.2878,
        "single_pr_auc":    0.1738,
    },
    "XGBoost": {
        "accuracy":   [0.9109, 0.9124, 0.9109, 0.9124, 0.9080],
        "precision":  [1.0000, 0.8000, 0.6667, 0.8000, 0.0000],
        "recall":     [0.0317, 0.0635, 0.0635, 0.0635, 0.0000],
        "f1":         [0.0615, 0.1176, 0.1159, 0.1176, 0.0000],
        "pr_auc":     [0.2385, 0.1994, 0.2253, 0.2495, 0.1854],
        "single_accuracy":  0.9095,
        "single_precision": 1.0000,
        "single_recall":    0.0159,
        "single_f1":        0.0312,
        "single_pr_auc":    0.2221,
    },
    "LightGBM": {
        "accuracy":   [0.8117, 0.3810, 0.7854, 0.7985, 0.7854],
        "precision":  [0.2054, 0.1068, 0.1957, 0.2093, 0.1957],
        "recall":     [0.3651, 0.7778, 0.4286, 0.4286, 0.4286],
        "f1":         [0.2629, 0.1877, 0.2687, 0.2813, 0.2687],
        "pr_auc":     [0.1668, 0.1799, 0.1794, 0.1716, 0.1895],
        "single_accuracy":  0.7664,
        "single_precision": 0.1911,
        "single_recall":    0.4762,
        "single_f1":        0.2727,
        "single_pr_auc":    0.2270,
    },
    "Random Forest": {
        "accuracy":   [0.8029, 0.7723, 0.8307, 0.8044, 0.8058],
        "precision":  [0.2188, 0.1921, 0.2523, 0.2248, 0.2266],
        "recall":     [0.4444, 0.4603, 0.4286, 0.4603, 0.4603],
        "f1":         [0.2932, 0.2710, 0.3176, 0.3021, 0.3037],
        "pr_auc":     [0.2166, 0.2004, 0.2176, 0.2266, 0.2317],
        "single_accuracy":  0.8175,
        "single_precision": 0.2075,
        "single_recall":    0.3492,
        "single_f1":        0.2604,
        "single_pr_auc":    0.2039,
    },
}

COLORS = {
    "LR":            "#4878CF",
    "XGBoost":       "#6ACC65",
    "LightGBM":      "#E53935",
    "Random Forest": "#FF8C00",
}

METRICS = [
    ("accuracy",  "Accuracy",           0.35, 0.97),
    ("precision", "Closed Precision",   0.00, 1.10),
    ("recall",    "Closed Recall",      0.00, 0.85),
    ("f1",        "F1 Score (Closed)",  0.00, 0.35),
    ("pr_auc",    "PR-AUC (Closed)",    0.10, 0.28),
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
    ax.set_title(f"Incremental Warm-Start Benchmark — {metric_label}",
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
    output_dir = Path(__file__).resolve().parent
    print("Creating incremental benchmark plots (warm-start)...")
    for metric_key, metric_label, y_min, y_max in METRICS:
        create_metric_plot(metric_key, metric_label, y_min, y_max, output_dir)
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
