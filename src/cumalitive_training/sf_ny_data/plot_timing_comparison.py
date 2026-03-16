"""
Timing comparison bar graphs: per-batch vs single-run for all 4 models.

Creates bar charts showing:
  - Batch 1-5 times (incremental training)
  - Single-run time (baseline)
  - One chart per model

Output: src/cumalitive_training/sf_ny_data/plots/timing_*.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Color palette (same as incremental benchmark plots)
COLORS = {
    "LR": "#4878CF",
    "RF": "#FF8C00",
    "XGBoost": "#6ACC65",
    "LightGBM": "#E53935",
}

# Timing data from sf_ny_alex_incremental_results.md
TIMING_DATA = {
    "LR": {
        "batches": [2.89, 0.66, 0.66, 0.67, 0.62],
        "single": 1.57,
    },
    "RF": {
        "batches": [0.56, 0.46, 0.40, 0.45, 0.42],
        "single": 1.33,
    },
    "XGBoost": {
        "batches": [0.32, 0.32, 0.33, 0.36, 0.41],
        "single": 0.84,
    },
    "LightGBM": {
        "batches": [1.01, 0.77, 0.82, 0.93, 1.11],
        "single": 1.66,
    },
}

MODEL_NAMES_DISPLAY = {
    "LR": "Logistic Regression",
    "RF": "Random Forest",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
}


def create_timing_bar_chart(model_key, model_display_name, timing_dict, output_dir):
    """Create a bar chart comparing batch times vs single-run time."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    
    batch_times = timing_dict["batches"]
    single_time = timing_dict["single"]
    
    # X positions
    batch_labels = [f"Batch {i}" for i in range(1, 6)]
    x_batch = np.arange(len(batch_labels))
    single_x = len(batch_labels) + 0.5
    
    # Colors: batches in model color, single-run in lighter gray
    batch_color = COLORS[model_key]
    single_color = "#CCCCCC"
    
    # Draw batch bars
    bars_batch = ax.bar(x_batch, batch_times, width=0.7, 
                        color=batch_color, alpha=0.85, 
                        edgecolor="white", linewidth=1.5, 
                        label="Per-batch (incremental)")
    
    # Draw single-run bar
    bars_single = ax.bar([single_x], [single_time], width=0.7, 
                         color=single_color, alpha=0.75, 
                         edgecolor="white", linewidth=1.5, 
                         label="Single-run (all data at once)")
    
    # Add value labels on bars
    for bar in bars_batch:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    
    for bar in bars_single:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    
    # Formatting
    ax.set_xticks(list(x_batch) + [single_x])
    ax.set_xticklabels(batch_labels + ["Single\nrun"], fontsize=11)
    ax.set_ylabel("Time (seconds)", fontsize=12, fontweight='bold')
    ax.set_title(f"Training Time Comparison — {model_display_name}",
                 fontsize=13, fontweight='bold', pad=14)
    
    # Add divider line between batches and single-run
    ax.axvline((len(batch_labels) - 1) / 2 + 2.5, color="#888888",
               linewidth=1.0, linestyle=":", alpha=0.6, zorder=1)
    
    # Styling
    ax.set_ylim(0, max(batch_times + [single_time]) * 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.set_facecolor("#FAFBFC")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(["Per-batch (incremental)", "Single-run"], 
              loc='upper left', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    out_path = output_dir / f"timing_{model_key.lower()}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {out_path}")
    plt.close(fig)


def create_combined_timing_chart(output_dir):
    """Create a combined bar chart with all models side-by-side."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    models = ["LR", "RF", "XGBoost", "LightGBM"]
    x = np.arange(6)  # 5 batches + 1 single-run
    width = 0.2
    
    # Collect total batch times for single-run comparison
    batch_data = {m: TIMING_DATA[m]["batches"] for m in models}
    single_data = {m: TIMING_DATA[m]["single"] for m in models}
    
    # Plot bars for each model
    for i, model in enumerate(models):
        batches = batch_data[model]
        single = single_data[model]
        
        # Batch bars
        ax.bar(x[:5] + i*width - 1.5*width, batches, width, 
               label=f"{model} (batches)", color=COLORS[model], alpha=0.8)
    
    # Single-run bars (grouped on the right)
    single_x = np.arange(len(models)) + 5.5
    for i, model in enumerate(models):
        ax.bar(single_x[i], single_data[model], width*4, 
               color=COLORS[model], alpha=0.5, edgecolor='black', linewidth=1)
    
    # Labels and formatting
    ax.set_ylabel("Time (seconds)", fontsize=12, fontweight='bold')
    ax.set_title("Training Time Comparison: All Models",
                 fontsize=13, fontweight='bold', pad=14)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS[m], alpha=0.8, edgecolor='black', label=m)
        for m in models
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.95)
    
    ax.set_facecolor("#FAFBFC")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    
    plt.tight_layout()
    out_path = output_dir / "timing_all_models_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {out_path}")
    plt.close(fig)


def main():
    output_dir = Path(__file__).resolve().parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating timing comparison bar charts...\n")
    
    # Create per-model timing charts
    for model_key, model_display_name in MODEL_NAMES_DISPLAY.items():
        create_timing_bar_chart(model_key, model_display_name, 
                               TIMING_DATA[model_key], output_dir)
    
    # Create combined comparison chart
    create_combined_timing_chart(output_dir)
    
    print("\nAll timing charts generated successfully!")


if __name__ == "__main__":
    main()
