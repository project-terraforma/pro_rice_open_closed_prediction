Cumulative Training — how to run and quick findings

This folder documents the cumulative/incremental training experiments and practical guidance for reproducing the SF/NY benchmark and evaluating warm‑start candidates.

Quick findings (summary)

- Random Forest (RF) is the strongest candidate for a production warm‑start approach in our experiments: it provided the best balance of incremental efficiency (lowest overhead ratio vs single‑run), stable metrics across batches, and reasonable model size. See `src/cumulative_training/sf_ny_data/BENCHMARK_SUMMARY.md` for details.
- XGBoost showed improved recall/F1 for the closed class in some alex‑filtered splits, but higher variability in precision and slightly higher incremental overhead.
- LightGBM performed slightly better in single‑run metrics in our SF/NY experiments but incurred larger persistence size and higher warm‑start overhead in incremental mode.
- Logistic Regression is easy to persist and interpret but had the largest relative time overhead for incremental training.

Where the code and artifacts live

- Data and canonical splits: `data/sf_ny/batches/` (test_set.csv, batch_1..batch_5.csv)
- Dataset‑specific driver and artifacts: `src/cumulative_training/sf_ny_data/`
  - Key files: `run_incremental_benchmark_sf_ny.py`, `BENCHMARK_SUMMARY.md`, `models_persistence/`, `plots/`
- Generic drivers & plotting utilities: `src/incremental_benchmarking/` (per‑model drivers and plotting helpers)
- Shared featurization foundation: `src/models_v2/shared_featurizer.py` (must be on PYTHONPATH when running drivers)

How to run (quick)

1. Create a Python environment and install requirements:
   - python -m venv .venv
   - source .venv/bin/activate
   - pip install -r requirements.txt

2. Ensure repo root is on PYTHONPATH (or run via a script that adjusts sys.path). Example:
   - PYTHONPATH=$(pwd) python src/cumulative_training/sf_ny_data/run_incremental_benchmark_sf_ny.py

3. Regenerate plots after a run (example):
   - PYTHONPATH=$(pwd) python src/incremental_benchmarking/plot_incremental_metrics.py --results-dir src/cumulative_training/sf_ny_data/

Notes and recommendations

- Model persistence: inspect `src/cumulative_training/sf_ny_data/models_persistence/` to load intermediate models (joblib) and compare decision thresholds.
- Reproducibility: use the shipped batch CSVs in `data/sf_ny/batches/` to reproduce exact training splits used in the report.
- Renaming note: the repository previously used a misspelled folder name `cumalitive_training`. That has been corrected to `cumulative_training` — references in docs and scripts were updated.

If you want a short script to run a single model or to convert the saved .pkl models to a neutral format for inspection, ask and I will add an example script in `scripts/`.
