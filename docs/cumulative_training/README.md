# Cumulative Training Docs

This directory contains the source-of-truth documentation and pointers for the cumulative / incremental training workstream (dataset drivers, persistence artifacts and plotting utilities).

## Start Here

1. `src/cumulative_training/sf_ny_data/run_incremental_benchmark_sf_ny.py`
   - dataset-specific driver that creates batches, runs incremental warm-start training for LR/RF/XGBoost/LightGBM, saves persisted models, and emits results markdown + plots
2. `src/cumulative_training/sf_ny_data/BENCHMARK_SUMMARY.md`
   - high-level numeric summary, timing, and persistence notes for the SF/NY alex-filtered experiments
3. `src/cumulative_training/sf_ny_data/models_persistence/`
   - serialized per-batch models (.pkl) used to resume or inspect warm-started models
4. `src/incremental_benchmarking/`
   - generic drivers and plotting utilities used across datasets (per-model drivers, aggregation, plotting helpers)

## Design / Workflow Docs

- `src/models_v2/shared_featurizer.py` — primary featurizer used by the drivers; review feature bundles (e.g. `low_plus_medium`) and expected nested place schema
- Incremental / warm-start patterns in drivers:
  - scikit-learn `warm_start` (LogisticRegression, RandomForest)
  - XGBoost continuation via `xgb_model=prev_booster`
  - LightGBM continuation via `init_model=prev_model`

## Supporting Files

- Data splits and canonical batches: `data/sf_ny/batches/` (test_set.csv, batch_1.csv .. batch_5.csv)
- Plots and visual artifacts: `src/cumulative_training/sf_ny_data/plots/`
- Full experiment artifacts and outputs: `artifacts/` (see `artifacts/README.md`)

## Related Adjacent Docs

- `../incremental_benchmarking/README.md` — higher-level incremental benchmarking guide and pointers
- `../WORKSTREAMS.md` — repo-wide workstream map and recommended reading order
- `../ceiling_study/README.md` — ceiling-study docs (different experimental track that shares `models_v2` foundation)

## Quick findings (summary)

- Random Forest (RF) is the strongest candidate for a production warm‑start approach in our experiments: it provided the best balance of incremental efficiency (lowest overhead ratio vs single‑run), stable metrics across batches, and reasonable model size. See `src/cumulative_training/sf_ny_data/BENCHMARK_SUMMARY.md` for details.
- XGBoost showed improved recall/F1 for the closed class in some alex‑filtered splits, but higher variability in precision and slightly higher incremental overhead.
- LightGBM performed slightly better in single‑run metrics in our SF/NY experiments but incurred larger persistence size and higher warm‑start overhead in incremental mode.
- Logistic Regression is easy to persist and interpret but had the largest relative time overhead for incremental training.

## How to run (quick)

1. Create a Python environment and install requirements:
   - python -m venv .venv
   - source .venv/bin/activate
   - pip install -r requirements.txt

2. Ensure repo root is on PYTHONPATH (or run via a script that adjusts sys.path). Example:
   - PYTHONPATH=$(pwd) python src/cumulative_training/sf_ny_data/run_incremental_benchmark_sf_ny.py

3. Regenerate plots after a run (example):
   - PYTHONPATH=$(pwd) python src/incremental_benchmarking/plot_incremental_metrics.py --results-dir src/cumulative_training/sf_ny_data/

## Notes and recommendations

- Model persistence: inspect `src/cumulative_training/sf_ny_data/models_persistence/` to load intermediate models (joblib) and compare decision thresholds.
- Reproducibility: use the shipped batch CSVs in `data/sf_ny/batches/` to reproduce exact training splits used in the report.
- Renaming note: the repository previously used a misspelled folder name `cumalitive_training`. That has been corrected to `cumulative_training` — references in docs and scripts were updated.
