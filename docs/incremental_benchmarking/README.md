# Incremental benchmarking — findings and pointers

This document summarizes the incremental (warm‑start) benchmarking work done on the SF/NY datasets and points to the code, artifacts and key results.

Short findings

- Incremental training yields metrics comparable to single‑run training in most cases; for our SF/NY experiments Random Forest (RF) was the best warm‑start candidate overall (good metric stability, lowest incremental overhead, reasonable serialized size).
- XGBoost sometimes improves recall/F1 for the closed class in alex‑filtered splits, but had higher precision variability and modest overhead.
- LightGBM slightly favored single‑run performance in our setup and had higher warm‑start overhead and model size.
- Logistic Regression is interpretable but had the largest relative time overhead for incremental training.

Recommended reading / scripts to run (order)

1. `docs/cumulative_training/README.md` — quick run instructions and high‑level findings.
2. `src/cumulative_training/sf_ny_data/BENCHMARK_SUMMARY.md` — detailed metrics, timing and persistence information for the SF/NY runs.
3. `src/cumulative_training/sf_ny_data/run_incremental_benchmark_sf_ny.py` — the dataset‑specific driver for reproducing the SF/NY benchmark.
4. `src/incremental_benchmarking/run_incremental_benchmark.py` and `src/incremental_benchmarking/run_incremental_benchmark_all_models.py` — generic drivers for per‑model or multi‑model benchmarking.
5. `src/models_v2/shared_featurizer.py` — review API expectations (feature bundles and nested place schema) used by the drivers.
6. Plotting scripts: `src/incremental_benchmarking/plot_incremental_metrics.py` and `src/cumulative_training/sf_ny_data/plot_*` to regenerate figures.

Most important output folders

- `src/cumulative_training/sf_ny_data/models_persistence/` — serialized models per batch (use for warm‑start inspection).
- `data/sf_ny/batches/` — canonical train/test splits and batch CSVs used for experiments.
- `src/cumulative_training/sf_ny_data/` — `BENCHMARK_SUMMARY.md`, results markdowns and `plots/`.
- `src/incremental_benchmarking/` — drivers and plotting utilities used across datasets.

Repro tips

- Ensure the repo root is on PYTHONPATH when running (so `src/models_v2` imports resolve).
- Use the provided batch CSVs to reproduce exact runs.
- If running on a different machine, set the same random seed for consistent batching.

If you want, I can also add a small helper script in `scripts/` to load a persisted model and print per‑class metrics — would you like that?