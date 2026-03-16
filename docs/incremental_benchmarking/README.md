# Incremental Benchmarking — concise guide and findings

This folder collects the practical artifacts and guidance for the incremental (warm‑start) training and benchmarking workstream. The content below summarizes key findings, points readers at the scripts to inspect/run first, lists the output folders that matter for review, and corrects/clarifies how this work relates to the shared `models_v2` foundation.

Summary findings (short)

- Incremental (warm‑start) training produces metrics that are generally comparable to single‑run training, with small gains for some models (Random Forest and XGBoost in some splits) and small losses for others (LightGBM in our experiments). See `src/cumulative_training/sf_ny_data/BENCHMARK_SUMMARY.md` for detailed numbers.
- Tradeoff: incremental training increases wall‑time (multiple fit cycles) but provides model persistence and the ability to continue training from intermediate states — valuable for production retraining workflows.
- Random Forest showed the best incremental efficiency (lower overhead) in our tests; XGBoost incremental runs improved closed‑class recall/F1 in the alex‑filtered SF/NY experiments. Logistic Regression performed similarly between approaches but incurred the largest relative overhead.

Where to start (scripts to read / run — recommended order)

1. Read the summary artifact: `src/cumulative_training/sf_ny_data/BENCHMARK_SUMMARY.md` (high‑level results and tradeoffs).
2. Review the data‑specific driver: `src/cumulative_training/sf_ny_data/run_incremental_benchmark_sf_ny.py` — shows end‑to‑end batching, featurization and persistence choices for the SF/NY dataset.
3. Inspect the more generic benchmarking drivers in `src/incremental_benchmarking/`:
   - `run_incremental_benchmark.py` (per‑model driver)
   - `run_incremental_benchmark_all_models.py` (aggregates models and comparisons)
4. Read the shared featurizer and models foundation: `src/models_v2/shared_featurizer.py` and `src/models_v2/README.md` (featurization is applied once per training pool and reused across batches).
5. Reproduce plots and artifacts: look at plotting scripts in `src/incremental_benchmarking/` and `src/cumulative_training/sf_ny_data/plot_*` to regenerate figures referenced in the summary.

Which output folders/results matter most (priority)

- `src/cumulative_training/sf_ny_data/models_persistence/` — serialized intermediate models created per batch (important for any warm‑start reproduction and inspection).
- `data/sf_ny/batches/` — batch CSVs and the test set used for benchmarking (the canonical split for the experiments).
- `src/cumulative_training/sf_ny_data/` — `BENCHMARK_SUMMARY.md`, `sf_ny_incremental_results.md` and the `plots/` subfolder (primary experiment artifacts and visualizations).
- `src/incremental_benchmarking/` — drivers and plotting utilities used across datasets (useful reference for methodology).
- `artifacts/` — project‑level artifacts and additional benchmark outputs (useful but secondary to the dataset‑specific folders above).

Corrections & clarifications vs. `models_v2` foundation

- This workstream relies on `models_v2` for feature engineering and consistent transform artifacts. The incremental drivers expect the `SharedPlaceFeaturizer` API (nested place schema, named feature bundles such as `low_plus_medium`) and a single `StandardScaler` fit on the training pool to be reused across batches.
- Warm‑start behaviour is model specific and implemented differently across libraries: scikit‑learn `warm_start` (LR, RF), XGBoost `xgb_model=prev_booster`, and LightGBM `init_model=prev_model`. The README and drivers document the approach used per model.
- Naming note: the code currently lives under `src/cumulative_training/` (typo). Renaming to `src/cumulative_training/` is sensible, but it is a structural change that touches many files and import paths — do this as a small, dedicated PR to avoid merge risk. All references should be updated together when performing that rename.

Quick reproduction notes

- To reproduce the SF/NY benchmark, follow the ordered checklist above and run the dataset driver (`run_incremental_benchmark_sf_ny.py`) with the `data/sf_ny/` CSVs present. Use the plotting scripts afterwards to regenerate the PNGs referenced in the summary.
- When reproducing, ensure `PYTHONPATH` (or sys.path) includes the repo root so `src/models_v2` imports resolve as in the drivers.

If you want, I can (separately) prepare a focused, single PR that renames `src/cumulative_training` → `src/cumulative_training` and updates all import references. That change is best isolated from documentation/content updates to minimize merge conflicts and review surface.
