# Incremental Benchmarking Docs

This directory is the landing page for the current `incremental training / benchmarking` workstream.

## Start Here

1. `docs/cumulative_training/README.md`
   - quick run instructions and high-level findings for dataset-specific experiments
2. `src/incremental_benchmarking/run_incremental_benchmark.py`
   - per-model incremental driver and example flags
3. `src/incremental_benchmarking/run_incremental_benchmark_all_models.py`
   - aggregate driver that runs all supported models and collates results
4. `src/cumulative_training/sf_ny_data/run_incremental_benchmark_sf_ny.py`
   - SF/NY dataset-specific driver used to produce the alex-filtered experiments

## Design / Workflow Docs

- incremental pattern: warm-start per batch using `warm_start` (scikit-learn), `xgb_model` (XGBoost) or `init_model` (LightGBM)
- featurization: `src/models_v2/shared_featurizer.py` — SharedPlaceFeaturizer expected nested place schema and named feature bundles (e.g. `low_plus_medium`)

## Supporting Files

- dataset batches: `data/sf_ny/batches/` (test_set.csv, batch_1..batch_5.csv)
- dataset-specific outputs: `src/cumulative_training/sf_ny_data/` (models_persistence, BENCHMARK_SUMMARY.md, plots)
- generic plotting: `src/incremental_benchmarking/plot_incremental_metrics.py`, `plot_incremental_benchmark.py`

## Related Adjacent Docs

- `../cumulative_training/README.md` — dataset-focused run instructions and quick findings
- `../WORKSTREAMS.md` — repo-wide workstream map and recommended reading order
- `../ceiling_study/README.md` — ceiling-study docs (distinct experimental track that reuses `models_v2` foundation)
