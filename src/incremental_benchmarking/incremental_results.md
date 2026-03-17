Incremental Benchmarking — Findings and Practical Notes

This note synthesizes insights from the incremental benchmarking drivers in
`src/incremental_benchmarking/` (project_c-style runs) and the dataset-specific
SF/NY experiments in `src/cumulative_training/sf_ny_data/`.

High-level summary

- Random Forest (RF) is the strongest warm-start candidate across experiments:
  lower incremental overhead vs other learners, stable per-batch metrics, and
  moderate persistence size make RF the most practical choice for production
  warm-start/retraining workflows.
- XGBoost can improve closed-class recall and closed F1 in some dataset
  variants (alex-filtered splits) when trained incrementally; however, it shows
  higher variability in closed-class precision and modest additional overhead.
- LightGBM tended to favor single-run performance in our SF/NY experiments and
  produced larger persisted artifacts and higher incremental wall-time.
- Logistic Regression is easy to persist and interpret, but incurred the
  largest relative wall-time overhead for incremental training in our tests.

Where the evidence comes from

- `src/incremental_benchmarking/run_incremental_benchmark_all_models.py`
  (project_c-style driver) — comparison between incremental warm-start and
  single-run on a frozen test set; writes `src/incremental_benchmarking/incremental_results.md`.
- `src/cumulative_training/sf_ny_data/run_incremental_benchmark_sf_ny.py`
  — dataset-specific end-to-end driver for the alex-filtered SF/NY experiments;
  results and detailed summary are in `src/cumulative_training/sf_ny_data/BENCHMARK_SUMMARY.md`.
- Plotting utilities under `src/incremental_benchmarking/` and
  `src/cumulative_training/sf_ny_data/plot_*` generate the figures used to
  inspect per-batch metric trajectories and timing comparisons.

Practical recommendations

- If you need warm-start persistence + reasonable run-time, start with Random
  Forest (use scikit-learn `warm_start=True`). It offers the best tradeoff
  observed in our experiments.
- If closed-class recall is the priority, evaluate incremental XGBoost on a
  held-out, alex-filtered split — it improved recall/F1 in some runs.
- Use the single-run approach when you need minimal wall‑time for a one-off
  training job (single-run is faster than incremental in all models tested).

Important files & folders to inspect

- `src/incremental_benchmarking/incremental_results.md` — project_c incremental
  run summary (if you ran the project_c driver)
- `src/cumulative_training/sf_ny_data/BENCHMARK_SUMMARY.md` — SF/NY numeric
  summary, timing, persistence notes and per-model comparisons
- `src/cumulative_training/sf_ny_data/models_persistence/` — persisted .pkl
  artifacts per batch for the SF/NY runs
- `data/sf_ny/batches/` — canonical batch CSVs and test set used for the
  SF/NY experiments
- plotting scripts: `src/incremental_benchmarking/plot_incremental_metrics.py`,
  `src/cumulative_training/sf_ny_data/plot_*` for regenerating figures

How to reproduce (quick)

1. Ensure Python requirements installed (`pip install -r requirements.txt`) and
   the repo root is on `PYTHONPATH`.
2. For project_c-style runs (smaller example pool):
   - PYTHONPATH=$(pwd) python src/incremental_benchmarking/run_incremental_benchmark_all_models.py
   - results will be written to `src/incremental_benchmarking/incremental_results.md`
3. For the SF/NY alex-filtered experiments:
   - place the required CSVs under `data/sf_ny/`
   - PYTHONPATH=$(pwd) python src/cumulative_training/sf_ny_data/run_incremental_benchmark_sf_ny.py
   - inspect `src/cumulative_training/sf_ny_data/BENCHMARK_SUMMARY.md` and
     `src/cumulative_training/sf_ny_data/models_persistence/`

Notes and caveats

- Featurizer + scaler: the drivers fit a shared `SharedPlaceFeaturizer` and a
  `StandardScaler` once on the full training pool to ensure consistent feature
  mappings across batches. This is intentional to avoid column-mismatch
  problems between batches and the single-run baseline, but it does mean the
  featurizer sees the full training pool (not the test set).
- Warm-start semantics differ by library and require careful persistence logic:
  scikit-learn `warm_start`, XGBoost `xgb_model=prev_booster`, LightGBM
  `init_model=prev_model`.
- Hyperparameters were sourced from HPO artifacts — consult the HPO outputs
  in `artifacts/hpo*/` if you want to re-run with alternate configs.

If you'd like, I can also:

- extract the numeric per-batch metrics into a single CSV for easier analysis,
- add a small `scripts/inspect_persisted_model.py` helper to load a .pkl and
  print basic metrics on the test set.
