# Open/Closed Place Prediction

## Team
- Clarice Park
- Matthew Kimotsuki

## Overview

This repository contains multiple lines of work around predicting whether places are `open` or `closed` from Overture-style data and related derived signals.

The current repo is organized around:

- a shared `v2` modeling foundation
- a `ceiling study` track focused on schema-native performance limits
- an `incremental training / benchmarking` track
- archived historical labeling, modeling, and exploratory work

## Current Status

- `shared v2 modeling foundation`
  - reusable modeling code, featurizers, feature bundles, configs, and evaluation helpers
  - main code: [`src/models_v2/README.md`](src/models_v2/README.md)
- `ceiling study`
  - active
  - current confirmed diagnostic leader: `RandomForest`, `single`, `v2_rf_single_no_spatial_prior`
  - main docs: [`docs/README.md`](docs/README.md)
  - main artifacts: [`artifacts/README.md`](artifacts/README.md)
- `incremental training / benchmarking`
  - active
  - findings summary: `pending workstream-owner summary`
  - current landing page: [`docs/incremental_benchmarking/README.md`](docs/incremental_benchmarking/README.md)
  - main areas:
    - `src/incremental_benchmarking/`
    - `src/cumulative_training/`
- `archive / historical`
  - older v1 modeling, labeling, label-coverage, and exploratory work kept for reference

## Where To Start

- If you want the current ceiling-study path:
  - [`docs/README.md`](docs/README.md)
  - [`src/models_v2/README.md`](src/models_v2/README.md)
  - [`artifacts/README.md`](artifacts/README.md)
- If you want the repo-wide status map:
  - [`docs/WORKSTREAMS.md`](docs/WORKSTREAMS.md)
- If you want the incremental-training / benchmarking work:
  - [`docs/incremental_benchmarking/README.md`](docs/incremental_benchmarking/README.md)
  - `src/incremental_benchmarking/`
  - `src/cumulative_training/`

## Repository Layout

- `src/models_v2/`
  - shared v2 modeling foundation used by the active workstreams
- `src/incremental_benchmarking/`
  - current incremental benchmarking work
- `src/cumulative_training/`
  - current cumulative-training work
- `src/archive/models/`
  - older v1 modeling code kept mainly for reference
- `docs/`
  - repo navigation, protocol, rationale, and results summaries
- `artifacts/`
  - generated outputs across workstreams
- `data/`
  - train/val/test parquet splits and supporting files

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ceiling-Study Docs

Use these for the current schema-native v2 study:

- [`docs/ceiling_study/README.md`](docs/ceiling_study/README.md)
- [`docs/ceiling_study/eval_protocol.md`](docs/ceiling_study/eval_protocol.md)
- [`docs/ceiling_study/hpo_results_summary.md`](docs/ceiling_study/hpo_results_summary.md)
- [`docs/ceiling_study/feature_importance_results.md`](docs/ceiling_study/feature_importance_results.md)
- [`docs/ceiling_study/feature_bundle_v2_conventions.md`](docs/ceiling_study/feature_bundle_v2_conventions.md)

## Incremental Workstream Docs

- [`docs/incremental_benchmarking/README.md`](docs/incremental_benchmarking/README.md)

## Cumulative Training Docs

Use these for the cumulative / incremental training experiments and reproducibility:

- [`docs/cumulative_training/README.md`](docs/cumulative_training/README.md)
  - dataset-focused run instructions and quick findings for SF/NY experiments
- [`docs/incremental_benchmarking/INCREMENTAL_FINDINGS.md`](docs/incremental_benchmarking/INCREMENTAL_FINDINGS.md)
  - curated findings, reproduction notes, and practical recommendations
- `src/cumulative_training/sf_ny_data/BENCHMARK_SUMMARY.md`
  - detailed numeric summary and timing notes for the SF/NY alex-filtered experiments
- `src/cumulative_training/sf_ny_data/run_incremental_benchmark_sf_ny.py`
  - dataset-specific driver that created the SF/NY incremental results
- `src/incremental_benchmarking/run_incremental_benchmark_all_models.py`
  - generic project_c-style incremental driver (per-batch vs single-run comparisons)

## Notes

- The top-level README is intentionally brief and repo-level.
- Track-specific details should live in the track docs rather than in this file.
- The incremental-training / benchmarking findings section above is a placeholder for a future summary.
