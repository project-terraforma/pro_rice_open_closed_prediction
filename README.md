# Open/Closed Place Prediction

## Team
- Clarice Park
- Matthew Kimotsuki

## Overview

This repository contains multiple lines of work around predicting whether places are `open` or `closed` from Overture-style data and related derived signals.

At a high level, the repo separates two main questions:

- how well can we predict `open` vs `closed` using the available data together with the schema-native features we engineered from it?
- `placeholder for workstream-owner summary`

The current repo is organized around:

- a shared `v2` modeling foundation
- a `ceiling study` track focused on whether low- and medium-cost schema-native features could already meet the production gate on the provided Project C sample data
- an `incremental training / benchmarking` track
- archived historical labeling, modeling, and exploratory work

In this repo, `ceiling study` means the main evaluation track for testing whether low- and medium-cost schema-native features alone were enough to reach the project's production gate on the provided Project C sample data. The `low`, `medium`, and `high` cost definitions for these engineered features are documented in [`docs/ceiling_study/feature_inventory.csv`](docs/ceiling_study/feature_inventory.csv) and [`docs/ceiling_study/feature_rationale.md`](docs/ceiling_study/feature_rationale.md). The idea was to train on part of that dataset and evaluate on a holdout split to see whether a relatively cheap ML approach already looked strong enough for this problem, or whether better performance would likely require more expensive features, a larger training set, or both. In practice, this study exposed meaningful limitations, especially on closed-place performance. One likely challenge is that the roughly `3k`-row sample is fairly spread out, which may make it harder for the model to learn a strong closed signal. As a result, the study suggests that low/medium-cost schema-native features alone may not be sufficient here, but it does not fully resolve whether the main bottleneck is feature cost, dataset size, or both.

Parallel explanation for the incremental training / benchmarking workstream: `placeholder for workstream-owner summary`

## Current Status

- `shared v2 modeling foundation`
  - reusable modeling code, featurizers, feature bundles, configs, and evaluation helpers
  - main code: [`src/models_v2/README.md`](src/models_v2/README.md)
- `ceiling study`
  - active
  - purpose: test whether low/medium-cost schema-native features could reach production-level performance on the provided Project C sample split
  - data scope: train/holdout evaluation on the provided Project C sample data, not a separately constructed dataset
  - current confirmed diagnostic leader: `RandomForest`, `single`, `v2_rf_single_no_spatial_prior`
  - main docs: [`docs/README.md`](docs/README.md)
  - main artifacts: [`artifacts/README.md`](artifacts/README.md)
- `incremental training / benchmarking`
  - active
  - purpose: `placeholder for workstream-owner high-level motivation`
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
  - use this path if you want the main answer to: `can low/medium-cost schema-native features on the Project C sample data reach the production gate, or would this likely require more data or more expensive features?`
- If you want the repo-wide status map:
  - [`docs/WORKSTREAMS.md`](docs/WORKSTREAMS.md)
- If you want the incremental-training / benchmarking work:
  - [`docs/incremental_benchmarking/README.md`](docs/incremental_benchmarking/README.md)
  - `src/incremental_benchmarking/`
  - `src/cumulative_training/`
  - high-level motivation: `placeholder for workstream-owner summary`

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

Use these for the current schema-native v2 study.

This is the main documentation path for the workstream that tests whether the current low/medium-cost schema-native feature setup can achieve production-level performance on the provided Project C sample data:

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
