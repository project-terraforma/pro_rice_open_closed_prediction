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
  - findings summary: `TODO - add current summary and key results for this workstream`
  - main areas:
    - `src/incremental_benchmarking/`
    - `src/cumalitive_training/`
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
  - `src/incremental_benchmarking/`
  - `src/cumalitive_training/`

## Repository Layout

- `src/models_v2/`
  - shared v2 modeling foundation used by the active workstreams
- `src/incremental_benchmarking/`
  - current incremental benchmarking work
- `src/cumalitive_training/`
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

## Notes

- The top-level README is intentionally brief and repo-level.
- Track-specific details should live in the track docs rather than in this file.
- The incremental-training / benchmarking findings section above is a placeholder for a future summary.
