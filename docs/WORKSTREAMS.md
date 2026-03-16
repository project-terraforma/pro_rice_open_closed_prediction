# Workstreams And Status

This repository contains multiple workstreams. This file exists to make the ownership and navigation boundaries explicit without forcing a full reorganization of the repo.

Use this file to answer three questions quickly:

- which parts of the repo are `active`
- which active areas share infrastructure vs. represent separate workstreams
- which areas should be treated as `archive / reference`

## Current Repo Map

- `Shared v2 Modeling Foundation`
  - status: active
  - primary owner: Matthew
  - code: [`../src/models_v2/README.md`](../src/models_v2/README.md)
  - purpose: shared modeling stack, featurizers, feature bundles, configs, and evaluation logic used by the active tracks
- `Ceiling Study`
  - status: active
  - primary owner: Matthew
  - docs: [`README.md`](README.md)
  - artifacts: [`../artifacts/README.md`](../artifacts/README.md)
  - purpose: estimate the current schema-native performance ceiling using the v2 foundation
- `Incremental Training / Benchmarking`
  - status: active, separate workstream
  - main areas:
    - `../src/incremental_benchmarking/`
    - `../src/cumulative_training/`
  - docs: [`incremental_benchmarking/README.md`](incremental_benchmarking/README.md)
  - note: this workstream is current and separate; it may reuse pieces of the shared v2 foundation, but it is not part of the main ceiling-study reading path
- `Archive / Historical`
  - status: reference only unless a local doc says otherwise
  - main areas:
    - `../src/archive/models/`
    - `../src/archive/matching_validation/`
    - `../data/archive/matching_validation/`
    - label-coverage and simulation-related outputs/docs
  - note: useful for historical context, not the first place to look for the current repo state

## Recommended Reading Order

If you want the current ceiling-study state:

1. [`../README.md`](../README.md)
2. [`README.md`](README.md)
3. [`ceiling_study/README.md`](ceiling_study/README.md)
4. [`../src/models_v2/README.md`](../src/models_v2/README.md)
5. [`../artifacts/README.md`](../artifacts/README.md)

If you want the incremental-training line of work:

1. start in `../src/models_v2/README.md` for shared modeling concepts
2. read [`incremental_benchmarking/README.md`](incremental_benchmarking/README.md)
3. then review `../src/incremental_benchmarking/`
4. then review `../src/cumulative_training/`

## Practical Guidance

- If you are trying to reproduce the most polished current results, follow the `ceiling study` path first.
- If you are trying to understand the repo as a collaboration between two active lines of work, treat `models_v2` as shared infrastructure and read the incremental workstream separately.
- If a path includes `archive`, assume it is for reference unless a current doc explicitly sends you there.

## Interpretation Rules

- `active`
  - currently maintained and part of a live workstream
- `separate current workstream`
  - current, distinct line of work that may share infrastructure with another active track
- `archive / historical`
  - preserved for reference; expect older assumptions, APIs, and conclusions
