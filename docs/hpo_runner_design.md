# HPO Runner Design Notes (`run_hpo_experiments.py`)

## Purpose
Document what the HPO runner does, why it was built this way, and how to interpret outputs.

File: `src/models_v2/run_hpo_experiments.py`

## High-Level Goal
Run per-model hyperparameter search under the evaluation protocol, then select and confirm one config per model/mode with policy-gate-aware logic.

## Why A Separate Runner
- Keeps HPO logic separate from baseline CV runner.
- Makes it easy to tune budget (`n_trials`, search CV, confirm CV) without changing core model files.
- Preserves reproducibility through explicit run-config artifact output.

## Core Flow
1. Load `train_split + val_split` for CV-only tuning (no test usage).
2. For each model family and mode:
   - sample `n_trials` hyperparameter configs
   - evaluate each config via search CV (`search_n_splits x search_n_repeats`)
3. Apply policy-gate-aware selection:
   - enforce floors (`accuracy`, `closed_precision`)
   - if gates pass: shortlist by PR-AUC top-band, tie-break by closed F1
   - if none pass: fallback to max closed F1 and document shortfall
4. Re-evaluate selected config with stronger confirmation CV (`confirm_n_splits x confirm_n_repeats`)
5. Save search, selected, and confirmed artifacts.

## Design Decisions

### 1) Per-Model Search
- Search spaces differ by model family.
- Keeps tuning interpretable and avoids mixing incompatible hyperparameter spaces.

### 2) Random Search First
- Chosen for simplicity and robust first-pass coverage.
- Good fit for laptop-scale budget and early exploration.
- Can be upgraded later to Optuna/Bayesian search after first-pass results.

### 3) Two-Stage Evaluation Integrity
- Uses each model’s stage-1 policy and full pipeline in CV scoring.
- For two-stage variants, stage-1 and stage-2 are both represented in metrics.

### 4) Gate-Aware Selection
- Aligns with `docs/eval_protocol.md`:
  - floors are pass/fail constraints
  - PR-AUC top-band + closed-F1 tie-break for gate-pass candidates
  - documented fallback when no gate-pass exists

### 5) Fast Search, Strong Confirm
- Search CV defaults to `5x1` for speed.
- Confirm CV defaults to `5x3` for stability before decisions.

### 6) Quiet Logging by Default
- Model-level logs are suppressed to make ETA/progress readable.
- Use `--show-model-logs` to re-enable verbose model output.

### 7) Runtime Visibility
- Progress prints include:
  - per-trial duration
  - average per-trial time
  - block ETA (current model/mode)
  - full-run ETA (all remaining trials)

## Search Spaces (Current)
- LR: `C`, `max_iter` (with balanced class weight)
- LightGBM: boosting/leaf/sampling/regularization parameters
- RandomForest: tree count/depth/split/feature-sampling parameters
- XGBoost: boosting/depth/sampling/regularization parameters

Rationales are documented inline in `_sample_params(...)`.

## Artifacts Produced
- `artifacts/hpo/hpo_search_trials.csv`
  - one row per trial with params and search-CV metrics
- `artifacts/hpo/hpo_selected_trials.csv`
  - selected trial per model/mode + rationale
- `artifacts/hpo/hpo_confirm_metrics.csv`
  - confirm-CV metrics for selected trials
- `artifacts/hpo/hpo_run_config.json`
  - run settings for reproducibility

## What This Runner Does Not Do (Yet)
- Threshold sweep per trial (uses fixed `--decision-threshold`)
- End-to-end runtime tie-break integration in final HPO selection
- Bayesian/Optuna search
- Automatic markdown summary generation from HPO artifacts

## Recommended Next Enhancements
1. Add optional threshold sweep on top-N search trials per model.
2. Add markdown report export (`artifacts/hpo/hpo_summary.md`).
3. Add optional Optuna backend after random-search baseline is established.
