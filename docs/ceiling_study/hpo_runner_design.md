# HPO Runner Design Notes

## Purpose
Document what the HPO runner does, why it was built this way, and how to interpret outputs.

Primary (baseline/frozen): `src/models_v2/run_hpo_experiments.py`  
Variant (imbalance-knob pass): `src/models_v2/run_hpo_experiments_weighted.py`
Narrow Optuna pass: `src/models_v2/run_hpo_optuna_narrow.py`
LR micro Optuna pass: `src/models_v2/run_hpo_optuna_lr_micro.py`
RF micro Optuna pass: `src/models_v2/run_hpo_optuna_rf_micro.py`
K+threshold sweep runner: `src/models_v2/run_k_threshold_sweep.py`

## Baseline vs Variant

- `run_hpo_experiments.py`:
  - frozen baseline used for first-pass reported HPO results
  - preserves reproducibility of documented outputs
- `run_hpo_experiments_weighted.py`:
  - explores explicit class-imbalance knobs (`class_weight` / `scale_pos_weight`)
  - intended for second-pass comparison against baseline
- `run_hpo_optuna_narrow.py`:
  - narrowed Optuna pass on frontier models (`lr`, `rf`)
  - uses dual-gate selection (`production`, `diagnostic`) + confirm CV
  - rationale and ranges documented in `docs/ceiling_study/hpo_optuna_narrow_design.md`
- `run_hpo_optuna_lr_micro.py`:
  - LR-only micro refinement pass (`single`, `two-stage`) after narrowed Optuna
  - tighter ranges around current LR winners
  - same dual-gate selection and confirm-CV reporting
- `run_hpo_optuna_rf_micro.py`:
  - RF-only micro refinement pass (`single`) after narrowed Optuna
  - tighter ranges around current RF winner region
  - same dual-gate selection and confirm-CV reporting
- `run_k_threshold_sweep.py`:
  - model-agnostic sweep of shared-featurizer k controls and decision thresholds
  - phased modes: `k_coarse`, `k_narrow`, `threshold`, `confirm`, `full`
  - expects frozen configs from selected-trials CSV
  - supports LR-now / RF-later workflow with shared artifact format

## High-Level Goal
Run per-model hyperparameter search under the evaluation protocol, then select and confirm one config per model/mode/gate using the current dual-gate policy.

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
   - for current dual-gate runners, evaluate both `production` and `diagnostic`
   - production:
     - require `accuracy >= 0.90`, `closed_precision >= 0.70`, `closed_recall >= 0.05`
     - rank passers by `closed_precision`, then `closed_f1`, then `pr_auc_closed`, then `accuracy`
   - diagnostic:
     - require `accuracy >= 0.84`, `closed_precision >= 0.20`
     - rank passers by `closed_f1`, then `pr_auc_closed`, then `closed_precision`, then `accuracy`
   - if none pass, use the documented gate-specific fallback and record the shortfall
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
- Aligns with `docs/ceiling_study/eval_protocol.md`:
  - gates are pass/fail constraints first
  - production and diagnostic use different ranking priorities
  - documented fallback is required when no gate-pass exists

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

## Search Spaces (Baseline Runner)
- LR: `C`, `max_iter` (with balanced class weight)
- LightGBM: boosting/leaf/sampling/regularization parameters
- RandomForest: tree count/depth/split/feature-sampling parameters
- XGBoost: boosting/depth/sampling/regularization parameters

Rationales are documented inline in `_sample_params(...)`.

## Search Spaces (Weighted Variant)
- Includes class-imbalance knob sweeps:
  - LR/RF: `class_weight` (`None`, `balanced`, custom closed-upweight)
  - LightGBM: `class_weight` mode and/or `scale_pos_weight`
  - XGBoost: `scale_pos_weight`
- Use for second-pass feasibility/comparison runs after baseline is recorded.

## Artifacts Produced
- `artifacts/archive/hpo/hpo_search_trials.csv`
  - one row per trial with params and search-CV metrics
- `artifacts/archive/hpo/hpo_selected_trials.csv`
  - selected trial per model/mode + rationale
- `artifacts/archive/hpo/hpo_confirm_metrics.csv`
  - confirm-CV metrics for selected trials
- `artifacts/archive/hpo/hpo_run_config.json`
  - run settings for reproducibility

## What This Runner Does Not Do (Yet)
- Threshold sweep per trial (uses fixed `--decision-threshold`)
- End-to-end runtime tie-break integration in final HPO selection
- Bayesian/Optuna search
- Automatic markdown summary generation from HPO artifacts

Historical note:
- older baseline runners and docs may reference legacy single-floor logic or PR-AUC top-band tie-breaks
- current selection policy for active workflows is the dual-gate policy in `docs/ceiling_study/eval_protocol.md`

## Recommended Next Enhancements
1. Add optional threshold sweep on top-N search trials per model.
2. Add markdown report export (`artifacts/archive/hpo/hpo_summary.md`).
3. Extend narrowed Optuna pass to optional additional model families if frontier changes.
4. Add one command-level switch to run random, narrowed Optuna, and LR micro modes from a single entrypoint.
