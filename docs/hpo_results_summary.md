# HPO Results Summary (Current Pass)

Date run: 2026-03-05  
Runner: `src/models_v2/run_hpo_experiments.py`  
Feature bundle: `low_plus_medium`  
Search budget: `40 trials/model`  
Search CV: `5x1`  
Confirm CV: `5x3`  
Decision threshold: `0.5`

## Artifact Sources

- `artifacts/hpo/lr_hpo/hpo_confirm_metrics.csv`
- `artifacts/hpo/lightgbm_hpo/hpo_confirm_metrics.csv`
- `artifacts/hpo/rf_hpo/hpo_confirm_metrics.csv`
- `artifacts/hpo/xgboost_hpo/hpo_confirm_metrics.csv`

## Confirmed Results (Sorted by Closed F1)

| Model | Mode | Accuracy Mean | Closed Precision Mean | Closed Recall Mean | Closed F1 Mean | PR-AUC Closed Mean |
|---|---|---:|---:|---:|---:|---:|
| LR | two-stage | 0.749 | 0.168 | 0.428 | 0.239 | 0.188 |
| LR | single | 0.695 | 0.155 | 0.525 | 0.239 | 0.139 |
| RF | single | 0.857 | 0.236 | 0.237 | 0.232 | 0.189 |
| LightGBM | single | 0.812 | 0.089 | 0.140 | 0.095 | 0.145 |
| LightGBM | two-stage | 0.833 | 0.089 | 0.115 | 0.085 | 0.145 |
| XGBoost | single | 0.883 | 0.113 | 0.040 | 0.059 | 0.144 |
| XGBoost | two-stage | 0.890 | 0.088 | 0.022 | 0.035 | 0.105 |

## First-Pass Hyperparameters Searched (Baseline Runner)

The first HPO pass used random search over the following parameters:

- LR:
  - `C` (log-uniform)
  - `max_iter`
  - `class_weight="balanced"` (fixed in first pass)
- LightGBM:
  - `n_estimators`
  - `learning_rate`
  - `num_leaves`
  - `subsample`
  - `colsample_bytree`
  - `min_child_samples`
  - `reg_lambda`
  - `class_weight="balanced"` (fixed in first pass)
- RandomForest:
  - `n_estimators`
  - `max_depth`
  - `min_samples_leaf`
  - `min_samples_split`
  - `max_features`
  - `class_weight="balanced"` (fixed in first pass)
- XGBoost:
  - `n_estimators`
  - `learning_rate`
  - `max_depth`
  - `subsample`
  - `colsample_bytree`
  - `min_child_weight`
  - `reg_lambda` (no explicit imbalance knob in first pass)

Exact sampled values for each trial are recorded in:
- `artifacts/hpo/*_hpo/hpo_search_trials.csv` (`params_json`)

Script used for this first pass:
- `src/models_v2/run_hpo_experiments.py` (baseline/frozen)

## Gate Status (Current Official Floors)

Current policy floors:
- `accuracy >= 0.85`
- `closed_precision >= 0.30`

Outcome:
- Gate pass count: `0 / 7`
- No model-mode config passed both floors in this HPO pass.
- Selection in each per-model run therefore used fallback policy (`max closed_f1` with shortfall documentation).

## Interpretation

- Best closed-class thresholded performance remains LR (`closed_f1 ~ 0.239`).
- RF is currently the strongest non-LR compromise (`accuracy ~ 0.857`, `closed_precision ~ 0.236`, `closed_f1 ~ 0.232`).
- XGBoost and LightGBM maintain higher accuracy but weak closed recall/F1 at threshold `0.5`.
- The main blocker for gate passing is closed precision floor (`0.30`), not just model family choice.

## Recommended Next Steps (Agreed Plan)

1. Run a second HPO pass with class-imbalance weighting as an explicit tunable knob.
2. Compare run 1 vs run 2 and identify promising parameter regions.
3. Run one final narrowed HPO pass (focused ranges) for this bundle.
4. Freeze shortlisted configs for this bundle.
5. Run decision-threshold sweeps on shortlisted configs.
6. Then proceed to bundle-iteration experiments (repeat HPO lightly for new bundles).

Notes:
- Threshold sweeps happen after the focused HPO pass, not before.
- Official gate status should still be reported using the protocol floors.

## Position in Workflow

This HPO pass happened **before** decision-threshold sweeps, consistent with protocol:
1. model/feature/gate framework fixed
2. per-model HPO
3. threshold sweeps on shortlisted configs

## Current Status Snapshot

- Cross-model HPO pass completed for:
  - `lr` (`single`, `two-stage`)
  - `lightgbm` (`single`, `two-stage`)
  - `rf` (`single`)
  - `xgboost` (`single`, `two-stage`)
- Gate pass count remains `0/7` under current official floors.
- Best confirmed closed-F1 remains LR (`~0.239`), with RF close (`~0.232`).

---

# HPO Results Summary (Weighted Pass 1)

Date run: 2026-03-06  
Runner: `src/models_v2/run_hpo_experiments_weighted.py`  
Feature bundle: `low_plus_medium`  
Search budget: `40 trials/model`  
Search CV: `5x1`  
Confirm CV: `5x3`  
Decision threshold: `0.5`

## Artifact Sources

- `artifacts/hpo_weighted_pass1/hpo_search_trials.csv`
- `artifacts/hpo_weighted_pass1/hpo_selected_trials.csv`
- `artifacts/hpo_weighted_pass1/hpo_confirm_metrics.csv`
- `artifacts/hpo_weighted_pass1/hpo_run_config.json`

## What Changed vs Baseline Pass

Weighted pass 1 introduces explicit class-imbalance tuning knobs:

- LR / RF:
  - `class_weight ∈ {None, "balanced", custom {0: w_closed, 1: 1.0}}`
- LightGBM:
  - `class_weight ∈ {None, "balanced"}`
  - optional `scale_pos_weight`
- XGBoost:
  - `scale_pos_weight`

## Confirmed Results (Sorted by Closed F1)

| Model | Mode | Accuracy Mean | Closed Precision Mean | Closed Recall Mean | Closed F1 Mean | PR-AUC Closed Mean |
|---|---|---:|---:|---:|---:|---:|
| RF | single | 0.865 | 0.247 | 0.219 | 0.230 | 0.179 |
| LR | two-stage | 0.862 | 0.241 | 0.225 | 0.229 | 0.185 |
| LightGBM | single | 0.859 | 0.115 | 0.097 | 0.098 | 0.145 |
| LightGBM | two-stage | 0.834 | 0.086 | 0.115 | 0.084 | 0.154 |
| LR | single | 0.900 | 0.347 | 0.033 | 0.057 | 0.160 |
| XGBoost | single | 0.883 | 0.106 | 0.038 | 0.055 | 0.134 |
| XGBoost | two-stage | 0.892 | 0.098 | 0.022 | 0.036 | 0.098 |

## Gate Status (Current Official Floors)

Current policy floors:
- `accuracy >= 0.85`
- `closed_precision >= 0.30`

Weighted pass 1 confirm outcome:
- Gate pass count: `1 / 7` (LR single)

Important caveat:
- The gate-pass config (`LR single`) achieves this mostly via high open-heavy behavior:
  - closed recall is very low (`~0.033`)
  - closed F1 is low (`~0.057`)
- So this is a technical gate pass under current floors, but not yet a strong closed-detection operating point.

## Interpretation

- Weighted tuning improved the frontier and produced two strong closed-oriented candidates:
  - `RF single` and `LR two-stage` both around `closed_f1 ~ 0.23` with `closed_precision ~ 0.24`.
- `LR single` can pass current floors, but with poor closed recall/F1 (not a preferred closed-performance candidate).
- Two-stage remains viable:
  - `LR two-stage` is competitive with RF on closed F1 and PR-AUC(closed), suggesting stage-1 filtering is directionally useful.
- LightGBM/XGBoost remained weaker on thresholded closed metrics in this pass.

## Recommended Next Steps

1. Run one narrowed weighted HPO pass (focused around best RF single and LR two-stage regions).
2. Keep official policy floors unchanged for reporting consistency.
3. After narrowed HPO, run threshold sweeps on shortlisted configs.
4. Re-evaluate gate status and closed-performance tradeoffs.
5. TODO: Split gate definitions into `production` vs `diagnostic` in protocol + runner outputs so deployability decisions are separated from research ranking.
