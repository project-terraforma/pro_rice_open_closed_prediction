# Artifacts Index

This directory contains generated outputs from different stages of the experiment workflow.

## Current High-Value Directories

- [`hpo_optuna_narrow_pass1`](./hpo_optuna_narrow_pass1)
  - narrowed frontier HPO outputs
- [`hpo_optuna_lr_micro_pass1`](./hpo_optuna_lr_micro_pass1)
  - LR micro-HPO outputs
- [`hpo_optuna_rf_micro_pass1`](./hpo_optuna_rf_micro_pass1)
  - RF micro-HPO outputs
- [`k_threshold_sweep_lr_pass1`](./k_threshold_sweep_lr_pass1)
  - frozen LR baseline `k`/threshold/confirm outputs
- [`k_threshold_sweep_rf_pass1`](./k_threshold_sweep_rf_pass1)
  - frozen RF baseline `k`/threshold/confirm outputs
- [`k_threshold_sweep_rf_no_spatial_prior_pass1`](./k_threshold_sweep_rf_no_spatial_prior_pass1)
  - confirmed RF v2 winner outputs
- [`k_threshold_sweep_lr_v2_pass1`](./k_threshold_sweep_lr_v2_pass1)
  - LR v2 fairness-check outputs
- [`feature_importance`](./feature_importance)
  - feature-importance exports
- [`ablation_v2_split_pass1`](./ablation_v2_split_pass1)
  - v2 feature-group ablation results

## Stage Mapping

### Early / Historical

- [`cv`](./cv)
  - early repeated-CV benchmark outputs
- [`hpo`](./hpo)
  - early baseline HPO outputs
- [`hpo_weighted_pass1`](./hpo_weighted_pass1)
  - weighted HPO pass
- [`hpo_weighted_smoke`](./hpo_weighted_smoke)
  - smoke-test outputs

### Active Ceiling-Study Flow

1. frontier HPO:
   - `hpo_optuna_narrow_pass1`
   - `hpo_optuna_lr_micro_pass1`
   - `hpo_optuna_rf_micro_pass1`
2. operating-point tuning:
   - `k_threshold_sweep_*`
3. bundle diagnostics:
   - `ablation_v2_pass1`
   - `ablation_v2_split_pass1`
4. feature understanding:
   - `feature_importance`

## Common File Names

Inside many artifact directories, the same filenames repeat. Use them like this:

- `*_search_trials.csv`
  - all trial-level search results
- `*_selected_trials*.csv`
  - chosen configs from a search stage
- `*_confirm_metrics*.csv`
  - stronger-CV confirm results
- `k_coarse_metrics.csv`
  - coarse `k` sweep
- `k_narrow_metrics.csv`
  - narrow `k` sweep
- `threshold_sweep_metrics.csv`
  - threshold grid results
- `threshold_final_best.csv`
  - selected threshold-stage finalists
- `group_ablation_summary.csv`
  - ablation summary by variant
- `*_run_config.json`
  - exact run settings

## Recommended Places To Look First

If you want the current best model:
- [`k_threshold_sweep_rf_no_spatial_prior_pass1/threshold_confirm_metrics.csv`](./k_threshold_sweep_rf_no_spatial_prior_pass1/threshold_confirm_metrics.csv)

If you want the frozen LR baseline:
- [`k_threshold_sweep_lr_pass1/threshold_confirm_metrics.csv`](./k_threshold_sweep_lr_pass1/threshold_confirm_metrics.csv)

If you want the LR v2 fairness check:
- [`k_threshold_sweep_lr_v2_pass1/threshold_confirm_metrics.csv`](./k_threshold_sweep_lr_v2_pass1/threshold_confirm_metrics.csv)

If you want feature-importance outputs:
- [`feature_importance`](./feature_importance)
