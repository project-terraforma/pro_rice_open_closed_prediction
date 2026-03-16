# Artifacts Index

This directory contains generated outputs from multiple workstreams in the repository.

Use this README as a map of:

- `active ceiling-study outputs`
- `other active or adjacent outputs`
- `historical / reference outputs`

## Start Here

If you are looking for the current ceiling-study outputs, start with:

- [`k_threshold_sweep_rf_no_spatial_prior_pass1`](./k_threshold_sweep_rf_no_spatial_prior_pass1)
  - confirmed RF v2 winner outputs
- [`holdout_test_rf_final_pass1`](./holdout_test_rf_final_pass1)
  - final held-out test results for the frozen RF winner
- [`feature_importance`](./feature_importance)
  - feature-importance exports
- [`ablation_v2_split_pass1`](./ablation_v2_split_pass1)
  - v2 feature-group ablation results

## Active Ceiling-Study Outputs

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
- [`holdout_test_rf_final_pass1`](./holdout_test_rf_final_pass1)
  - final held-out test results for the frozen RF winner
- [`holdout_test_lr_final_pass1`](./holdout_test_lr_final_pass1)
  - final held-out test results for the frozen LR reference
- [`feature_importance`](./feature_importance)
  - feature-importance exports
- [`ablation_v2_pass1`](./ablation_v2_pass1)
  - feature-group ablation outputs
- [`ablation_v2_split_pass1`](./ablation_v2_split_pass1)
  - split-feature ablation outputs
- [`alex_transfer_rf_final_pass1`](./alex_transfer_rf_final_pass1)
  - Alex transfer evaluation outputs
- [`alex_retrain_rf_final_pass1`](./alex_retrain_rf_final_pass1)
  - Alex retrain evaluation outputs
- [`projectc_top_budget_oof_rf_final_pass1`](./projectc_top_budget_oof_rf_final_pass1)
  - top-budget triage analysis outputs
- [`projectc_two_bucket_oof_rf_final_pass1`](./projectc_two_bucket_oof_rf_final_pass1)
  - two-bucket triage analysis outputs

## Other Active Or Adjacent Outputs

- [`projectc_oof_rf_final_pass1`](./projectc_oof_rf_final_pass1)
  - project-specific OOF evaluation outputs
- [`alex_filtered_datasets`](./alex_filtered_datasets)
  - exported filtered Alex datasets and manifest

Incremental training / benchmarking outputs:

- `TODO - add the current artifact directories for the incremental-training / benchmarking workstream`

## Historical / Reference Outputs

- [`archive/cv`](./archive/cv)
  - early repeated-CV benchmark outputs
- [`archive/hpo`](./archive/hpo)
  - early baseline HPO outputs
- [`archive/hpo_weighted_pass1`](./archive/hpo_weighted_pass1)
  - weighted HPO pass
- [`archive/hpo_weighted_smoke`](./archive/hpo_weighted_smoke)
  - smoke-test outputs
- [`archive/ablation`](./archive/ablation)
  - older ablation outputs
- [`archive/holdout_test_final_pass1`](./archive/holdout_test_final_pass1)
  - older holdout test outputs
- [`archive/projectc_oof_lr2_final_pass1`](./archive/projectc_oof_lr2_final_pass1)
  - archived project-specific LR2 OOF output directory
- [`archive/projectc_two_bucket_holdout_rf_final_pass1`](./archive/projectc_two_bucket_holdout_rf_final_pass1)
  - archived two-bucket holdout output directory
- [`archive/stage1_policy_compare_lr2_final_pass1`](./archive/stage1_policy_compare_lr2_final_pass1)
  - archived stage-1 policy comparison output directory

Reference / archive outputs that are not part of the main active tracks:

- [`archive/label_coverage`](./archive/label_coverage)
  - archived label-coverage outputs and explanations

## Ceiling-Study Stage Map

1. frontier HPO:
   - `hpo_optuna_narrow_pass1`
   - `hpo_optuna_lr_micro_pass1`
   - `hpo_optuna_rf_micro_pass1`
2. operating-point tuning:
   - `k_threshold_sweep_*`
3. final holdout evaluation:
   - `holdout_test_rf_final_pass1`
   - `holdout_test_lr_final_pass1`
4. bundle diagnostics:
   - `ablation_v2_pass1`
   - `ablation_v2_split_pass1`
5. feature understanding:
   - `feature_importance`
6. downstream evaluation / triage:
   - `holdout_test_*`
   - `alex_transfer_*`
   - `alex_retrain_*`
   - `projectc_top_budget_*`
   - `projectc_two_bucket_*`

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

If you want final held-out test results:
- [`holdout_test_rf_final_pass1/holdout_test_metrics.csv`](./holdout_test_rf_final_pass1/holdout_test_metrics.csv)
- [`holdout_test_lr_final_pass1/holdout_test_metrics.csv`](./holdout_test_lr_final_pass1/holdout_test_metrics.csv)

If you want feature-importance outputs:
- [`feature_importance`](./feature_importance)

If you want the repo-wide workstream map:
- [`../docs/WORKSTREAMS.md`](../docs/WORKSTREAMS.md)
