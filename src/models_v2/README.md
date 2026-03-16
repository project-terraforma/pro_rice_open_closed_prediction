# `src/models_v2` Index

This directory contains the shared `v2` modeling foundation for the active workstreams in this repository.

It is the main code directory behind the ceiling-study workflow, and parts of it are also reused by the incremental-training / benchmarking workstream.

Use this README as a map of:

- `core shared infrastructure`
- `active ceiling-study runners`
- `adjacent utilities that are not part of the main ceiling-study path`

This directory is not the home for older v1 modeling code, and it is not a complete map of every active workstream in the repo.

## Start Here

If you are entering this directory for the first time:

1. read [`../../docs/ceiling_study/eval_protocol.md`](../../docs/ceiling_study/eval_protocol.md)
2. read [`shared_featurizer.py`](./shared_featurizer.py)
3. read one active model implementation:
   - [`random_forest_model_v2.py`](./random_forest_model_v2.py)
   - [`logistic_regression_v2.py`](./logistic_regression_v2.py)
4. read [`run_k_threshold_sweep.py`](./run_k_threshold_sweep.py)
5. read [`run_feature_group_ablation_screen.py`](./run_feature_group_ablation_screen.py)

## Core Shared Infrastructure

- [`shared_featurizer.py`](./shared_featurizer.py)
  - shared feature construction and bundle enforcement
- [`shared_metrics.py`](./shared_metrics.py)
  - metric computation, including closed-class metrics
- [`shared_evaluator.py`](./shared_evaluator.py)
  - shared evaluation helpers
- [`triage_policy.py`](./triage_policy.py)
  - stage-1 / triage policy logic
- [`label_store.py`](./label_store.py)
  - label access utilities

## Model Implementations

- [`logistic_regression_v2.py`](./logistic_regression_v2.py)
- [`random_forest_model_v2.py`](./random_forest_model_v2.py)
- [`lightgbm_model_v2.py`](./lightgbm_model_v2.py)
- [`xgboost_model_v2.py`](./xgboost_model_v2.py)

## Active Ceiling-Study Runners

These are the main scripts to look at if you are following the current ceiling-study workflow.

- [`export_feature_importance.py`](./export_feature_importance.py)
  - feature-importance export runner
- [`run_hpo_optuna_narrow.py`](./run_hpo_optuna_narrow.py)
  - narrowed frontier HPO
- [`run_hpo_optuna_lr_micro.py`](./run_hpo_optuna_lr_micro.py)
  - LR-only micro refinement
- [`run_hpo_optuna_rf_micro.py`](./run_hpo_optuna_rf_micro.py)
  - RF-only micro refinement
- [`run_k_threshold_sweep.py`](./run_k_threshold_sweep.py)
  - phased `k_coarse -> k_narrow -> threshold -> confirm` workflow
- [`run_feature_group_ablation_screen.py`](./run_feature_group_ablation_screen.py)
  - group and split-feature ablation screen
- [`run_holdout_test_eval.py`](./run_holdout_test_eval.py)
  - final holdout evaluation for frozen configs
- [`run_alex_transfer_eval.py`](./run_alex_transfer_eval.py)
  - transfer evaluation on the Alex dataset
- [`run_alex_retrain_eval.py`](./run_alex_retrain_eval.py)
  - retrain/evaluation on the Alex dataset
- [`run_top_budget_analysis.py`](./run_top_budget_analysis.py)
  - fixed-budget triage analysis
- [`run_two_bucket_analysis.py`](./run_two_bucket_analysis.py)
  - two-bucket triage analysis

## Comparison / Legacy Ceiling-Study Runners

- [`run_cv_experiments.py`](./run_cv_experiments.py)
  - repeated-CV comparison runner
- [`run_hpo_experiments.py`](./run_hpo_experiments.py)
  - early baseline HPO runner
- [`run_hpo_experiments_weighted.py`](./run_hpo_experiments_weighted.py)
  - weighted HPO variant
- [`run_hpo_experiments_weighted_dualgate.py`](./run_hpo_experiments_weighted_dualgate.py)
  - weighted HPO with current dual-gate logic
- [`run_hpo_bootstrap_confirm.py`](./run_hpo_bootstrap_confirm.py)
  - bootstrap/confirm utility for selected configs
- [`run_lr_ablation.py`](./run_lr_ablation.py)
  - older LR-only ablation script

## Adjacent Utilities And Reference Material

- [`simulate_label_coverage.py`](./simulate_label_coverage.py)
  - label-coverage experiments kept for reference, not part of the main active tracks
- [`build_sim_batches.py`](./build_sim_batches.py)
  - batch-simulation helper kept near related code, but not part of the core ceiling-study path
- [`run_projectc_oof_eval.py`](./run_projectc_oof_eval.py)
  - project-specific evaluation helper outside the main reading path

## Cross-References

- ceiling-study docs:
  - [`../../docs/README.md`](../../docs/README.md)
  - [`../../docs/ceiling_study/README.md`](../../docs/ceiling_study/README.md)
- repo-wide workstream map:
  - [`../../docs/WORKSTREAMS.md`](../../docs/WORKSTREAMS.md)

## Naming Guide

When filenames look similar:

- `run_hpo_*`
  - hyperparameter search
- `run_hpo_optuna_*`
  - later-stage narrowing or micro-refinement
- `run_k_threshold_*`
  - post-HPO operating-point tuning
- `run_*ablation*`
  - bundle/feature diagnostics
- `run_*eval*`
  - transfer, holdout, or project-specific evaluation
- `run_*analysis*`
  - downstream triage or budget analysis
- `*_model_v2.py`
  - model implementation
- `shared_*`
  - reusable infrastructure
