# `src/models_v2` Index

This directory contains the active modeling workflow. The easiest way to navigate it is by stage rather than by filename prefix.

## Core Building Blocks

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

## Baseline / Comparison Runners

- [`run_cv_experiments.py`](./run_cv_experiments.py)
  - repeated-CV comparison runner
- [`export_feature_importance.py`](./export_feature_importance.py)
  - feature-importance export runner

## HPO Runners

- [`run_hpo_experiments.py`](./run_hpo_experiments.py)
  - early baseline HPO runner
- [`run_hpo_experiments_weighted.py`](./run_hpo_experiments_weighted.py)
  - weighted HPO variant
- [`run_hpo_experiments_weighted_dualgate.py`](./run_hpo_experiments_weighted_dualgate.py)
  - weighted HPO with current dual-gate logic
- [`run_hpo_optuna_narrow.py`](./run_hpo_optuna_narrow.py)
  - narrowed frontier HPO
- [`run_hpo_optuna_lr_micro.py`](./run_hpo_optuna_lr_micro.py)
  - LR-only micro refinement
- [`run_hpo_optuna_rf_micro.py`](./run_hpo_optuna_rf_micro.py)
  - RF-only micro refinement
- [`run_hpo_bootstrap_confirm.py`](./run_hpo_bootstrap_confirm.py)
  - bootstrap/confirm utility for selected configs

## Post-HPO Operating-Point Tuning

- [`run_k_threshold_sweep.py`](./run_k_threshold_sweep.py)
  - phased `k_coarse -> k_narrow -> threshold -> confirm` workflow

## Bundle / Feature Diagnostics

- [`run_feature_group_ablation_screen.py`](./run_feature_group_ablation_screen.py)
  - group and split-feature ablation screen
- [`run_lr_ablation.py`](./run_lr_ablation.py)
  - older LR-only ablation script

## Data / Labeling Utilities

- [`simulate_label_coverage.py`](./simulate_label_coverage.py)
  - label-coverage experiments
- [`build_sim_batches.py`](./build_sim_batches.py)
  - batch simulation helper

## Recommended Reading Order

If you are trying to understand the current modeling stack:

1. [`../../docs/eval_protocol.md`](../../docs/eval_protocol.md)
2. [`shared_featurizer.py`](./shared_featurizer.py)
3. one model file:
   - [`random_forest_model_v2.py`](./random_forest_model_v2.py)
   - [`logistic_regression_v2.py`](./logistic_regression_v2.py)
4. [`run_k_threshold_sweep.py`](./run_k_threshold_sweep.py)
5. [`run_feature_group_ablation_screen.py`](./run_feature_group_ablation_screen.py)

## Practical Naming Guide

When filenames look similar:

- `run_hpo_*`
  - hyperparameter search
- `run_k_threshold_*`
  - post-HPO operating-point tuning
- `run_*ablation*`
  - bundle/feature diagnostics
- `*_model_v2.py`
  - model implementation
- `shared_*`
  - reusable infrastructure
