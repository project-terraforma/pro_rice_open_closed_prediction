# Docs Index

## Start Here

If you want the current project state, read these in order:

1. [`../README.md`](../README.md)
   - high-level project summary and current winners
2. [`eval_protocol.md`](eval_protocol.md)
   - source of truth for gates, ranking, and what counts as a "good" result
3. [`hpo_results_summary.md`](hpo_results_summary.md)
  - current experiment narrative and frozen reference configs
4. [`feature_importance_results.md`](feature_importance_results.md)
  - feature-importance takeaways
5. [`alex_transfer_results.md`](alex_transfer_results.md)
  - direct-transfer and retrain-on-Alex results
6. [`top_budget_triage_results.md`](top_budget_triage_results.md)
  - outcome of the fixed-budget closure-review ranking check

## Current Source Of Truth

- [`eval_protocol.md`](eval_protocol.md)
  - current evaluation contract
- [`hpo_results_summary.md`](hpo_results_summary.md)
  - current frontier, frozen configs, v2 bundle outcome
- [`alex_transfer_results.md`](alex_transfer_results.md)
  - current Alex-dataset transfer/retrain findings
- [`top_budget_triage_results.md`](top_budget_triage_results.md)
  - current top-budget review-triage conclusion
- [`feature_importance_results.md`](feature_importance_results.md)
  - feature importance results and interpretation
- [`feature_bundles.json`](feature_bundles.json)
  - actual bundle definitions used by the code
- [`feature_bundles.yaml`](feature_bundles.yaml)
  - human-readable bundle mirror
- [`feature_inventory.csv`](feature_inventory.csv)
  - per-feature inventory with `cost_tier`
- [`feature_rationale.md`](feature_rationale.md)
  - why each feature exists and how cost tiers are defined

## Workflow / Design Docs

- [`hpo_runner_design.md`](hpo_runner_design.md)
  - how the HPO and sweep runners are organized
- [`hpo_optuna_narrow_design.md`](hpo_optuna_narrow_design.md)
  - rationale for the narrow frontier pass
- [`feature_bundle_v2_conventions.md`](feature_bundle_v2_conventions.md)
  - rules for creating and evaluating v2 bundles
- [`feature_bundle_v2_rationale.md`](feature_bundle_v2_rationale.md)
  - why the v2 bundles were chosen
- [`open_triage_policy_plan.md`](open_triage_policy_plan.md)
  - proposed two-bucket open-safe / review-queue follow-up
- [`top_budget_triage_plan.md`](top_budget_triage_plan.md)
  - proposed fixed-budget top-risk review follow-up

## Historical / Exploratory Docs

These are useful for history, but not the first place to look for the current state:

- [`cv_results_summary.md`](cv_results_summary.md)
  - early repeated-CV snapshot before the mature sweep workflow
- [`model_results.md`](model_results.md)
  - legacy summary
- [`logistic_regression.md`](logistic_regression.md)
  - model-family notes
- [`random_forest.md`](random_forest.md)
  - model-family notes
- [`lightgbm.md`](lightgbm.md)
  - model-family notes
- [`rules_only_baseline.md`](rules_only_baseline.md)
  - baseline notes

## Problem / Data Context

- [`problem_scope.md`](problem_scope.md)
- [`data_dictionary.md`](data_dictionary.md)

## Batch Simulation

Batch-simulation planning and implementation docs live under:
- [`batch_simulation/README.md`](batch_simulation/README.md)

Use these only if you are working on the labeling / simulation / deployment side of the repo.
