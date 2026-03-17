# Docs Index

This directory contains documentation for multiple workstreams in the repository.

Use this README as a map of:

- `repo-wide navigation docs`
- `active ceiling-study docs`
- `other active or adjacent docs`
- `historical / reference docs`

## Start Here

If you want the repo-wide view:

1. [`../README.md`](../README.md)
   - high-level repo summary and active tracks
2. [`WORKSTREAMS.md`](WORKSTREAMS.md)
   - which parts of the repo are active, separate, or historical

If you want the current ceiling-study path:

1. [`ceiling_study/README.md`](ceiling_study/README.md)
   - index for the current ceiling-study docs
2. [`ceiling_study/eval_protocol.md`](ceiling_study/eval_protocol.md)
   - source of truth for gates, ranking, and what counts as a good result
3. [`ceiling_study/hpo_results_summary.md`](ceiling_study/hpo_results_summary.md)
   - current experiment narrative and frozen reference configs
4. [`ceiling_study/feature_importance_results.md`](ceiling_study/feature_importance_results.md)
   - feature-importance takeaways
5. [`ceiling_study/alex_transfer_results.md`](ceiling_study/alex_transfer_results.md)
   - direct-transfer and retrain-on-Alex results
6. [`ceiling_study/top_budget_triage_results.md`](ceiling_study/top_budget_triage_results.md)
   - fixed-budget closure-review ranking results

## Repo-Wide Navigation Docs

- [`../README.md`](../README.md)
  - top-level repo landing page
- [`WORKSTREAMS.md`](WORKSTREAMS.md)
  - ownership and status map across workstreams

## Active Ceiling-Study Docs

These documents are the source of truth for the `ceiling study` track. The ceiling-study track is one active use of the shared `models_v2` foundation; other repo areas may also reuse that foundation for different current workstreams.

- [`ceiling_study/README.md`](ceiling_study/README.md)
  - index for the current ceiling-study docs
- [`ceiling_study/eval_protocol.md`](ceiling_study/eval_protocol.md)
  - current evaluation contract
- [`ceiling_study/hpo_results_summary.md`](ceiling_study/hpo_results_summary.md)
  - current frontier, frozen configs, v2 bundle outcome
- [`ceiling_study/alex_transfer_results.md`](ceiling_study/alex_transfer_results.md)
  - current Alex-dataset transfer/retrain findings
- [`ceiling_study/top_budget_triage_results.md`](ceiling_study/top_budget_triage_results.md)
  - current top-budget review-triage conclusion
- [`ceiling_study/feature_importance_results.md`](ceiling_study/feature_importance_results.md)
  - feature importance results and interpretation
- [`ceiling_study/feature_bundles.json`](ceiling_study/feature_bundles.json)
  - actual bundle definitions used by the code
- [`ceiling_study/feature_bundles.yaml`](ceiling_study/feature_bundles.yaml)
  - human-readable bundle mirror
- [`ceiling_study/feature_inventory.csv`](ceiling_study/feature_inventory.csv)
  - per-feature inventory with `cost_tier`
- [`ceiling_study/feature_rationale.md`](ceiling_study/feature_rationale.md)
  - why each feature exists and how cost tiers are defined

## Ceiling-Study Workflow / Design Docs

- [`ceiling_study/hpo_runner_design.md`](ceiling_study/hpo_runner_design.md)
  - how the HPO and sweep runners are organized
- [`ceiling_study/hpo_optuna_narrow_design.md`](ceiling_study/hpo_optuna_narrow_design.md)
  - rationale for the narrow frontier pass
- [`ceiling_study/feature_bundle_v2_conventions.md`](ceiling_study/feature_bundle_v2_conventions.md)
  - rules for creating and evaluating v2 bundles
- [`ceiling_study/feature_bundle_v2_rationale.md`](ceiling_study/feature_bundle_v2_rationale.md)
  - why the v2 bundles were chosen

## Other Active Or Adjacent Docs

- [`adjacent/labeling_simulation/README.md`](adjacent/labeling_simulation/README.md)
  - labeling / simulation / deployment-oriented docs outside the main ceiling-study path
- [`adjacent/handoff/alex_dataset_handoff.md`](adjacent/handoff/alex_dataset_handoff.md)
  - handoff / session-transfer context for Alex-dataset follow-up work
- [`adjacent/planning/open_triage_policy_plan.md`](adjacent/planning/open_triage_policy_plan.md)
  - planning doc for the open-safe / review-queue follow-up
- [`adjacent/planning/top_budget_triage_plan.md`](adjacent/planning/top_budget_triage_plan.md)
  - planning doc for the fixed-budget review-prioritization follow-up

## Incremental training / benchmarking docs:

- [`incremental_benchmarking/README.md`](incremental_benchmarking/README.md)
  - overview and run pointers
- [`incremental_benchmarking/incremental_results.md`](../src/incremental_benchmarking/incremental_results.md)
  - raw incremental run summary (per-batch metrics / single-run comparison)

## Historical / Reference Docs

These are useful for history, but not the first place to look for the current state:

- [`archive/cv_results_summary.md`](archive/cv_results_summary.md)
  - early repeated-CV snapshot before the mature sweep workflow
- [`archive/model_results.md`](archive/model_results.md)
  - legacy summary
- [`archive/logistic_regression.md`](archive/logistic_regression.md)
  - model-family notes
- [`archive/random_forest.md`](archive/random_forest.md)
  - model-family notes
- [`archive/lightgbm.md`](archive/lightgbm.md)
  - model-family notes
- [`archive/rules_only_baseline.md`](archive/rules_only_baseline.md)
  - baseline notes

## Problem / Data Context

- [`problem_scope.md`](problem_scope.md)
- [`data_dictionary.md`](data_dictionary.md)
