# Alex Dataset Handoff

## Purpose

Capture the key context from the current ceiling-study phase so the next session can move directly into holdout interpretation and Alex-dataset transfer work without reconstructing the full project history.

## Final Status From This Session

The current ceiling study is effectively complete for the provided sample dataset under low/medium-cost feature engineering.

What is established:
- strong model-side tuning was completed
- LR and RF were both pushed through HPO, `k` tuning, threshold tuning, confirm CV, and bundle refinement/fairness checks
- RF v2 is the final winner on both confirm CV and the untouched holdout test split
- strict production gates are still not met

Interpretation:
- this is a practical ceiling for the current dataset/sample and current feature-cost regime
- it is not a universal ceiling for the real problem
- remaining shortfall is now more plausibly a data/label limitation than a simple lack of model tuning

## Final Evaluation Policy

Source of truth:
- `docs/ceiling_study/eval_protocol.md`

Current dual-gate policy:

- `production`
  - `accuracy >= 0.90`
  - `closed_precision >= 0.70`
  - `closed_recall >= 0.05`
  - rank by `closed_precision`, then `closed_f1`, then `pr_auc_closed`, then `accuracy`

- `diagnostic`
  - `accuracy >= 0.84`
  - `closed_precision >= 0.20`
  - rank by `closed_f1`, then `pr_auc_closed`, then `closed_precision`, then `accuracy`

Important:
- current model comparisons and recommendations should use this dual-gate policy, not the older legacy floors

## Final Frozen Winner

Final overall winner:
- model: `RandomForest`
- mode: `single`
- feature bundle: `v2_rf_single_no_spatial_prior`
- selected trial: `35`
- hyperparameters:
  - `n_estimators=450`
  - `max_depth=16`
  - `min_samples_leaf=6`
  - `min_samples_split=12`
  - `max_features=log2`
  - `class_weight=balanced`
- featurizer settings:
  - `category_top_k=25`
  - `dataset_top_k=5`
  - `cluster_top_k=45`
- threshold:
  - `0.39`

Confirm-CV metrics:
- `accuracy=0.886`
- `closed_precision=0.343`
- `closed_recall=0.263`
- `closed_f1=0.297`
- `pr_auc_closed=0.262`

Holdout test metrics:
- `accuracy=0.901`
- `closed_precision=0.435`
- `closed_recall=0.323`
- `closed_f1=0.370`
- `pr_auc_closed=0.280`

Production status:
- still not production-pass because `closed_precision=0.435 < 0.70`

## Final LR Reference

Best LR reference for comparison:
- model: `LogisticRegression`
- mode: `two-stage`
- feature bundle: `low_plus_medium`
- selected trial: `19`
- hyperparameters:
  - `C=0.018049168526244785`
  - `class_weight={0:5.0,1:1.0}`
  - `max_iter=1000`
  - `solver=lbfgs`
- featurizer settings:
  - `category_top_k=50`
  - `dataset_top_k=5`
  - `cluster_top_k=45`
- threshold:
  - `0.51`

Confirm-CV metrics:
- `accuracy=0.860`
- `closed_precision=0.251`
- `closed_recall=0.260`
- `closed_f1=0.254`
- `pr_auc_closed=0.205`

Holdout test metrics:
- `accuracy=0.863`
- `closed_precision=0.278`
- `closed_recall=0.323`
- `closed_f1=0.299`
- `pr_auc_closed=0.201`

## V2 Bundle Outcome

Bundle conclusions:
- `v2_rf_single_no_spatial_prior` is frozen as the RF v2 winner
- `v2_lr2` was fully checked for fairness and did not beat the frozen LR baseline

Important fairness result:
- LR was not simply left behind because bundle iteration stopped too early
- after full phased sweep + confirm, LR v2 still remained behind RF v2

## Holdout Evaluation

New runner added in this session:
- `src/models_v2/run_holdout_test_eval.py`

Purpose:
- read frozen configs from an artifact CSV
- train on `train_split + val_split`
- evaluate once on untouched `test_split`
- write both metrics and per-row predictions

Current holdout artifact directories:
- `artifacts/holdout_test_rf_final_pass1/`
- `artifacts/holdout_test_lr_final_pass1/`

Files produced:
- `holdout_test_metrics.csv`
- `holdout_test_predictions.csv`
- `holdout_test_run_config.json`

The predictions files can be used later for:
- confusion matrix
- false positives
- false negatives
- category/source error slicing

## Feature Bundle / Cost Context

Canonical pre-v2 bundle:
- `low_plus_medium`
- `55` total features
- `40` low-cost
- `15` medium-cost

Final v2 bundles:
- `v2_lr2`
  - `34` total
  - `24` low-cost
  - `10` medium-cost
- `v2_rf_single_no_spatial_prior`
  - `33` total
  - `23` low-cost
  - `10` medium-cost

Cost-tier definitions are now documented in:
- `README.md`
- `docs/ceiling_study/feature_rationale.md`

## Important Docs Updated This Session

Main current docs:
- `README.md`
- `docs/README.md`
- `docs/ceiling_study/eval_protocol.md`
- `docs/ceiling_study/hpo_results_summary.md`
- `docs/ceiling_study/feature_bundle_v2_conventions.md`
- `docs/ceiling_study/feature_bundle_v2_rationale.md`
- `artifacts/README.md`
- `src/models_v2/README.md`

Historical docs were left in place but marked more clearly where needed.

## Recommended Next Step: Alex Dataset Work

Given limited time, the recommended order is:

1. Use the frozen final RF winner directly on Alex’s dataset
   - this is the independent transfer test
   - goal: measure how the learned signal transfers without re-optimization

2. Use the same frozen RF configuration structure retrained on Alex’s dataset
   - same bundle, same hyperparameters, same `k`, same threshold as a starting point
   - goal: see whether Alex’s dataset itself supports stronger performance without paying for full HPO first

3. Optionally run the frozen LR reference too
   - useful as a sanity baseline / transfer comparison

Recommended interpretation:
- if direct transfer is weak but retraining on Alex’s data is much better:
  - likely dataset shift
- if both are weak:
  - likely a harder underlying problem or persistent label-quality limits
- if both are decent:
  - the current approach generalizes better than expected

## Recommended Scope For Alex Work

Do first:
- frozen-config transfer
- frozen-config retrain on Alex’s dataset

Do not do first:
- full HPO on Alex’s dataset
- new bundle-optimization round on Alex’s dataset

Reason:
- frozen configs are now strong enough to serve as a baseline
- this is the fastest way to learn whether more/better data changes the ceiling

## Suggested First Commands To Reconstruct Context

In a new session, likely first files to open:

1. `README.md`
2. `docs/ceiling_study/eval_protocol.md`
3. `docs/ceiling_study/hpo_results_summary.md`
4. `docs/adjacent/handoff/alex_dataset_handoff.md`
5. `src/models_v2/run_holdout_test_eval.py`

Then inspect the final RF artifacts:
- `artifacts/k_threshold_sweep_rf_no_spatial_prior_pass1/threshold_confirm_metrics.csv`
- `artifacts/holdout_test_rf_final_pass1/holdout_test_metrics.csv`

## Bottom Line

Use the frozen RF v2 winner as the main transfer candidate for Alex’s dataset.

That is:
- the best confirmed model
- the best holdout-tested model
- the cleanest baseline for asking whether larger/better data, rather than more tuning, is what moves the project forward

---

## Update After Running The Recommended Alex Checks

The recommended first two checks have now been run and documented in:
- `docs/ceiling_study/alex_transfer_results.md`

High-level outcome:
- direct frozen transfer to Alex's labeled `SF + NYC` data was very weak
- retraining the same frozen RF structure on Alex's labeled data was much stronger
- the current best interpretation is dataset shift / calibration mismatch, not total RF failure
- even after retraining, the project production gate is still not met

Practical implication:
- the next Alex-side step should be threshold/tuning comparison and/or comparison against Alex's own pipeline, not a claim of immediate production readiness
