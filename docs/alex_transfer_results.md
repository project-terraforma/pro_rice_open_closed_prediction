# Alex Dataset Transfer Results

## Purpose

Record the first two Alex-dataset checks run after the main ceiling study:

1. frozen-config direct transfer
2. frozen-config retrain on Alex's labeled data

This document is the source of truth for the current Alex-dataset interpretation before any Alex-side threshold tuning, `k` tuning, or HPO.

## Evaluated Frozen Reference

Main candidate used:
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

Reference in-domain result for context:
- holdout test on current project split:
  - `accuracy=0.901`
  - `closed_precision=0.435`
  - `closed_recall=0.323`
  - `closed_f1=0.370`
  - `pr_auc_closed=0.280`

## Alex Dataset Scope

Alex evaluation assets:
- `places-status-engine/assets/sf_places_raw.parquet`
- `places-status-engine/assets/sf_places_labeled_checkpoint.parquet`
- `places-status-engine/assets/nyc_places_raw.parquet`
- `places-status-engine/assets/nyc_places_labeled_checkpoint.parquet`

Rows used for metrics:
- labeled rows only (`fsq_label in {open, closed}`)
- `suspected_closed` and unlabeled rows excluded

Labeled evaluation counts:
- `SF`: `36,848`
- `NYC`: `42,886`
- total: `79,734`

Important clarification:
- Alex's `projectc` labeled assets are not an independent transfer dataset for this repo's `project_c_samples`; labels matched exactly in the check that was run.
- The meaningful Alex transfer test is therefore `SF + NYC`, not `projectc`.

## Run 1: Frozen Direct Transfer

Runner:
- `src/models_v2/run_alex_transfer_eval.py`

Artifacts:
- `artifacts/alex_transfer_rf_final_pass1/alex_transfer_metrics.csv`
- `artifacts/alex_transfer_rf_final_pass1/alex_transfer_predictions.csv`
- `artifacts/alex_transfer_rf_final_pass1/alex_transfer_run_config.json`

Setup:
- train frozen RF config on this repo's `train_split + val_split`
- apply that trained model directly to Alex's labeled `SF + NYC`
- keep bundle, hyperparameters, `k`, and threshold fixed

### Results

Overall:
- `accuracy=0.843`
- `closed_precision=0.0086`
- `closed_recall=0.00025`
- `closed_f1=0.00048`
- `pr_auc_closed=0.155`

Per-city:
- `SF`
  - `accuracy=0.913`
  - `closed_precision=0.000`
  - `closed_recall=0.000`
  - `closed_f1=0.000`
  - `pr_auc_closed=0.081`
- `NYC`
  - `accuracy=0.782`
  - `closed_precision=0.150`
  - `closed_recall=0.00032`
  - `closed_f1=0.00064`
  - `pr_auc_closed=0.234`

Observed behavior:
- the frozen transfer model predicted almost everything as `open`
- only `3` true closed predictions were made across all `79,734` labeled rows

### Interpretation

This direct transfer failed as an operating point.

What it means:
- the learned signal does not cleanly carry over at the frozen threshold / calibration
- there is strong dataset shift and/or severe calibration mismatch between the current repo's sample dataset and Alex's `SF + NYC` labeling regime

What it does **not** mean:
- it does not prove the RF feature set or model family is unusable on Alex's data
- it does not prove the underlying open/closed problem is impossible on Alex's dataset

## Run 2: Frozen-Structure Retrain On Alex Data

Runner:
- `src/models_v2/run_alex_retrain_eval.py`

Artifacts:
- `artifacts/alex_retrain_rf_final_pass1/alex_retrain_metrics.csv`
- `artifacts/alex_retrain_rf_final_pass1/alex_retrain_oof_predictions.csv`
- `artifacts/alex_retrain_rf_final_pass1/alex_retrain_run_config.json`

Setup:
- use the same frozen RF bundle, hyperparameters, `k`, and threshold
- retrain on Alex's labeled data
- evaluate with `5-fold x 1-repeat` out-of-fold CV on Alex's labeled `SF + NYC`

### Results

Overall:
- `accuracy=0.821`
- `closed_precision=0.451`
- `closed_recall=0.792`
- `closed_f1=0.575`
- `pr_auc_closed=0.584`

Per-city:
- `SF`
  - `accuracy=0.918`
  - `closed_precision=0.485`
  - `closed_recall=0.759`
  - `closed_f1=0.592`
  - `pr_auc_closed=0.571`
- `NYC`
  - `accuracy=0.737`
  - `closed_precision=0.442`
  - `closed_recall=0.802`
  - `closed_f1=0.570`
  - `pr_auc_closed=0.587`

### Delta vs Direct Transfer

Overall deltas:
- `accuracy`: `-0.022`
- `closed_precision`: `+0.443`
- `closed_recall`: `+0.792`
- `closed_f1`: `+0.574`
- `pr_auc_closed`: `+0.429`

Interpretation:
- retraining on Alex's labels radically changes the result
- the same RF configuration structure becomes strong diagnostically once fit on Alex's own label distribution
- the bottleneck is therefore much more consistent with transfer/calibration shift than with a total model failure

## Gate Status On Alex Retrain Run

Current production gate:
- `accuracy >= 0.90`
- `closed_precision >= 0.70`
- `closed_recall >= 0.05`

Current diagnostic gate:
- `accuracy >= 0.84`
- `closed_precision >= 0.20`

### Alex Retrain Gate Readout

Overall Alex retrain result:
- production: **fail**
  - fails on `accuracy` and `closed_precision`
- diagnostic: **fail overall**
  - `closed_precision` passes
  - `accuracy=0.821 < 0.84`

Per-city:
- `SF`: diagnostic-pass, production-fail
- `NYC`: diagnostic-fail, production-fail

Important note:
- despite strong closed-class retrieval, this result is still well short of the production bar
- moving from `closed_precision ~0.45` to `0.70` while simultaneously keeping `accuracy >= 0.90` would require a large additional gain

## Current Conclusions

1. The frozen RF winner does **not** transfer well as-is to Alex's labeled `SF + NYC` data.
2. The same RF configuration structure performs much better once retrained on Alex's labels.
3. The strongest current explanation is dataset shift / calibration mismatch, not total failure of the model family.
4. Alex's data appears to support materially stronger closed-detection performance than the direct transfer run suggested.
5. Even so, the retrained result still does not satisfy the project's production gate.

## Recommended Next Steps

Recommended next sequence:

1. Compare against Alex's own pipeline on the same `SF + NYC` labeled rows.
   - Goal: determine whether the gain is mostly from larger/better labels or whether Alex's feature stack still has a major edge.

2. Tune threshold on Alex's retrained RF before running HPO.
   - Goal: check whether precision/accuracy can improve meaningfully with operating-point adjustment alone.

3. If threshold-only gains are insufficient, tune featurizer `k` on Alex's data.
   - Goal: test whether the current bundle generalizes but the frozen vocab/cluster settings do not.

4. Only then consider Alex-side HPO.
   - Goal: determine whether model-side optimization can materially narrow the remaining production-gap.

## Bottom Line

The Alex-dataset work changed the project conclusion in one specific way:

- the current frozen RF operating point does **not** transfer directly
- but the same RF structure retrained on Alex's labels is genuinely useful diagnostically

So the project is no longer at "model-side work is exhausted everywhere."
It is now at:

- transfer is weak
- retrain is promising
- production readiness is still not met
- the most defensible next question is whether Alex-side thresholding / tuning or Alex's own pipeline can push precision materially higher
