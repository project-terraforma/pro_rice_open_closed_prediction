# Open/Closed Place Prediction (Schema-Native Ceiling Study)

## Team
- Clarice Park
- Matthew Kimotsuki

## Project Goal
Estimate the practical performance ceiling for predicting `open` vs `closed` places using only scalable schema-native Overture features, then use that ceiling to judge whether further gains are more likely to come from model tuning or from better data/label coverage.

## Current Outcome

This repo now contains a reasonably mature ceiling study for the current dataset under low/medium-cost feature engineering.

Current confirmed diagnostic leader:
- model: `RandomForest`
- mode: `single`
- feature bundle: `v2_rf_single_no_spatial_prior`
- hyperparameters:
  - `n_estimators=450`
  - `max_depth=16`
  - `min_samples_leaf=6`
  - `min_samples_split=12`
  - `max_features=log2`
  - `class_weight=balanced`
- featurizer settings: `k=(25,5,45)`
- threshold: `0.39`
- confirm metrics:
  - `accuracy=0.886`
  - `closed_precision=0.343`
  - `closed_recall=0.263`
  - `closed_f1=0.297`
  - `pr_auc_closed=0.262`

Current confirmed LR reference leader:
- model: `LogisticRegression`
- mode: `two-stage`
- feature bundle: `low_plus_medium`
- confirm metrics:
  - `accuracy=0.860`
  - `closed_precision=0.251`
  - `closed_recall=0.260`
  - `closed_f1=0.254`
  - `pr_auc_closed=0.205`

Practical conclusion:
- within the current data regime and low/medium-cost feature policy, RF with the frozen v2 bundle is the top confirmed diagnostic configuration
- strict production gates are still not met, so this remains a ceiling/iteration study rather than a production-ready model

## What Is In Scope

- schema-native Overture features only
- scalable low/medium-cost feature engineering
- fair cross-model comparison and per-model ceiling evaluation
- HPO, featurizer `k` tuning, threshold tuning, confirm-CV, and bundle ablation

Out of scope for this phase:
- external APIs or joins
- live website probing
- expensive per-record inference pipelines
- non-schema-native high-cost enrichment

## Cost Tier Definitions

- `low cost`
  - schema-native, record-local, cheap deterministic transforms
  - examples: counts, booleans, simple bins, regex flags, direct field-presence indicators
- `medium cost`
  - still schema-native and scalable, but requires added fitting or preprocessing
  - examples: fold-safe target-encoding priors, top-K one-hot vocabularies, spatial bucketing, interaction features
- `high cost`
  - requires external services/data, runtime probing, expensive enrichment, or heavy inference
  - out of scope for this study

Per-feature cost tiers live in:
- [`docs/feature_inventory.csv`](docs/feature_inventory.csv)
- [`docs/feature_rationale.md`](docs/feature_rationale.md)

## Evaluation Policy

The current source of truth is:
- [`docs/eval_protocol.md`](docs/eval_protocol.md)

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

How “goodness” is determined:
- gates first
- gate-specific ranking second
- confirm CV after selection
- only confirmed gains count as meaningful frontier movement

## Main Docs

- [`docs/eval_protocol.md`](docs/eval_protocol.md)
- [`docs/hpo_runner_design.md`](docs/hpo_runner_design.md)
- [`docs/feature_importance_results.md`](docs/feature_importance_results.md)
- [`docs/hpo_results_summary.md`](docs/hpo_results_summary.md)
- [`docs/feature_bundle_v2_conventions.md`](docs/feature_bundle_v2_conventions.md)
- [`docs/feature_bundle_v2_rationale.md`](docs/feature_bundle_v2_rationale.md)

Historical-only docs:
- [`docs/cv_results_summary.md`](docs/cv_results_summary.md)

## Feature Bundles

Feature bundles are defined in:
- [`docs/feature_bundles.json`](docs/feature_bundles.json)
- [`docs/feature_bundles.yaml`](docs/feature_bundles.yaml)

Key bundles:
- `low_only`
  - low-cost schema-native features only
- `low_plus_medium`
  - canonical fair-comparison bundle
- `v2_lr2`
  - LR v2 comparison bundle
- `v2_rf_single_no_spatial_prior`
  - frozen RF v2 winner for this round

## Repository Layout

- `src/models_v2/`
  - current modeling, HPO, sweep, ablation, and evaluation code
- `docs/`
  - protocol, rationale, and results summaries
- `artifacts/`
  - generated HPO, sweep, ablation, and importance outputs
- `data/`
  - train/val/test parquet splits and supporting files

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main Workflow

The project’s mature workflow is:
1. freeze gate logic and split policy
2. run HPO
3. freeze shortlisted configs
4. tune featurizer `k`
5. tune thresholds
6. run confirm CV
7. optionally run bundle-v2 ablation/refinement
8. re-run the same phased workflow on serious v2 candidates

## Useful Commands

### 1) Feature Importance

```bash
python src/models_v2/export_feature_importance.py \
  --models lr lightgbm rf xgboost \
  --modes single two-stage \
  --feature-bundle low_plus_medium \
  --use-train-plus-val
```

### 2) Narrow Frontier HPO

```bash
python src/models_v2/run_hpo_optuna_narrow.py \
  --models lr rf \
  --feature-bundle low_plus_medium
```

### 3) LR Micro HPO

```bash
python src/models_v2/run_hpo_optuna_lr_micro.py \
  --feature-bundle low_plus_medium
```

### 4) RF Micro HPO

```bash
python src/models_v2/run_hpo_optuna_rf_micro.py \
  --feature-bundle low_plus_medium
```

### 5) Phased K + Threshold Sweep

```bash
python src/models_v2/run_k_threshold_sweep.py \
  --phase full \
  --models rf \
  --modes single \
  --gate-types diagnostic \
  --selected-configs-csv artifacts/hpo_optuna_narrow_pass1/hpo_selected_trials_dualgate.csv \
  --feature-bundle-overrides '{"rf:single":"v2_rf_single_no_spatial_prior"}' \
  --output-dir artifacts/k_threshold_sweep_rf_no_spatial_prior_pass1
```

### 6) Feature Group Ablation Screen

```bash
python src/models_v2/run_feature_group_ablation_screen.py \
  --selected-configs-csv artifacts/hpo_optuna_narrow_pass1/hpo_selected_trials_dualgate.csv \
  --models lr rf \
  --modes two-stage single \
  --gate-types diagnostic \
  --variant-scope all \
  --n-splits 5 \
  --n-repeats 2 \
  --decision-threshold 0.5 \
  --category-top-k 25 \
  --dataset-top-k 20 \
  --cluster-top-k 30 \
  --output-dir artifacts/ablation_v2_split_pass1
```

## Notes

- The current documented conclusion is a practical ceiling for this dataset under low/medium-cost feature engineering, not a proof of the absolute theoretical maximum.
- Frozen configs are intended to be reused as starting points for future larger-sample experiments; full HPO does not need to be rerun immediately after every data expansion.
- If future work focuses on label coverage or larger training samples, start from the frozen configs first and only retune if the new data materially changes the frontier.
