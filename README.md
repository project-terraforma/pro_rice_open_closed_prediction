# Open/Closed Place Prediction (Schema-Native Ceiling Study)

## Team
- Clarice Park
- Matthew Kimotsuki

## Quick Links
- Project Drive: https://drive.google.com/drive/folders/1Knb7acGgndajjU07SU2VA8BTJDqZOe1A

## Project Goal
Estimate the performance ceiling for predicting `open` vs `closed` places using only low-cost, schema-native Overture features, then identify the best performance-per-cost path to production.

## Current Status
- v2 shared framework is in place:
  - shared featurizer
  - shared metrics/evaluator
  - repeated stratified CV runner
  - model family runners: LR, LightGBM, RandomForest, XGBoost
- Evaluation protocol and decision rules are documented in:
  - [`docs/eval_protocol.md`](docs/eval_protocol.md)
- Current summarized outputs:
  - [`docs/cv_results_summary.md`](docs/cv_results_summary.md)
  - [`docs/feature_importance_results.md`](docs/feature_importance_results.md)

## Key Findings So Far
- Logistic Regression currently leads on thresholded closed-class F1 in repeated CV.
- Tree models currently achieve higher overall accuracy but lower closed recall/F1 at threshold `0.5`.
- Spatial and category risk features are consistently important across model families:
  - `spatial_cluster_closed_rate`
  - `category_closure_risk`
- Two-stage tree-model collapse was resolved by aligning stage-1 gate policy across models.

## Current CV Snapshot (`low_plus_medium`, threshold `0.5`)

| Model Family       | Mode      | Accuracy | Closed Precision | Closed Recall | Closed F1 | PR-AUC Closed |
|--------------------|-----------|---------:|-----------------:|--------------:|----------:|--------------:|
| LogisticRegression | two-stage |    0.682 |            0.146 |         0.505 |     0.226 |         0.130 |
| LogisticRegression | single    |    0.618 |            0.135 |         0.588 |     0.220 |         0.138 |
| LightGBM           | two-stage |    0.882 |            0.139 |         0.078 |     0.092 |         0.116 |
| LightGBM           | single    |    0.871 |            0.115 |         0.070 |     0.081 |         0.138 |
| RandomForest       | single    |    0.877 |            0.099 |         0.043 |     0.059 |         0.133 |
| XGBoost            | single    |    0.882 |            0.094 |         0.033 |     0.049 |         0.140 |
| XGBoost            | two-stage |    0.890 |            0.101 |         0.027 |     0.042 |         0.102 |

Source: `artifacts/cv/metrics_cv_summary.csv` and `docs/cv_results_summary.md`.

## Feature Bundle Policy

Feature bundles are defined in `docs/feature_bundles.json` and enforced by `SharedPlaceFeaturizer`.

- `low_only`
  - low-cost schema-native counts/booleans/recency/profile features only.
- `low_plus_medium`
  - `low_only` + medium-cost schema-native features such as one-hot vocab expansions and risk priors (for example `ohe_*`, `category_closure_risk`, `spatial_cluster_closed_rate`, `geo_cluster_id`).
- `full_schema_native`
  - all allowed schema-native features in current policy boundary (currently equivalent to `low_plus_medium` in this repo version).

Use `--feature-bundle` in runners to force a bundle:

```bash
--feature-bundle low_only
--feature-bundle low_plus_medium
--feature-bundle full_schema_native
```

For rationale and per-feature cost tiering:
- `docs/feature_inventory.csv`
- `docs/feature_rationale.md`

## Repository Layout
- `src/models_v2/`: current modeling/evaluation pipeline
- `docs/`: protocol, feature inventory/rationale, and current results summaries
- `artifacts/`: generated CV/importance outputs
- `data/`: train/val/test split parquet files and supporting datasets

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How To Run

### 1) Repeated CV Comparison (primary benchmark)

```bash
python src/models_v2/run_cv_experiments.py \
  --models lr lightgbm rf xgboost \
  --feature-bundle low_plus_medium \
  --n-splits 5 \
  --n-repeats 3 \
  --decision-threshold 0.5
```

Outputs:
- `artifacts/cv/metrics_cv.csv`
- `artifacts/cv/metrics_cv_summary.csv`

### 2) Feature Importance Export

```bash
python src/models_v2/export_feature_importance.py \
  --models lr lightgbm rf xgboost \
  --modes single two-stage \
  --feature-bundle low_plus_medium \
  --use-train-plus-val
```

Outputs:
- `artifacts/feature_importance/*_importance.csv`
- `artifacts/feature_importance/feature_importance_summary.csv`

### 3) LR Ablation (quick diagnosis)

```bash
python src/models_v2/run_lr_ablation.py \
  --feature-bundle low_plus_medium \
  --n-splits 5 \
  --n-repeats 2 \
  --decision-threshold 0.5
```

Outputs:
- `artifacts/ablation/lr_ablation_folds.csv`
- `artifacts/ablation/lr_ablation_summary.csv`

## Notes
- `docs/model_results.md` is legacy and preserved for historical context only.
- Use `docs/cv_results_summary.md` and `docs/feature_importance_results.md` for current project state.
- Use the protocol in `docs/eval_protocol.md` as the contract for fair model comparison.
