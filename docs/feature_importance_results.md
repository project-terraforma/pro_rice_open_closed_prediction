# Feature Importance Results (Low Plus Medium Bundle)

Date run: 2026-03-02  
Script: `src/models_v2/export_feature_importance.py`  
Feature bundle: `low_plus_medium`  
Training data: `train_split + val_split` (`--use-train-plus-val`)  
Rows: 3082 total (2800 open, 282 closed)

## Command Used

```bash
python src/models_v2/export_feature_importance.py \
  --models lr lightgbm rf xgboost \
  --modes single two-stage \
  --feature-bundle low_plus_medium \
  --use-train-plus-val
```

## Output Artifacts

- `artifacts/feature_importance/lr_single_low_plus_medium_importance.csv`
- `artifacts/feature_importance/lr_two-stage_low_plus_medium_importance.csv`
- `artifacts/feature_importance/lightgbm_single_low_plus_medium_importance.csv`
- `artifacts/feature_importance/lightgbm_two-stage_low_plus_medium_importance.csv`
- `artifacts/feature_importance/rf_single_low_plus_medium_importance.csv`
- `artifacts/feature_importance/xgboost_single_low_plus_medium_importance.csv`
- `artifacts/feature_importance/xgboost_two-stage_low_plus_medium_importance.csv`
- `artifacts/feature_importance/feature_importance_summary.csv`

Note: RF currently supports single-stage only.

## Top Features by Model

### Logistic Regression (single-stage)

| Rank | Feature                                           | Normalized Importance |
|-----:|---------------------------------------------------|----------------------:|
| 1 | `spatial_cluster_closed_rate` | 0.0792 |
| 2 | `category_closure_risk` | 0.0575 |
| 3 | `address_completeness` | 0.0483 |
| 4 | `fresh` | 0.0434 |
| 5 | `ohe_primary_category__pizza_restaurant` | 0.0289 |
| 6 | `ohe_primary_category__discount_store` | 0.0269 |
| 7 | `completeness_score` | 0.0248 |
| 8 | `ohe_primary_category__mobile_phone_store` | 0.0244 |
| 9 | `ohe_geo_cluster__CL_41.9_-87.6` | 0.0237 |
| 10 | `ohe_primary_category__coffee_shop` | 0.0235 |

### Logistic Regression (two-stage)

| Rank | Feature                                           | Normalized Importance |
|-----:|---------------------------------------------------|----------------------:|
| 1 | `spatial_cluster_closed_rate` | 0.0952 |
| 2 | `category_closure_risk` | 0.0722 |
| 3 | `address_completeness` | 0.0684 |
| 4 | `fresh` | 0.0557 |
| 5 | `ohe_primary_category__holiday_rental_home` | 0.0278 |
| 6 | `ohe_geo_cluster__CL_41.9_-87.6` | 0.0246 |
| 7 | `ohe_primary_category__coffee_shop` | 0.0229 |
| 8 | `ohe_primary_category__printing_services` | 0.0216 |
| 9 | `completeness_score` | 0.0207 |
| 10 | `ohe_primary_category__funeral_services_and_cemeteries` | 0.0171 |

### LightGBM (single-stage)

| Rank | Feature                                           | Normalized Importance |
|-----:|---------------------------------------------------|----------------------:|
| 1 | `category_closure_risk` | 0.2118 |
| 2 | `geo_cluster_id` | 0.1735 |
| 3 | `name_length` | 0.1721 |
| 4 | `spatial_cluster_closed_rate` | 0.1272 |
| 5 | `max_update_time` | 0.0474 |
| 6 | `address_completeness` | 0.0288 |
| 7 | `num_categories` | 0.0246 |
| 8 | `ohe_source_dataset__Microsoft` | 0.0246 |
| 9 | `recency_days` | 0.0173 |
| 10 | `completeness_score` | 0.0133 |

### LightGBM (two-stage)

| Rank | Feature                                           | Normalized Importance |
|-----:|---------------------------------------------------|----------------------:|
| 1 | `category_closure_risk` | 0.2243 |
| 2 | `geo_cluster_id` | 0.1746 |
| 3 | `name_length` | 0.1697 |
| 4 | `spatial_cluster_closed_rate` | 0.1083 |
| 5 | `max_update_time` | 0.0533 |
| 6 | `num_categories` | 0.0269 |
| 7 | `completeness_score` | 0.0247 |
| 8 | `address_completeness` | 0.0229 |
| 9 | `recency_days` | 0.0204 |
| 10 | `ohe_primary_category__lawyer` | 0.0190 |

### Random Forest (single-stage)

| Rank | Feature                                           | Normalized Importance |
|-----:|---------------------------------------------------|----------------------:|
| 1 | `spatial_cluster_closed_rate` | 0.4930 |
| 2 | `category_closure_risk` | 0.1437 |
| 3 | `name_length` | 0.0391 |
| 4 | `geo_cluster_id` | 0.0369 |
| 5 | `address_completeness` | 0.0339 |
| 6 | `max_update_time` | 0.0248 |
| 7 | `recency_days` | 0.0231 |
| 8 | `num_categories` | 0.0131 |
| 9 | `num_phones` | 0.0087 |
| 10 | `recent_with_contacts` | 0.0086 |

### XGBoost (single-stage)

| Rank | Feature                                           | Normalized Importance |
|-----:|---------------------------------------------------|----------------------:|
| 1 | `spatial_cluster_closed_rate` | 0.1987 |
| 2 | `num_datasets` | 0.0869 |
| 3 | `address_completeness` | 0.0627 |
| 4 | `stale` | 0.0450 |
| 5 | `has_phones` | 0.0288 |
| 6 | `very_stale` | 0.0285 |
| 7 | `ohe_geo_cluster__CL_36.1_-115.2` | 0.0233 |
| 8 | `ohe_primary_category__shopping` | 0.0220 |
| 9 | `category_closure_risk` | 0.0212 |
| 10 | `ohe_geo_cluster__CL_34.0_-118.3` | 0.0203 |

### XGBoost (two-stage)

| Rank | Feature                                           | Normalized Importance |
|-----:|---------------------------------------------------|----------------------:|
| 1 | `spatial_cluster_closed_rate` | 0.2249 |
| 2 | `address_completeness` | 0.0741 |
| 3 | `socials_present` | 0.0734 |
| 4 | `ohe_source_dataset__meta` | 0.0491 |
| 5 | `ohe_primary_category__flowers_and_gifts_shop` | 0.0295 |
| 6 | `max_update_time` | 0.0282 |
| 7 | `stale` | 0.0271 |
| 8 | `ohe_geo_cluster__CL_34.0_-118.3` | 0.0264 |
| 9 | `has_phones` | 0.0255 |
| 10 | `category_closure_risk` | 0.0222 |

## Cross-Model Patterns

- Spatial + category priors dominate across all models:
  - `spatial_cluster_closed_rate`
  - `category_closure_risk`
- Tree models rely more heavily on location identifiers:
  - `geo_cluster_id` and geo one-hot cluster features appear repeatedly.
- Basic profile quality and recency signals are consistently useful:
  - `address_completeness`, `name_length`, `max_update_time`, `recency_days`, `stale`.

## Training Warnings Observed

- LR (`single` and `two-stage`) showed `ConvergenceWarning` at `max_iter=1000`.
  - Suggested next step: increase `max_iter`, standardize continuous features, and/or test solver alternatives.
- Featurizer emitted Pandas `PerformanceWarning` (`DataFrame is highly fragmented`) during column construction.
  - This is performance-related, not a correctness failure, but worth optimizing for faster runs.
- LightGBM printed repeated `No further splits with positive gain`.
  - Common with constrained signal/imbalance and not necessarily fatal; still worth monitoring during hyperparameter tuning.

## Caveats

- Importance values are model-specific and not directly comparable in absolute meaning across model families.
- Results are from a single fit over train+val, not fold-averaged importances.

