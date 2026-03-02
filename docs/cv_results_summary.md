# Repeated CV Results Summary (Low Plus Medium)

Date run: 2026-03-02  
Script: `src/models_v2/run_cv_experiments.py`  
Feature bundle: `low_plus_medium`  
Splits: 5-fold CV, 3 repeats  
Decision threshold: `0.5`

## Command Used

```bash
python src/models_v2/run_cv_experiments.py \
  --models lr lightgbm rf xgboost \
  --feature-bundle low_plus_medium \
  --n-splits 5 \
  --n-repeats 3 \
  --decision-threshold 0.5
```

## Top Rows by Closed F1 (Mean)

| Model Family       | Mode      | Feature Bundle   | Accuracy Mean | Closed Precision Mean | Closed Recall Mean | Closed F1 Mean | PR-AUC Closed Mean |
|--------------------|-----------|------------------|--------------:|----------------------:|-------------------:|---------------:|-------------------:|
| LogisticRegression | two-stage | low_plus_medium  |         0.682 |                 0.146 |              0.505 |          0.226 |              0.130 |
| LogisticRegression | single    | low_plus_medium  |         0.618 |                 0.135 |              0.588 |          0.220 |              0.138 |
| LightGBM           | two-stage | low_plus_medium  |         0.882 |                 0.139 |              0.078 |          0.092 |              0.116 |
| LightGBM           | single    | low_plus_medium  |         0.871 |                 0.115 |              0.070 |          0.081 |              0.138 |
| RandomForest       | single    | low_plus_medium  |         0.877 |                 0.099 |              0.043 |          0.059 |              0.133 |
| XGBoost            | single    | low_plus_medium  |         0.882 |                 0.094 |              0.033 |          0.049 |              0.140 |
| XGBoost            | two-stage | low_plus_medium  |         0.890 |                 0.101 |              0.027 |          0.042 |              0.102 |

## Interpretation

- Two-stage tree-model collapse to all-open is resolved (closed metrics are non-zero).
- Logistic Regression is currently strongest on thresholded closed-class performance (`closed_f1`).
- Tree models have higher overall accuracy but much lower closed recall and closed F1 at threshold `0.5`.
- XGBoost has the highest PR-AUC closed, suggesting decent ranking quality but weak thresholded classification under the current threshold.

## Practical Takeaway

- If selecting by thresholded closed detection right now, LR (especially two-stage) is the best current candidate.
- If selecting by score ranking quality (threshold-independent), XGBoost is competitive and may improve with threshold tuning/calibration.
- Next comparison step should include threshold sweeps per model under the same guardrails.
