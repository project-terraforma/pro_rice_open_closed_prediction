# LightGBM Experiments: Findings and Conclusions

This document summarizes the LightGBM experiments and why we are **not** proceeding with it (for now).

## Summary
LightGBM underperformed the rules baseline and logistic regression on closed‑class performance. Even after tuning and weighting attempts, it did not produce a meaningful lift and often reduced overall accuracy.

## Configurations Tested
- **Two‑stage** LightGBM (rule filter → model on uncertain subset)
- **Single‑stage** LightGBM
- **Manual interactions ON/OFF**
- **Small hyperparameter sweep** (n_estimators, learning_rate, num_leaves, min_child_samples)
- **Class imbalance handling** via `class_weight="balanced"`
- **Scale_pos_weight** and **decision threshold** experiments (two‑stage)

## Key Observations
- **Two‑stage performed better than single‑stage**, but still weak on closed recall/precision.
- **Single‑stage was significantly worse** (lower accuracy and worse closed F1).
- The **hyperparameter sweep** did not improve closed F1 meaningfully.
- **Scale_pos_weight** in two‑stage hurt closed recall even further (stage‑1 filter already biases toward open).
- Feature importance consistently over‑weighted `name_length` and `recency_days`, suggesting weak signal strength.

## Representative Results (Validation)
Two‑Stage LightGBM (No Confidence):  
- Open: Precision ~0.92, Recall ~0.96, F1 ~0.94  
- Closed: Precision ~0.32, Recall ~0.19, F1 ~0.24  
- Accuracy ~0.89

Single‑Stage LightGBM (No Confidence):  
- Open: Precision ~0.92, Recall ~0.74, F1 ~0.82  
- Closed: Precision ~0.13, Recall ~0.37, F1 ~0.19  
- Accuracy ~0.71

Hyperparameter sweep (single‑stage) max closed F1 ≈ **0.21**.

## Conclusion
LightGBM does **not** currently outperform:
- the rules‑only baseline, or
- logistic regression baselines.

Given the small labeled set and weak closed‑class signals, LightGBM is unlikely to provide a meaningful lift without stronger features or additional upstream signals.

## Next Steps
- Focus on **logistic regression** and **feature engineering**.
- Revisit LightGBM only if we obtain stronger signals or significantly more labeled data.
