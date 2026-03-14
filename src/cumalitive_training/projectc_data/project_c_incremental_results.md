# Incremental Benchmark Results — All Models (Warm-Start)

## Setup

- **Dataset**: `data/project_c_samples/project_c_samples.csv` (3,425 rows)
- **Test set**: 20% stratified hold-out (685 rows, fixed throughout)
- **Training pool**: 80% (2,740 rows) split into 5 **stratified** batches (equal open/closed ratio per batch)
- **Feature bundle**: `low_plus_medium`
- **Featurizer + scaler**: fitted once on full training pool, frozen for all runs

### Hyperparameters (from HPO artifacts)

| Model         | Source | Key params |
|:--------------|:-------|:-----------|
| LR            | `hpo_optuna_lr_micro_pass1` | C=0.0311, class_weight={0:4.5, 1:1.0}, max_iter=2000, solver=lbfgs |
| XGBoost       | `xgboost_model_v2.py` defaults | n_estimators=300/batch, lr=0.05, max_depth=6, scale_pos_weight=class-ratio |
| LightGBM      | `lightgbm_hpo/hpo_selected_trials.csv` | n_estimators=263/batch, lr=0.0121, num_leaves=111, min_child_samples=68, class_weight=balanced |
| Random Forest | `hpo_optuna_rf_micro_pass1` | n_estimators=375/batch, max_depth=8, max_features=log2, min_samples_leaf=7, min_samples_split=6, class_weight=balanced |

### Training strategy

| Run type | Description |
|:---------|:------------|
| Incremental — After batch *k* | **Warm-start**: model trained on batch 1, then continued on batch 2, … through batch *k*. Accumulated knowledge from all prior batches. |
| Single run (all at once) | Train a **fresh** model on `all_batches_combined.csv` (2,740 rows) |

#### Warm-start mechanism per model

| Model | Mechanism | What happens each batch |
|:------|:----------|:------------------------|
| LR | `warm_start=True` | lbfgs resumes from previous coefficients |
| XGBoost | `xgb_model=prev_booster` | adds 300 new boosting rounds on top of prior model |
| LightGBM | `init_model=prev_model` | adds 263 new boosting rounds on top of prior model |
| Random Forest | `warm_start=True` | adds 375 new trees each batch |

> The featurizer and scaler are fitted **once** on the full training pool
> to ensure a consistent feature space across all batch iterations.
> No test labels are ever seen during featurizer fitting.

---

## Batch Sizes (Stratified)

| Batch        | Rows    | Open    | Closed |
| ------------ | ------: | ------: | -----: |
| 1            |     548 |     498 |     50 |
| 2            |     548 |     498 |     50 |
| 3            |     548 |     498 |     50 |
| 4            |     548 |     498 |     50 |
| 5            |     548 |     498 |     50 |
| **Test set** | **685** | **622** | **63** |

---

## Results

> Evaluation is always on the **same frozen test set** (20% hold-out).
>
> "After batch k" = model has been warm-started through batches 1 … k.
> The single run trains a fresh model on all 2,740 rows at once.

| Model             | Run                          | Batch rows | Total seen | Accuracy   | Closed Prec | Closed Rec | F1 (closed) | PR-AUC (closed) |
| ----------------- | ---------------------------- | ---------: | ---------: | ---------: | ----------: | ---------: | ----------: | --------------: |
| LR                | After batch 1                |        548 |        548 |     0.8555 |      0.2429 |     0.2698 |      0.2556 |          0.1893 |
| LR                | After batch 2                |        548 |       1096 |     0.8949 |      0.3200 |     0.1270 |      0.1818 |          0.1792 |
| LR                | After batch 3                |        548 |       1644 |     0.8934 |      0.3214 |     0.1429 |      0.1978 |          0.2093 |
| LR                | After batch 4                |        548 |       2192 |     0.8774 |      0.3158 |     0.2857 |      0.3000 |          0.2012 |
| LR                | After batch 5                |        548 |       2740 |     0.8978 |      0.3158 |     0.0952 |      0.1463 |          0.1669 |
| **LR**            | **Single run (all at once)** |       2740 |       2740 | **0.8555** |      0.2632 |     0.3175 |      0.2878 |          0.1738 |
|                   |                              |            |            |            |             |            |             |                 |
| XGBoost           | After batch 1                |        548 |        548 |     0.9109 |      1.0000 |     0.0317 |      0.0615 |          0.2385 |
| XGBoost           | After batch 2                |        548 |       1096 |     0.9124 |      0.8000 |     0.0635 |      0.1176 |          0.1994 |
| XGBoost           | After batch 3                |        548 |       1644 |     0.9109 |      0.6667 |     0.0635 |      0.1159 |          0.2253 |
| XGBoost           | After batch 4                |        548 |       2192 |     0.9124 |      0.8000 |     0.0635 |      0.1176 |          0.2495 |
| XGBoost           | After batch 5                |        548 |       2740 |     0.9080 |      0.0000 |     0.0000 |      0.0000 |          0.1854 |
| **XGBoost**       | **Single run (all at once)** |       2740 |       2740 | **0.9095** |      1.0000 |     0.0159 |      0.0312 |          0.2221 |
|                   |                              |            |            |            |             |            |             |                 |
| LightGBM          | After batch 1                |        548 |        548 |     0.8117 |      0.2054 |     0.3651 |      0.2629 |          0.1668 |
| LightGBM          | After batch 2                |        548 |       1096 |     0.3810 |      0.1068 |     0.7778 |      0.1877 |          0.1799 |
| LightGBM          | After batch 3                |        548 |       1644 |     0.7854 |      0.1957 |     0.4286 |      0.2687 |          0.1794 |
| LightGBM          | After batch 4                |        548 |       2192 |     0.7985 |      0.2093 |     0.4286 |      0.2813 |          0.1716 |
| LightGBM          | After batch 5                |        548 |       2740 |     0.7854 |      0.1957 |     0.4286 |      0.2687 |          0.1895 |
| **LightGBM**      | **Single run (all at once)** |       2740 |       2740 | **0.7664** |      0.1911 |     0.4762 |      0.2727 |          0.2270 |
|                   |                              |            |            |            |             |            |             |                 |
| Random Forest     | After batch 1                |        548 |        548 |     0.8029 |      0.2188 |     0.4444 |      0.2932 |          0.2166 |
| Random Forest     | After batch 2                |        548 |       1096 |     0.7723 |      0.1921 |     0.4603 |      0.2710 |          0.2004 |
| Random Forest     | After batch 3                |        548 |       1644 |     0.8307 |      0.2523 |     0.4286 |      0.3176 |          0.2176 |
| Random Forest     | After batch 4                |        548 |       2192 |     0.8044 |      0.2248 |     0.4603 |      0.3021 |          0.2266 |
| Random Forest     | After batch 5                |        548 |       2740 |     0.8058 |      0.2266 |     0.4603 |      0.3037 |          0.2317 |
| **Random Forest** | **Single run (all at once)** |       2740 |       2740 | **0.8175** |      0.2075 |     0.3492 |      0.2604 |          0.2039 |
|                   |                              |            |            |            |             |            |             |                 |

---

## Key Comparison — After All Batches vs Single Run

> Both have seen the same 2,740 training rows.
> Incremental = warm-started through 5 batches.  Single = trained on all at once.

| Model         | Metric          | After batch 5 | Single run | Δ (single − inc) |
| ------------- | --------------- | ------------: | ---------: | ---------------: |
| LR            | Accuracy        |        0.8978 |     0.8555 |          -0.0423 |
|               | Closed Prec     |        0.3158 |     0.2632 |          -0.0526 |
|               | Closed Rec      |        0.0952 |     0.3175 |          +0.2222 |
|               | F1 (closed)     |        0.1463 |     0.2878 |          +0.1414 |
|               | PR-AUC (closed) |        0.1669 |     0.1738 |          +0.0069 |
| XGBoost       | Accuracy        |        0.9080 |     0.9095 |          +0.0015 |
|               | Closed Prec     |        0.0000 |     1.0000 |          +1.0000 |
|               | Closed Rec      |        0.0000 |     0.0159 |          +0.0159 |
|               | F1 (closed)     |        0.0000 |     0.0312 |          +0.0312 |
|               | PR-AUC (closed) |        0.1854 |     0.2221 |          +0.0367 |
| LightGBM      | Accuracy        |        0.7854 |     0.7664 |          -0.0190 |
|               | Closed Prec     |        0.1957 |     0.1911 |          -0.0046 |
|               | Closed Rec      |        0.4286 |     0.4762 |          +0.0476 |
|               | F1 (closed)     |        0.2687 |     0.2727 |          +0.0041 |
|               | PR-AUC (closed) |        0.1895 |     0.2270 |          +0.0376 |
| Random Forest | Accuracy        |        0.8058 |     0.8175 |          +0.0117 |
|               | Closed Prec     |        0.2266 |     0.2075 |          -0.0190 |
|               | Closed Rec      |        0.4603 |     0.3492 |          -0.1111 |
|               | F1 (closed)     |        0.3037 |     0.2604 |          -0.0433 |
|               | PR-AUC (closed) |        0.2317 |     0.2039 |          -0.0278 |

---

## Notes on Results

### What the experiment shows

- **Incremental (warm-start)**: the model accumulates knowledge across
  batches.  After batch 5 it has seen all 2,740 training rows, but via
  5 sequential warm-start steps rather than one big `.fit()` call.
- **Single run**: a fresh model trained on all 2,740 rows at once.
- The Δ column shows how the single run compares to the fully
  warm-started model (after all 5 batches).

### Interpretation

- **Positive Δ** = single run outperforms warm-start → training on all
  data simultaneously gives the optimizer a better global view.
- **Negative Δ** = warm-start outperforms single run → sequential
  exposure helped (possible regularisation effect, or the optimizer
  found a better local minimum via the warm-start trajectory).
- **Near-zero Δ** = both strategies are equivalent.

### How warm-start differs from training on each batch alone

- **Batch-alone** (previous experiment): each batch trains a fresh model.
  The model only ever sees ~548 rows.  Performance is unstable because
  each batch has only ~50 closed examples.
- **Warm-start** (this experiment): the model carries forward its learned
  parameters.  By batch 5 it has been trained on all 2,740 rows
  sequentially.  Performance should converge toward the single-run result.

---

## File Artifacts

| File | Description |
|:-----|:------------|
| `data/project_c_samples/batches/test_set.csv` | Fixed 20% held-out test set |
| `data/project_c_samples/batches/batch_1.csv` … `batch_5.csv` | Individual stratified training batches |
| `data/project_c_samples/batches/all_batches_combined.csv` | All 5 batches combined (single-run input) |
