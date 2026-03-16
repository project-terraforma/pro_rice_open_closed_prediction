# Alex-Filtered SF/NY Incremental Benchmark Results

## Setup

- **Dataset**: `data/sf_ny/alex_nyc_filtered.csv` + `data/sf_ny/alex_sf_filtered.csv`
- **Combined rows**: 79,734
- **Test set**: 15,947 rows (20% stratified hold-out)
- **Training pool**: 63,787 rows (5 stratified batches)
- **Feature bundle**: `low_plus_medium`
- **Label column**: `open` (1=open, 0=closed)
- **Excluded**: confidence, fsq_label, operating_status
- **Balanced distribution**: stratified by open/closed and alex_city (NYC/SF)

## Batch Sizes (Stratified)

| Batch | Rows | Open | Closed |
|-------|-----:|-----:|-------:|
| 1 | 12,758 | 10,803 | 1,955 |
| 2 | 12,758 | 10,803 | 1,955 |
| 3 | 12,757 | 10,803 | 1,954 |
| 4 | 12,757 | 10,803 | 1,954 |
| 5 | 12,757 | 10,803 | 1,954 |
| **Test** | **15,947** | **13,504** | **2,443** |

## Incremental Training Results (Warm-start across batches)


### LogisticRegression

| Batch | Accuracy | Closed Prec | Closed Rec | F1 (Closed) | PR-AUC | Time (s) |
|-------|----------|-------------|------------|-------------|--------|----------|
| After 1 | 0.8190 | 0.4464 | 0.7548 | 0.5610 | 0.5256 | 2.89 |
| After 2 | 0.8097 | 0.4319 | 0.7675 | 0.5528 | 0.5264 | 0.66 |
| After 3 | 0.8085 | 0.4298 | 0.7659 | 0.5506 | 0.5277 | 0.66 |
| After 4 | 0.8130 | 0.4368 | 0.7618 | 0.5552 | 0.5199 | 0.67 |
| After 5 | 0.8151 | 0.4398 | 0.7560 | 0.5561 | 0.5256 | 0.62 |

### RandomForest

| Batch | Accuracy | Closed Prec | Closed Rec | F1 (Closed) | PR-AUC | Time (s) |
|-------|----------|-------------|------------|-------------|--------|----------|
| After 1 | 0.7721 | 0.3878 | 0.8428 | 0.5312 | 0.5186 | 0.56 |
| After 2 | 0.7748 | 0.3908 | 0.8416 | 0.5337 | 0.5182 | 0.46 |
| After 3 | 0.7752 | 0.3914 | 0.8424 | 0.5345 | 0.5187 | 0.40 |
| After 4 | 0.7758 | 0.3920 | 0.8408 | 0.5347 | 0.5197 | 0.45 |
| After 5 | 0.7739 | 0.3901 | 0.8440 | 0.5336 | 0.5196 | 0.42 |

### XGBoost

| Batch | Accuracy | Closed Prec | Closed Rec | F1 (Closed) | PR-AUC | Time (s) |
|-------|----------|-------------|------------|-------------|--------|----------|
| After 1 | 0.8541 | 0.7164 | 0.0786 | 0.1416 | 0.5440 | 0.32 |
| After 2 | 0.8543 | 0.6776 | 0.0929 | 0.1634 | 0.5336 | 0.32 |
| After 3 | 0.8539 | 0.6840 | 0.0860 | 0.1527 | 0.5323 | 0.33 |
| After 4 | 0.8546 | 0.6911 | 0.0925 | 0.1632 | 0.5333 | 0.36 |
| After 5 | 0.8565 | 0.7305 | 0.0999 | 0.1757 | 0.5450 | 0.41 |

### LightGBM

| Batch | Accuracy | Closed Prec | Closed Rec | F1 (Closed) | PR-AUC | Time (s) |
|-------|----------|-------------|------------|-------------|--------|----------|
| After 1 | 0.7810 | 0.3970 | 0.8281 | 0.5367 | 0.5464 | 1.01 |
| After 2 | 0.7827 | 0.3981 | 0.8183 | 0.5356 | 0.5357 | 0.77 |
| After 3 | 0.7853 | 0.4006 | 0.8088 | 0.5358 | 0.5357 | 0.82 |
| After 4 | 0.7865 | 0.4033 | 0.8215 | 0.5410 | 0.5420 | 0.93 |
| After 5 | 0.7783 | 0.3922 | 0.8142 | 0.5294 | 0.5462 | 1.11 |


## Single-Run Training Results (All batches at once)

| Model | Accuracy | Closed Prec | Closed Rec | F1 (Closed) | PR-AUC | Time (s) |
|-------|----------|-------------|------------|-------------|--------|----------|
| LogisticRegression | 0.8129 | 0.4367 | 0.7622 | 0.5552 | 0.5263 | 1.57 |
| RandomForest | 0.7697 | 0.3855 | 0.8465 | 0.5297 | 0.5205 | 1.33 |
| XGBoost | 0.8490 | 0.8070 | 0.0188 | 0.0368 | 0.5622 | 0.84 |
| LightGBM | 0.7845 | 0.4022 | 0.8359 | 0.5431 | 0.5590 | 1.66 |

## Timing Comparison (per-batch vs single-run)

| Model | Incremental (Total) | Single-run | Overhead Ratio |
|-------|-------------------:|----------:|---------------:|
| LogisticRegression | 5.49 | 1.57 | 3.49x |
| RandomForest | 2.28 | 1.33 | 1.71x |
| XGBoost | 1.74 | 0.84 | 2.08x |
| LightGBM | 4.64 | 1.66 | 2.80x |

## Data Files

| File | Description |
|------|-------------|
| `data/sf_ny/batches/test_set.csv` | Fixed 20% hold-out test set |
| `data/sf_ny/batches/batch_1.csv` … `batch_5.csv` | Stratified training batches |
| `data/sf_ny/batches/all_batches_combined.csv` | All batches combined (single-run input) |

## Model Persistence

Models saved after each batch for warm-start continuation:
- `lr_batch_1.pkl` … `lr_batch_5.pkl` (Logistic Regression)
- `rf_batch_1.pkl` … `rf_batch_5.pkl` (Random Forest)
- `xgb_batch_1.pkl` … `xgb_batch_5.pkl` (XGBoost)
- `lgb_batch_1.pkl` … `lgb_batch_5.pkl` (LightGBM)

Load with: `model = joblib.load('lr_batch_5.pkl')` and continue training.