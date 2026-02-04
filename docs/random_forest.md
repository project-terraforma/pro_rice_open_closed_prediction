# Random Forest: Rationale and Tradeoffs

This document explains the design decisions, feature selection, and performance tradeoffs of the random forest model for open/closed prediction.

## Goals and Constraints

- **Leverage tree-based ensemble** to capture non-linear relationships and feature interactions naturally.
- **Handle class imbalance** using balanced class weights.
- **Provide comparison** against linear models (logistic regression).
- **Maintain reasonable performance** for production consideration.

## Architecture

**Model:** Random Forest Classifier
- Ensemble of 100 decision trees
- Balanced class weighting to handle 91% open imbalance
- Maximum tree depth: 10 (prevent overfitting)

**Hyperparameters:**
```python
RandomForestClassifier(
    n_estimators=100,           # 100 trees
    class_weight='balanced',    # Handle imbalance
    max_depth=10,              # Prevent deep overfitting
    random_state=42            # Reproducibility
)
```

## Feature Selection (20 Features)

**Includes Confidence Scores:**
- `overall_confidence` - Global system confidence
- `mean_source_conf` - Average per-source confidence
- `max_source_conf` - Maximum per-source confidence
- `conf_very_high`, `conf_high`, `conf_medium`, `conf_low`, `conf_very_low` - Binned confidence

**Contact Information (3 features):**
- `has_websites` - Website presence
- `has_phones` - Phone presence
- `has_socials` - Social media presence

**Data Source (1 feature):**
- `num_sources` - Number of data sources

**Business Signals (4 features):**
- `has_brand` - Brand/chain presence
- `has_primary_category` - Category presence
- `name_length` - Place name length
- `num_addresses` - Address count

**Temporal (2 features):**
- `recency_days` - Days since last update
- `very_fresh` - Binary for recent (≤180 days)
- `stale` - Binary for stale (≥1500 days)

**Completeness (1 feature):**
- `completeness_score` - Sum of presence indicators

## Performance

**Validation Set:**
```
Open:   Precision: 0.929    Recall: 0.904    F1: 0.916
Closed: Precision: 0.250    Recall: 0.317    F1: 0.280
Accuracy: 0.850
```

**Test Set:**
```
Open:   Precision: 0.935    Recall: 0.881    F1: 0.908
Closed: Precision: 0.245    Recall: 0.387    F1: 0.300
Accuracy: 0.837
```

## Strengths

- Captures non-linear relationships naturally
- Feature interactions handled implicitly (no manual engineering needed)
- Provides feature importances (interpretability)
- Moderate accuracy (83.7%)
- Reasonable closed recall (38.7%)

## Weaknesses

- Lower accuracy than Logistic Regression (83.7% vs 90.4%)
- Low closed precision (24.5% vs 37.5% for Two-Stage LR)
- Relies on confidence scores (provider bias, inconsistency)
- Slower inference (ensemble of 100 trees vs linear model)
- Larger model size (~5MB vs ~50KB for LR)
- No Stage 1 filtering (processes all 100% of data)

### Why Only 20 Features

**Decision:** Use basic feature set with confidence instead of comprehensive engineering.

**Rationale:** Compare tree-based approach against logistic regression fairly.

**Result:** Even with minimal features, trees couldn't overcome:
1. Confidence score bias
2. Severe class imbalance (91% open)
3. Limited feature interactions to learn from

### Why max_depth=10

**Decision:** Limit tree depth to prevent overfitting.

**Rationale:** Deep trees overfit on small dataset (2.4K training samples).

### Why 100 Trees

**Decision:** Use standard ensemble size.

**Rationale:** Empirical sweet spot between performance and training time.

## When Random Forest Might Be Useful

1. **Comparison baseline:** Validate that LR isn't missing non-linear patterns
2. **Feature exploration:** Extract feature importances to guide engineering
3. **Production redundancy:** Dual model for critical systems (vote between RF and LR)
4. **Future improvements:** If better features found, RF might leverage them better

## Scalability

### Current
- Training: ~2 seconds (2.4K samples)
- Inference: ~10ms per record (100 trees × decision logic)
- Memory: ~5MB for trained model

### For 100M Places
- Batch inference: Process 1M at a time
- Estimated time: ~20 hours single machine
- **2.4x slower than Two-Stage LR** (also 20 hours for two-stage Stage 1 + Stage 2)
