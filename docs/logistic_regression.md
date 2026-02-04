# Logistic Regression: Rationale and Tradeoffs

This document explains the design decisions, feature engineering approach, and tradeoffs in the logistic regression models (both single-stage and two-stage variants).

## Goals and Constraints

- **Remove reliance on provider confidence scores** which are inconsistent across sources and potentially biased.
- **Leverage rich feature engineering** to capture business signals (contact info, data freshness, completeness).
- **Maximize accuracy** while maintaining high closed precision (minimize false closed predictions).
- **Use a hybrid two-stage approach** combining deterministic rules with learned parameters for efficiency.

## Feature Engineering Philosophy

**Key Insight:** Instead of using provider confidence scores, use observable business signals that indicate operational status.

### 40 Engineered Features (No Confidence)

**Contact Information (9 features):**
- Website presence, count, and multiplicity
- Phone presence, count, and multiplicity
- Social media presence, count, and multiplicity

**Why:** Active businesses maintain multiple contact channels. More channels = more operational.

**Data Source Features (4 features):**
- Number of sources and datasets
- Multiple sources/datasets indicators

**Why:** Places maintained by multiple independent datasets are more likely active and current.

**Brand & Category (4 features):**
- Brand presence
- Primary category and alternate categories
- Category count

**Why:** Branded chains have centralized management. Rich categorization suggests detailed record maintenance.

**Address Features (4 features):**
- Address count, presence, multiplicity
- Address field completeness score

**Why:** Complete, verified addresses indicate well-maintained records.

**Name Features (4 features):**
- Name presence and length
- Long/short name indicators
- Chain pattern detection (McDonald's, Starbucks, etc.)

**Why:** Chain presence indicates well-known establishments. Name characteristics hint at record quality.

**Temporal Features (6 features):**
- Recency (days since last update)
- Very fresh (≤90 days), fresh (90-365 days), stale (>730 days), very stale (>1825 days)
- Source temporal diversity (std dev of update times)

**Why:** Fresh data reflects currently active places. Multiple updates show ongoing maintenance.

**Composite Features (8 features):**
- Completeness score (0-7)
- Rich profile indicator
- Contact diversity (0-3)
- Full contact info indicator
- Interaction features (brand×contact, recency×contact, sources×contact)

**Why:** Capture synergistic signals (e.g., brand + fresh data stronger than brand alone).

## Why NO Confidence Features

**Confidence scores intentionally excluded:**
- Overall confidence (existence score)
- Per-source confidence scores
- Confidence binning (conf_very_high, conf_high, etc.)

**Rationale:**
1. **Provider bias:** Different sources have constant confidence (Microsoft always 0.77)
2. **Task mismatch:** Confidence measures existence, not operational status
3. **Empirical proof:** Removing confidence improved accuracy from 88.3% → 90.4%

## Model Variants

### Variant 1: Single-Stage Logistic Regression

**Architecture:**
- Single classifier trained on full dataset
- Balanced class weights to handle 91% open imbalance
- All 40 features used for every prediction

**Test Performance:**
```
Open:   Precision: 0.941    Recall: 0.824    F1: 0.879
Closed: Precision: 0.214    Recall: 0.484    F1: 0.297
Accuracy: 0.793
```

**Hyperparameters:**
- `class_weight='balanced'` - Automatic minority class weighting
- `random_state=42`
- `max_iter=1000`

**Pros:**
- Simple, interpretable single model
- Good closed recall (48.4%)
- Fast training

**Cons:**
- Lowest accuracy (79.3%)
- Low closed precision (21.4%) - many false closed predictions
- Treats obvious and uncertain cases equally

**Best For:** When catching closed places is prioritized over overall accuracy

---

### Variant 2: Two-Stage Logistic Regression ⭐ (Recommended)

**Stage 1 Results:**
- Filters: 69.8% of training data
- Accuracy on filtered: 92.6%
- Reduces Stage 2 imbalance from 91% → 14%

**Stage 2 - ML Classification (Learned):**
- Trains logistic regression on 30.2% uncertain cases only
- 725 training samples, 13.1% closed distribution
- `class_weight={0: 3, 1: 1}` - Higher weight for closed class

**Test Performance:**
```
Open:   Precision: 0.916    Recall: 0.984    F1: 0.949
Closed: Precision: 0.375    Recall: 0.097    F1: 0.154
Accuracy: 0.904
```

**Pros:**
- **Best overall accuracy (90.4%)**
- **Highest closed precision (37.5%)** - Minimizes false closed predictions
- Stage 1 rules transparent and interpretable
- Efficient: ~70% of data skips ML processing
- Stage 1 deterministic (no model bias)

**Cons:**
- Low closed recall (9.7%) - Misses ~90% of closed places
- More complex than single-stage
- Stage 2 trained on smaller, slightly different distribution

### Priority 1: Minimize False Closed Predictions

**Implementation:**
1. Two-stage approach filters obvious cases first
2. Stage 2 class weights favor closed class (3:1 ratio)
3. Result: 37.5% closed precision (vs 21.4% single-stage)

### Priority 2: Maximize Overall Accuracy

**Implementation:**
1. Rich feature engineering (40 features capture multiple signals)
2. Two-stage architecture handles imbalance more effectively
3. Result: 90.4% accuracy (vs 79.3% baseline)

### Trade-off: Closed Recall

**Closed recall: 9.7%** (two-stage) vs **48.4%** (single-stage)

**Why acceptable:**
- Product goal is travelers getting accurate status
- Better to say "uncertain" than "definitely closed" when unsure
- Users can verify uncertain cases (call, check website)

## Feature Engineering Decisions

### Included (Validated)
All 40 features perform well. Key drivers:
- Multiple contact types (strongest signal)
- Data source diversity
- Brand presence
- Recency and freshness
- Completeness score

### Excluded (Intentional)
- Confidence scores (provider bias)
- Geographic features (requires additional engineering)
- Review data (not available)
- Operating hours (not available)

## Reproducibility

**Saved artifacts:**
- Model coefficients: `logistic_regression_model.pkl`
- Feature engineering: Deterministic code

**To retrain:**
```bash
python src/models/two_stage_logistic_regression.py
```

**To predict:**
```python
from src.models.two_stage_logistic_regression import LogisticRegressionModel