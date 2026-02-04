# Rules-Only Baseline: Rationale and Tradeoffs

This document explains the thought process behind the rules-only baseline, the tradeoffs considered, and how thresholds were selected.

## Goals and Constraints
- Use **only low-cost, schema-native features** (counts, booleans, simple thresholds).
- Keep the system **interpretable and deterministic** (no learned parameters).
- Prioritize **avoiding false “closed”** predictions while still catching a meaningful fraction of closed places.
- Provide a strong baseline to compare against ML models.

## Feature Selection (Low-Cost Only)
Included features:
- Global `confidence` (existence confidence).
- `sources` signals: count of sources.
- Presence of contact signals: `websites`, `phones`, `socials`, `addresses`.
- Recency (derived from `sources.update_time`, using latest timestamp as a snapshot proxy).

Excluded for v1:
- Categories and names (useful but heavier to encode for rules-only).
- Geometry/bbox (requires spatial feature engineering).

## Scoring Strategy
Rather than a single if/else rule, we use a **weighted score** (still rules-only) so multiple weak signals can accumulate.

Key ideas:
- High confidence should strongly push toward **OPEN**.
- Missing phone/website, single source, and stale data should push toward **CLOSED**.
- When evidence is ambiguous, default to **OPEN** to avoid false-closed mistakes.

## Tradeoffs and Priorities
We explicitly prioritized:
1) **Closed precision** (minimize false-closed predictions).
2) **Open recall** (avoid missing open places).
3) **Open precision** (least harmful if it drops slightly).

This matches the product intuition: falsely marking an open place as closed is more harmful than the inverse.

## Threshold Sweeps
We performed threshold sweeps to tune the decision boundaries:
- **Open threshold sweep**: varied the score cutoff for predicting open.
- **Closed threshold sweep**: varied the score cutoff for predicting closed.

Result:
- Threshold tuning alone did **not** significantly improve closed performance.
- The score distribution had most records strongly on the OPEN side.

## Recency Sweep (Staleness)
We added recency and swept:
- `stale_days`: 730, 1095, 1500
- `stale_penalty`: 1.0, 1.5, 2.0

Result:
- More aggressive staleness rules improved **closed recall**, but reduced open recall and accuracy.
- We selected a **balanced** setting:  
  - `stale_days = 1500`  
  - `stale_penalty = 1.5`

This improves closed recall without overly damaging open precision.

## Confidence Adjustments (Low-Only)
We revised confidence handling to avoid over-rewarding “existence” when a place may still be closed:
- **Removed high-confidence boosts** entirely by default.
- Added **low-confidence penalty only**, and swept:
  - `low_conf_cutoff`: 0.2, 0.3, 0.4
  - `low_conf_penalty`: 1.0, 1.5, 2.0

Result:
- Best balance came from **`low_conf_cutoff = 0.40`** and **`low_conf_penalty = 1.0`**, which improved closed metrics while keeping open performance strong.
- This aligns with the idea that *existence confidence ≠ operating status*.

### Provider Guidance (Confidence Removed)
After feedback from the Overture team, we **removed confidence signals** in the rules baseline:
- The confidence score is computed provider-side and can be constant for some sources.
- For modeling, we were advised to see how far we can get **without** confidence.

We keep the earlier confidence-based sweeps and metrics **for documentation purposes only** (historical comparison), but the **current baseline does not use confidence** (global or per-source).

## Threshold Re-Sweep (After Confidence Change)
After adjusting confidence, we re-ran the score threshold sweep:
- Selected **`open_threshold = 2.0`** and **`closed_threshold = 1.0`** as a better balance point.
- This improved closed recall while keeping open precision high.

## Final Defaults (Balanced)
Current defaults in `src/models/rules_only_baseline.py`:
- `open_threshold = 2.0`
- `closed_threshold = 1.0`
- `default_open = True`
- `fresh_days = 180`
- `stale_days = 1500`
- `fresh_bonus = 1.0`
- `stale_penalty = 1.5`
- `low_conf_cutoff = None`
- `low_conf_penalty = 0.0`

## Final Metrics (Validation and Test)
Reported with the finalized rules and thresholds. We keep **two variants** until the meaning of per‑source confidence is clarified.

### Variant A: No confidence signals (recommended default)
**Note:** Does not use global confidence or per‑source confidence.

**Validation (val split)**  
Command: `python src/models/rules_only_baseline.py --split val`  
Confusion: tp=585, tn=16, fp=47, fn=37  
Precision (open): 0.926  
Recall (open): 0.941  
F1 (open): 0.933  
Precision (closed): 0.302  
Recall (closed): 0.254  
F1 (closed): 0.276  
Accuracy: 0.877

**Test (test split)**  
Command: `python src/models/rules_only_baseline.py --split test`  
Confusion: tp=290, tn=3, fp=28, fn=22  
Precision (open): 0.912  
Recall (open): 0.929  
F1 (open): 0.921  
Precision (closed): 0.120  
Recall (closed): 0.097  
F1 (closed): 0.107  
Accuracy: 0.854

### Variant B: With per‑source confidence (experimental)
**Note:** Uses per‑source confidence; keep only for comparison until meaning is confirmed.

**Validation (val split)**  
Command: `python src/models/rules_only_baseline.py --split val` (with `use_source_conf=True`)  
Confusion: tp=574, tn=21, fp=42, fn=48  
Precision (open): 0.932  
Recall (open): 0.923  
F1 (open): 0.927  
Precision (closed): 0.304  
Recall (closed): 0.333  
F1 (closed): 0.318  
Accuracy: 0.869

**Test (test split)**  
Command: `python src/models/rules_only_baseline.py --split test` (with `use_source_conf=True`)  
Confusion: tp=278, tn=10, fp=21, fn=34  
Precision (open): 0.930  
Recall (open): 0.891  
F1 (open): 0.910  
Precision (closed): 0.227  
Recall (closed): 0.323  
F1 (closed): 0.267  
Accuracy: 0.840

## Recent Experiments (Post-Confidence Removal)

### Threshold Sweep
After removing confidence and per-source confidence, we re-ran the threshold sweep.
Observation: the sweep did not change predictions; top rows were identical across open/closed thresholds.

### Ablation Sweep
We tested removing feature groups to see if simpler rules improved performance:
- **Baseline** remained best overall for accuracy and open metrics.
- Removing socials + addresses slightly improved closed F1, but significantly reduced open recall and accuracy.
- Removing recency or using a “core only” rule set degraded performance.

Conclusion: keep the baseline feature set.

## Historical Results (Confidence-Based)
These are retained for comparison only; they were produced **before** removing confidence from the rules.

- Validation (val split) with confidence:  
  tp=553, tn=25, fp=38, fn=69,  
  Precision(open)=0.936, Recall(open)=0.889, F1(open)=0.912,  
  Precision(closed)=0.266, Recall(closed)=0.397, F1(closed)=0.318, Accuracy=0.844

- Test (test split) with confidence:  
  tp=268, tn=11, fp=20, fn=44,  
  Precision(open)=0.931, Recall(open)=0.859, F1(open)=0.893,  
  Precision(closed)=0.200, Recall(closed)=0.355, F1(closed)=0.256, Accuracy=0.813

## Why We Consider This “Rules-Only”
All weights, thresholds, and logic are **hand-set** and **deterministic**.  
No parameters are learned from data training, so it remains a rules-based baseline.
