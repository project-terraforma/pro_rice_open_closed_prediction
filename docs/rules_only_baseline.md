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
- `sources` signals: count of sources and max per-source confidence.
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
- `low_conf_cutoff = 0.40`
- `low_conf_penalty = 1.0`

## Why We Consider This “Rules-Only”
All weights, thresholds, and logic are **hand-set** and **deterministic**.  
No parameters are learned from data training, so it remains a rules-based baseline.
