# Feature Bundle V2 Rationale

## Purpose
Document the feature-level rationale for the model-specific v2 bundles:
- `v2_lr2` (LR two-stage draft/reference bundle)
- `v2_rf_single` (RF single draft predecessor)
- `v2_rf_single_no_spatial_prior` (RF single frozen winner)

This document complements:
- `docs/ceiling_study/feature_bundle_v2_conventions.md` (selection process rules)
- `docs/ceiling_study/feature_rationale.md` (global feature definitions)

## Shared Anchor Features (Used Across the V2 Bundle Round)

| Feature | Why Kept |
|---|---|
| `spatial_cluster_closed_rate` | Strong candidate spatial prior during the draft round; retained in LR-oriented bundles, but later removed from the final RF winner after confirm-backed ablation evidence. |
| `category_closure_risk` | Strong stable category-level prior; consistently high signal across models. |
| `geo_cluster_id` | Preserves coarse location identity signal for non-prior spatial effects. |
| `source_temporal_diversity` | Captures source-update consistency/churn not covered by pure recency bins. |
| `recency_days` | Core temporal freshness axis; supports threshold tradeoff tuning. |
| `fresh` | Mid-recency indicator with stable utility in tree and linear settings. |
| `very_fresh` | Captures highly updated records; complements continuous recency. |
| `stale` | Explicit stale-risk bin. |
| `very_stale` | Strong staleness tail-risk bin. |
| `address_completeness` | High-signal listing quality/completeness proxy. |
| `completeness_score` | Aggregate quality anchor; robust baseline feature. |
| `contact_diversity` | Core contact richness signal across model families. |
| `has_addresses` | Presence anchor for structured location metadata quality. |
| `has_alternate_categories` | Extra category detail/provenance richness signal. |
| `has_chain_pattern` | Distinct chain-vs-independent behavior prior. |
| `has_full_contact_info` | High-confidence listing richness indicator (web+phone+social). |
| `has_phones` | Reliable contact presence signal. |
| `has_websites` | Reliable digital-presence signal. |
| `name_length` | Simple lexical signal with stable utility. |
| `num_categories` | Category breadth/detail signal. |
| `num_datasets` | Independent-source diversity evidence. |
| `num_phones` | Contact richness depth signal. |
| `num_socials` | Social presence depth signal. |
| `num_sources` | Source corroboration depth signal. |
| `single_source` | Fragility/risk counter-signal for sparse provenance. |
| `rich_profile` | Thresholded profile richness with stable interaction value. |
| `ohe_primary_category__*` | Retains category identity effects beyond scalar priors. |
| `ohe_geo_cluster__*` | Retains discrete location effects beyond scalar priors. |
| `ohe_source_dataset__*` | Retains dataset-identity effects beyond count/diversity. |

## LR Two-Stage Specific Features (`v2_lr2`)

| Feature | Why Kept for LR2 |
|---|---|
| `has_name` | Stabilizes linear boundary on sparse records where namedness matters. |
| `has_primary_category` | Helps LR separate sparse/category-missing edge cases. |
| `has_socials` | Useful binary counterpart to `num_socials` for LR sparsity handling. |
| `num_addresses` | Adds linear detail axis not fully captured by `has_addresses`. |
| `single_source_no_socials` | Explicit high-risk interaction useful for LR decision boundary shaping. |

## RF Single Specific Features (`v2_rf_single`)

| Feature | Why Kept for RF |
|---|---|
| `has_long_name` | Tree splits can exploit nonlinear lexical-length effects directly. |
| `has_multiple_datasets` | Thresholded corroboration split signal works well in RF. |
| `has_multiple_sources` | Nonlinear corroboration threshold feature for tree partitioning. |
| `num_websites` | Additional digital-depth split feature for RF. |
| `recent_with_contacts` | Interaction retained because RF benefits from nonlinear interaction splits. |

## RF Follow-Up Variant (`v2_rf_single_no_spatial_prior`)

| Feature Change | Why |
|---|---|
| drop `spatial_cluster_closed_rate` | Split ablation showed RF improved when removing the scalar spatial prior while still retaining spatial ID features (`geo_cluster_id`, `ohe_geo_cluster__*`). This suggests the scalar prior may be overspecifying local risk for RF, while discrete spatial partitioning remains useful. |

Confirmed outcome:
- follow-up phased sweep + confirm established this bundle as the new overall diagnostic leader
- final confirmed RF operating point: `closed_f1=0.297`, `pr_auc_closed=0.262`, `accuracy=0.886`

## Why Some Features Were Not Kept

- Confidence-derived features remain excluded by policy.
- Several lower-impact aliases/redundant variants were intentionally omitted to reduce noise and bundle size.
- The v2 bundle process prioritized stable anchors + model-specific useful signals over maximal feature count.

## Status

- `v2_lr2` remains the documented LR v2 reference bundle for comparison work.
- `v2_rf_single` was an intermediate draft bundle used for ablation.
- `v2_rf_single_no_spatial_prior` is the frozen RF v2 bundle for this round after confirm-level validation.
