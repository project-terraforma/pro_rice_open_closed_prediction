# Problem Scope: Open vs Closed Prediction

## Problem Definition
Build a model that predicts whether a place is `open` (`1`) or `closed` (`0`) using Overture place-record attributes.

This is a binary classification task on a strongly imbalanced distribution that is close to production reality.

## What We Have Access To
- Labeled open/closed parquet sample used in this project (`train/val/test` splits).
- Overture place schema fields available in that sample (e.g., `sources`, `names`, `categories`, `websites`, `phones`, `socials`, `brand`, `addresses`, `geometry`, `bbox`).
- Source provenance metadata (`sources.dataset`, `sources.property`, `sources.update_time`, etc.).

## Confirmed Constraints from Overture Team
- Labels come from an external provider using agents/web search and may include noise.
- Labeling was done in Dec 2025; no explicit timestamp field is provided for the sample snapshot.
- Class imbalance (~90% open / ~10% closed) reflects production and should be treated as part of the real problem.
- No larger labeled open/closed sample is currently available.
- Upstream provider signals are not available for this project.
- Overture currently does not impose constraints on trying external data sources.

## Important Non-Goal / Exclusions
- `data/matching_validation/samples_3k_project_c_updated.csv` is not for open/closed prediction.
- That dataset is for place matching validation (`label=1` match, `label=0` no match), so it must not be used as open/closed ground truth.

## Modeling Implications
- Favor robust methods under imbalance (class weighting, threshold tuning, error tradeoff analysis).
- Treat confidence fields cautiously:
  - global/provider confidence can leak provider-side status logic,
  - some providers emit near-constant confidence values.
- Focus first on schema-native features that are scalable and reproducible.

## Current Practical Objective
Given available constraints, maximize open/closed performance using only valid open/closed data and defensible features, while clearly documenting:
- tradeoffs (open vs closed precision/recall),
- known data limitations,
- what additional signals or labels would most likely improve performance.
