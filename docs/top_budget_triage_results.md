# Top-Budget Triage Results

## Purpose

Record the first top-budget review-prioritization check proposed in `docs/top_budget_triage_plan.md`.

This analysis treats the current model as a ranking tool:
- sort by `p(closed)` descending
- review only the top fixed budget slice
- leave the remaining rows unflagged

It does **not** interpret the non-reviewed rows as certified open.

## Artifact Used

Training-side artifact used for the initial check:
- `artifacts/projectc_oof_rf_final_pass1/projectc_oof_predictions.csv`

Runner:
- `src/models_v2/run_top_budget_analysis.py`

Output artifact:
- `artifacts/projectc_top_budget_oof_rf_final_pass1/top_budget_sweep.csv`

Dataset size:
- total rows: `3,082`
- total true closed: `282`
- closed base rate: `9.15%`

## Budget Sweep Summary

Selected review-budget results:

- top `1%`
  - `review_n=31`
  - `closed_in_review=17`
  - `closed_rate_in_review=54.8%`
  - `closure_capture=6.0%`
  - `lift_vs_base=5.99x`

- top `5%`
  - `review_n=155`
  - `closed_in_review=59`
  - `closed_rate_in_review=38.1%`
  - `closure_capture=20.9%`
  - `lift_vs_base=4.16x`

- top `10%`
  - `review_n=309`
  - `closed_in_review=83`
  - `closed_rate_in_review=26.9%`
  - `closure_capture=29.4%`
  - `lift_vs_base=2.94x`

- top `25%`
  - `review_n=771`
  - `closed_in_review=130`
  - `closed_rate_in_review=16.9%`
  - `closure_capture=46.1%`
  - `lift_vs_base=1.84x`

## Interpretation

The ranking signal is real but not strong enough to make the top-budget triage framing operationally compelling under the current sample regime.

What looks positive:
- the highest-risk slice is meaningfully enriched for true closures relative to the base rate
- the top `1%` to `5%` budgets show clear lift

What remains weak:
- closure capture is still low at practical review budgets
- top `10%` review captures only `29.4%` of true closures
- even top `25%` review captures only `46.1%` of true closures
- enrichment declines materially as the review budget expands

Relative to the plan's suggested bar:
- this does **not** approach outcomes like top `10%` capturing `70%` to `80%` of true closures
- the model therefore does not currently look strong enough as a review-prioritization tool either

## Holdout Decision

The plan recommended evaluating holdout only if one training-side review budget looked operationally promising.

That condition was not met here, so no fixed-budget holdout run was executed.

Reason:
- running holdout without a clearly justified frozen budget would add noise, not a meaningful product decision

## Current Conclusion

Current conclusion:
- under the current model and sample regime, the RF shows some closure-risk ranking signal
- however, that signal is not concentrated enough to justify a top-budget review workflow as a strong follow-up success case
- this leaves the project's main conclusion unchanged: there is still no production-pass binary model, and the top-budget triage fallback also looks weak
