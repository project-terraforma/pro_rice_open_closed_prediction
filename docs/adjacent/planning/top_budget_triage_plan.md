# Top-Budget Triage Plan

## Purpose

Document a lightweight follow-up analysis that treats the current model as a ranking tool for review prioritization.

This plan is separate from:
- the hard binary production-gate evaluation
- the two-bucket `confidently_open` / `review_queue` analysis

Instead, it asks:

- if we review only the highest `p(closed)` slice, how many true closed places do we recover?

## Why This Follow-Up Is Worth Checking

The two-bucket analysis suggested that a threshold-based open-safe policy was not very compelling under the current sample regime:
- the confidently-open bucket stayed relatively small unless the review bucket became very large

That does **not** rule out a different triage use case:
- flag only the highest-risk places
- review a fixed top budget
- leave the rest unflagged

This framing is often more natural for maintenance workflows than one global binary threshold.

## Core Question

Can the model concentrate a large share of actual closed places into a relatively small top-ranked slice?

Examples:
- top `1%`
- top `5%`
- top `10%`
- top `20%`

## Recommended Data Source

Use project-side OOF predictions first, not holdout.

Reason:
- this is where review-budget selection should happen
- it avoids contaminating the holdout

Current artifact to use:
- `artifacts/projectc_oof_rf_final_pass1/projectc_oof_predictions.csv`

If the training-side result looks promising, then evaluate the fixed review budget once on:
- `artifacts/holdout_test_rf_final_pass1/holdout_test_predictions.csv`

## Policy Definition

For a chosen review budget `b`:

1. sort rows by `p(closed)` descending
2. select the top `b%` of rows
3. treat those as the review slice
4. leave the remaining rows unflagged

Important:
- this does **not** mean the remaining rows are certified open
- it only means no action is taken on them under this policy

## Metrics To Report

For each review budget, report:

- `review_budget_pct`
- `review_n`
- `closed_base_rate`
- `closed_in_review`
- `closed_rate_in_review`
- `closure_capture`
  - fraction of all true closed places found within the review slice
- `lift_vs_base`
  - `closed_rate_in_review / closed_base_rate`

Optional:
- `open_in_review`
- cumulative capture curve across budget levels

## What Would Count As Promising

This is not a strict rule, but the general pattern you would want is:

- small review slice
- high closure capture
- strong enrichment relative to base rate

Examples of potentially interesting outcomes:
- top `10%` captures `>= 70%` or `80%` of true closures
- top `5%` is several times more closure-dense than the base rate

Weak outcomes would look like:
- needing to review a very large share of the dataset
- only modest enrichment over base rate
- low closure capture even at moderate review budgets

## Why This Can Be Done Quickly

This analysis should be cheap because it requires:
- no new feature engineering
- no new HPO
- no new bundle optimization
- no new model fitting if the OOF prediction file already exists

It is only a ranking / budget sweep over an existing prediction artifact.

## Suggested Execution Order

1. Load `artifacts/projectc_oof_rf_final_pass1/projectc_oof_predictions.csv`
2. Sort by `p_closed_oof` descending
3. Sweep review budgets such as:
   - `1%`
   - `2%`
   - `5%`
   - `10%`
   - `15%`
   - `20%`
   - `25%`
4. Compute capture and lift metrics at each budget
5. If one budget looks operationally promising, freeze that budget
6. Only then evaluate that fixed budget once on holdout

## Recommended Framing For Interpretation

If the result is strong:
- the model may still be useful as a ranking/prioritization tool even if it is not good enough for hard binary production labeling

If the result is weak:
- the current model likely does not create enough concentrated closure signal to justify a review-prioritization workflow either

## Relationship To Current Project Conclusion

This plan does not change the current main finding:
- under the current binary production gate, the project does not yet have a production-pass model

It is only a follow-up to test whether the same model might still have value as a top-risk review prioritizer.

## Fresh-Session Prompt

If continuing in a fresh session, start with:

1. `docs/adjacent/planning/open_triage_policy_plan.md`
2. `docs/adjacent/planning/top_budget_triage_plan.md`
3. `artifacts/projectc_oof_rf_final_pass1/projectc_oof_predictions.csv`
4. `src/models_v2/run_two_bucket_analysis.py`

Then implement a small top-budget sweep runner or notebook analysis over `p_closed_oof`.
