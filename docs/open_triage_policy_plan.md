# Open-Safe Triage Policy Plan

## Purpose

Document a low-cost follow-up analysis that reframes the current open/closed model as a maintenance triage tool rather than a single hard binary classifier.

This plan is motivated by two observations:

1. false closed predictions are the high-cost error
2. a ranked closure-risk signal may still be useful even if the current production gate for a hard binary decision is not met

## Why This Follow-Up Exists

The current project evaluation policy assumes a binary operating point:
- predict `open` or `closed`
- require strong `closed_precision`
- require high overall `accuracy`

That framing makes sense **if** the intended Overture use case is a strong binary decision such as a production closed/open flag.

However, if the more realistic use case is database maintenance and review prioritization, then a different policy may be more appropriate:
- leave clearly open places alone
- route uncertain or suspicious places into a review/risk queue
- use `p(closed)` as a ranking signal rather than forcing a hard binary action for every row

## Proposed Policy: Two Buckets First

Start with a simple two-bucket policy:

1. `confidently_open`
   - if `p(closed) < t_open`
   - model is confident enough that the place is low-risk for closure
   - default action: leave the place alone

2. `review_queue`
   - if `p(closed) >= t_open`
   - do **not** treat this as automatically closed
   - instead, sort by `p(closed)` descending and use this as a review/risk queue

Important:
- this is **not** equivalent to saying the review bucket is closed
- it only means those places are suspicious enough to warrant attention

## Why Two Buckets First

This is the cheapest useful analysis because it does not require:
- new HPO
- new feature-bundle optimization
- new model training

It can be evaluated using the probabilities already produced by the current model artifacts.

It also isolates the key question:
- can the model safely clear a large portion of places as low-risk open while concentrating true closures into a smaller suspicious bucket?

## Evaluation Questions

For a sweep over candidate `t_open` values, measure:

1. Coverage of the `confidently_open` bucket
   - what fraction of all rows are left alone?

2. Safety of the `confidently_open` bucket
   - among rows assigned to `confidently_open`, how many are actually closed?
   - equivalently: how low is the false-open / missed-closure rate in the open-safe bucket?

3. Enrichment of the `review_queue`
   - what fraction of the review bucket is actually closed?
   - how much more closure-dense is the review bucket than the raw base rate?

4. Capture of true closures
   - what fraction of all actual closed places fall into the review bucket?

5. Review workload
   - how large is the review bucket?
   - for example, can the review slice be kept near `10%`, `20%`, or `30%` of the data while still capturing a large share of actual closures?

## Example Policy Questions

Examples of useful operational questions:

- What threshold leaves `80%` of places in a confidently-open bucket?
- In that remaining `20%`, what is the actual closure rate?
- What share of all truly closed places are captured in that `20%` review slice?
- Is the top `5%` or `10%` by `p(closed)` enriched enough to justify manual review?

These questions are more directly tied to maintenance workflows than one global binary threshold.

## Metrics To Report

For each threshold candidate, report at least:

- `open_bucket_fraction`
- `review_bucket_fraction`
- `closed_rate_in_open_bucket`
- `closed_rate_in_review_bucket`
- `closure_capture_in_review_bucket`
- review-bucket lift over base rate

If using a top-K or top-percent review budget, also report:
- `review_budget_pct`
- `actual_closed_found_within_budget`

## Why This Should Be Quick To Explore

This should be a relatively fast analysis because:

- no new label generation is needed
- at worst, only one lightweight OOF prediction pass is required on the frozen config
- only threshold/bucket sweeps are needed
- the outputs are simple aggregates over already-scored examples

Rigorous workflow for `project_c`:

1. generate OOF predictions on `train + val` for the frozen config
2. sweep `t_open` on those OOF predictions
3. choose the triage threshold on the training side only
4. apply that fixed threshold once to:
   - `artifacts/holdout_test_rf_final_pass1/holdout_test_predictions.csv`

Useful prediction artifacts:
- training-side OOF:
  - `artifacts/projectc_oof_eval/...`
- holdout:
  - `artifacts/holdout_test_rf_final_pass1/holdout_test_predictions.csv`
- optional Alex-side comparison:
  - `artifacts/alex_retrain_rf_final_pass1/alex_retrain_oof_predictions.csv`

## Relationship To Existing Project Conclusions

This plan does **not** replace the current binary-gate conclusion:
- under the current binary production gate, the project does not yet have a production-pass model

Instead, it asks a narrower follow-up:
- even if the model is not strong enough for hard binary production labeling, is it already useful as a low-risk triage / review-ranking tool?

## Current Recommendation

Recommended first step:

1. Generate OOF predictions for the frozen RF on `train + val`.
2. Run a two-bucket threshold sweep on those OOF predictions.
3. Select `t_open` on the training side only.
4. Apply that fixed threshold once to the holdout predictions.
5. Stop if the result is not operationally compelling; do not automatically expand into more complex policy variants.

## Scope Boundary

This document is intentionally limited to the two-bucket analysis.

Out of scope for now:
- threshold-based replacements for the existing two-stage architecture
- new stage-1 policy variants
- three-bucket extensions

Reason:
- the two-bucket analysis is intended as a quick decision check, not the start of a new model-variant branch

## Caveat

This plan is most appropriate if the intended use case is database maintenance / review prioritization.

If the intended use case is instead a hard binary open/closed production label, then the current binary gate remains the more appropriate primary evaluation framework.
