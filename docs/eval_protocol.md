# Evaluation Protocol (Draft)

## Purpose
Define one fixed evaluation contract for all open/closed experiments so results are comparable and not driven by ad hoc threshold or split choices.

Status: draft for team review.

## 1) Label and Metric Conventions
- Ground-truth label: `open=1`, `closed=0`.
- Report metrics in every experiment:
  - Overall accuracy
  - Open precision
  - Open recall
  - Open F1
  - Closed precision
  - Closed recall
  - Closed F1 - useful for evaluating performance once u have determined the threshold
  - PR-AUC for the closed class - useful for picking the best model family
- Closed-class metric computation rule: - all this is really saying is that for these closed metrics "closed" is treated as the target/postive class
  - For threshold metrics, use predictions mapped to closed as the positive target.
  - For PR-AUC, use closed probability score (`p_closed = 1 - p_open`).

## 2) Data Split Policy
- Keep the existing held-out test split fixed and untouched for selection/tuning.
- Use train+val only for model and threshold selection.
- For robustness, run repeated stratified CV (cross-validation) on train+val: - each fold gets a turn as validation, reduces split luck, then for overall validation performance u do mean across folds, look at the spread across folds to ensure stability
  - Proposed default: 5 folds x 3 repeats (`random_state=42`).
- Test split is evaluated once per final chosen configuration.

## 3) Feature Policy (OKR Constraint)
- Allowed features: schema-native Overture fields with simple deterministic transforms.
- Disallowed:
  - external datasets or APIs
  - web-derived signals outside Overture schema
  - expensive per-place inference pipelines
- Every feature used in experiments must be listed in a feature inventory table with:
  - source field
  - transform
  - cost tier (`low` or `medium`)

## 4) Model Families in Ceiling Study
- Minimum set to run under the same feature policy:
  - Rules baseline
  - Logistic Regression
  - Random Forest or constrained GBDT
  - LightGBM (constrained)
- Optional variants (single-stage, two-stage, with/without source confidence) are allowed, but must follow this protocol.

## 5) Thresholding and Selection Rule
- Primary ranking metric (cross-validation): PR-AUC (closed).
- Operating threshold rule (cross-validation):
  - Choose threshold that maximizes closed F1
  - Subject to guardrail floors
- Open metrics are reported as guardrails/context, but are not the primary ranking metric.
- Proposed default precision floor: `closed_precision >= 0.30`.
- Proposed default accuracy floor: `accuracy >= 0.85`.
- If no threshold satisfies the floor, pick threshold with max closed F1 and document the precision shortfall.

### Rationale for Accuracy Floor (`0.85`)
- The dataset is strongly imbalanced (~90% open / ~10% closed), so accuracy alone is not a reliable ranking metric.
- A pure all-open policy can achieve high accuracy while failing closed detection objectives.
- We still include an accuracy floor to prevent selecting operating points that improve closed metrics only by causing excessive overall errors.
- `0.85` is intended as a guardrail based on current observed model ranges, not as a primary optimization target.
- Model ranking remains driven by closed-class objectives (PR-AUC and thresholded closed F1), with accuracy used as a minimum acceptable reliability constraint.

## 6) Final Model Decision Rule
- From all candidate model+feature-set combinations:
  - Step 1: apply guardrails/floors from Section 5.
  - Step 2: rank by mean CV PR-AUC(closed) and define a top band near the best score.
    - Initial heuristic: include configs within `0.01` absolute PR-AUC of the best.
    - This `0.01` value is a starting point and should be revisited after observing CV variability.
  - Step 3:
    - If only one config is in the top band, select it.
    - If multiple configs are in the top band, treat them as effectively tied on primary metric and select using pragmatic criteria:
      - lower measured end-to-end runtime cost (`feature_extraction_time + inference_time`) on the same hardware and batch-size protocol
      - lower fold-to-fold variance on closed metrics
      - lower inference runtime (if end-to-end cost is still tied)
      - lower feature cost tier (as secondary context)
      - simpler pipeline
  - Step 4 final deterministic tie-break (if still tied): choose the simplest model family.

### Rationale for Decision Rule
- Closed performance remains the primary objective, so model ranking is anchored to CV PR-AUC(closed).
- Small PR-AUC differences can be within split noise, so a top-band approach prevents over-optimizing to negligible deltas.
- The `0.01` band is an explicit, transparent starting heuristic rather than a fixed truth.
- Once CV variability is known, the top-band width should be calibrated to observed uncertainty (for example, fold-level standard deviation or confidence intervals).
- Cost and engineering simplicity are used only after primary-performance ties, which preserves performance-first selection while still supporting production practicality.
- Cost tie-breaking is based on measured end-to-end runtime rather than feature tier alone, since production cost can be dominated by either feature generation or model inference depending on implementation.

## 7) Robustness Criterion
- For selected top configs, report mean and standard deviation across CV splits for:
  - closed precision
  - closed F1
- Target stability check:
  - variation within 5% across folds/repeats, or
  - explicitly document instability and qualify conclusions.

## 8) Required Reporting Outputs
- Per-run artifact files:
  - `metrics_cv.csv` (per fold/repeat for all configs/models)
  - `metrics_test.csv` (single final test evaluation, only make for config/model that we chose)
  - `config.json` (model, feature set, threshold, seed)
- Summary tables in `docs/model_results.md`:
  - Ceiling comparison table (all required metrics)
  - Robustness table (mean/std on closed metrics)
  - Performance-per-cost table/curve
  - Final recommendation with rationale

## 9) Reproducibility Defaults
- Default random seed: `42`.
- Same seed list across all models for repeated CV.
- Any deviation from this protocol must be logged in the run summary.

## 10) Open Items for Team Review
- Priority review items (high contention / high impact):
  - Confirm primary ranking objective: PR-AUC(closed) as the performance anchor.
  - Confirm threshold guardrails and floors:
    - `closed_precision >= 0.30` (proposed)
    - `accuracy >= 0.85` (proposed)
    - whether to add an explicit `open_recall` floor.
  - Confirm top-band rule for near-tied models:
    - initial heuristic: within `0.01` PR-AUC of best
    - plan to recalibrate this band using observed CV variability.
  - Confirm tie-break policy uses measured end-to-end runtime cost
    (`feature_extraction_time + inference_time`) rather than feature tier alone.
- Additional review items:
  - Confirm whether final decision is anchored to thresholded closed F1 after PR-AUC shortlist.
  - Confirm exact low-cost vs medium-cost feature definitions.
