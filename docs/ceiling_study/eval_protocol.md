# Evaluation Protocol

## Purpose
Define the current evaluation contract for all open/closed experiments so model comparisons, bundle comparisons, and future data-expansion studies are measured the same way.

Status:
- current working policy for model-selection and bundle-selection
- replaces earlier single-floor/legacy-gate language except where explicitly labeled historical

## 1) Label and Metric Conventions

- Ground-truth label: `open=1`, `closed=0`.
- Always report:
  - `accuracy`
  - `open_precision`, `open_recall`, `open_f1`
  - `closed_precision`, `closed_recall`, `closed_f1`
  - `pr_auc_closed`
- Closed-class computation rule:
  - for thresholded metrics, treat `closed` as the positive class
  - for PR-AUC, use `p_closed = 1 - p_open`

Metric roles:
- `pr_auc_closed`:
  - threshold-independent ranking signal
  - useful for comparing model quality when threshold choice is not fixed yet
- `closed_f1`:
  - thresholded operating-point quality
  - primary metric for choosing among diagnostic candidates after thresholding
- `closed_precision`:
  - main deployability guardrail
- `accuracy`:
  - global reliability guardrail, not primary optimization target

## 2) Data Split Policy

- Keep held-out test split fixed and untouched during tuning.
- Use `train + val` only for HPO, `k` tuning, threshold tuning, and bundle comparison.
- Default robustness protocol:
  - search/tuning CV: `5 folds x 1 repeat` or `5 folds x 3 repeats`, depending on stage
  - confirm CV: `5 folds x 10 repeats` for final shortlisted operating points
- Evaluate the held-out test split once for the final chosen configuration only.

## 3) Feature Policy

- Allowed:
  - schema-native Overture features
  - simple deterministic transforms
  - low/medium-cost aggregate or fold-safe prior features
- Disallowed:
  - external APIs or external datasets
  - runtime web probing features
  - expensive per-record inference pipelines
  - confidence-derived features excluded by team policy
- Special handling:
  - label-derived priors such as `category_closure_risk` and `spatial_cluster_closed_rate` are allowed only if computed fold-safely

## 4) Evaluation Views

We maintain two distinct evaluation views and do not mix them:

- Fair cross-model comparison:
  - same feature bundle
  - same featurizer settings
  - same CV protocol
  - same gate logic
- Per-model ceiling:
  - model-specific bundle refinement is allowed
  - model-specific `k` and threshold tuning is allowed
  - CV protocol and gate logic remain fixed

This separation matters because a model can lose under the fair-comparison view but win under its own optimized bundle.

## 5) Current Gate Policy

The current policy is dual-gate:

### Production Gate

Purpose:
- deployability filter
- precision-first operating point

Required floors:
- `accuracy >= 0.90`
- `closed_precision >= 0.70`
- `closed_recall >= 0.05`

If multiple production-pass candidates exist, rank by:
1. `closed_precision`
2. `closed_f1`
3. `pr_auc_closed`
4. `accuracy`

If no production-pass candidate exists:
- keep reporting the best-effort production-oriented point
- explicitly document the shortfall
- do not claim production readiness

### Diagnostic Gate

Purpose:
- research-time ranking under current data constraints
- compare which model/config is most useful for iteration

Required floors:
- `accuracy >= 0.84`
- `closed_precision >= 0.20`

If multiple diagnostic-pass candidates exist, rank by:
1. `closed_f1`
2. `pr_auc_closed`
3. `closed_precision`
4. `accuracy`

If no diagnostic-pass candidate exists:
- select fallback by best `closed_f1`
- document the shortfall

## 6) How We Determine "Goodness"

"Goodness" is not one number. It is determined in layers.

### For Model/Bundle Development

Primary question:
- under the current data regime, which candidate gives the best closed-detection performance without violating basic reliability floors?

Decision rule:
1. apply the appropriate gate
2. among passers, rank using the gate-specific metric order
3. if no candidate passes, use the defined fallback and document the shortfall
4. confirm the chosen operating point with stronger CV

### For Threshold Selection

Thresholds are not chosen by PR-AUC directly.

Threshold rule:
1. generate CV fold predictions for a frozen config
2. sweep thresholds
3. apply the relevant gate
4. choose the best gated threshold using the gate-specific ranking rule

In practice:
- diagnostic thresholding optimizes for `closed_f1` subject to `accuracy` and `closed_precision` floors
- production thresholding optimizes for `closed_precision` first, then `closed_f1`, subject to all production floors

### For Final Frontier Conclusions

We treat a result as materially better only if:
- the gain is visible on primary closed-class metrics (`closed_f1`, `pr_auc_closed`)
- the gain survives confirm CV
- the gain is larger than ordinary fold/repeat noise

This is why confirmed operating points carry more weight than ablation or threshold-stage-only results.

## 7) Execution Order

Current recommended workflow:
1. Freeze data split policy and gate definitions.
2. Run HPO on the chosen feature bundle.
3. Freeze shortlisted hyperparameter configs.
4. Tune featurizer `k` values.
5. Tune thresholds.
6. Run confirm CV on finalists.
7. Only after that, run bundle-v2 refinement if needed.
8. Re-run the same phased workflow on any serious v2 candidate.

## 8) Robustness Criterion

For any final claimed operating point, report:
- mean and standard deviation across confirm CV for:
  - `closed_precision`
  - `closed_recall`
  - `closed_f1`
  - `pr_auc_closed`

Interpretation rule:
- treat unstable gains with caution
- prefer confirmed improvements that remain directionally consistent across repeats

## 9) Required Reporting Outputs

Every major run should record:
- selected config artifact
- confirm metrics artifact
- run config artifact
- short written summary in docs

For final comparison docs, include:
- bundle name
- hyperparameters
- `k` settings
- threshold
- confirm metrics
- selection rationale
- whether the point is production-pass, diagnostic-pass, or fallback

## 10) Historical Note

Earlier work used legacy guardrails such as:
- `accuracy >= 0.85`
- `closed_precision >= 0.30`

Those should now be treated as historical context only.
Current decisions and recommendations must use the dual-gate policy in this document.
