# Optuna Narrow HPO Design

## Purpose
Document the narrowed Optuna-based HPO pass that follows broad random-search passes.

Runner:
- `src/models_v2/run_hpo_optuna_narrow.py`

## Why This Pass Exists
- Broad random search already mapped the global landscape and identified the frontier.
- Frontier configs are concentrated in:
  - `rf` (single-stage)
  - `lr` (two-stage), with `lr` (single-stage) retained as LR baseline/control.
- A narrowed pass increases sample efficiency by focusing compute in promising regions.

## Input Evidence Used To Narrow
Narrowing decisions were based on:
- `artifacts/hpo_weighted_pass1/hpo_confirm_metrics.csv`
- `artifacts/hpo_weighted_pass1/bootstrap_confirm/bootstrap_confirm_metrics.csv`
- `artifacts/hpo_weighted_pass1/bootstrap_confirm/bootstrap_selected_candidates.csv`

Observed frontier:
- `rf single` and `lr two-stage` were strongest on closed F1.
- `lr single` remained useful for precision-oriented tradeoff checks and as stage-2 control baseline.

## Narrowed Search Spaces (And Rationale)

### LR (`single`, `two-stage`)
- `C`: `0.005` to `0.5` (log)
  - Focuses on the low-`C` region where best weighted-pass candidates clustered.
- `max_iter`: `{1000, 1500, 2000, 3000}`
  - Keeps convergence robustness without expanding cost.
- `class_weight`:
  - `single`: `{None, balanced, custom}`
  - `two-stage`: `{balanced, custom}` (drops `None`)
- `closed_class_weight` (custom): `2.0` to `6.0` (step `0.5`)
  - Centers around successful weighted-pass closed upweighting behavior.

### RF (`single`)
- `n_estimators`: `250` to `700` (step `25`)
  - Keeps enough ensemble size for stability while avoiding very large runs.
- `max_depth`: `{8, 10, 12, 14, 16, None}`
  - Focuses around depths seen in stronger trials.
- `min_samples_leaf`: `3` to `6`
  - Encourages smoother trees; aligns with pass-1 frontier behavior.
- `min_samples_split`: `3` to `12`
  - Moderate split conservatism range.
- `max_features`: `{log2, sqrt}`
  - Best-performing variants from prior pass.
- `class_weight`: `{balanced, custom}`
- `closed_class_weight` (custom): `2.0` to `6.0` (step `0.5`)

## Selection Policy In This Runner
This runner uses dual-gate selection by default:

- Production candidate (strict):
  - `accuracy >= 0.90`
  - `closed_precision >= 0.70`
  - `closed_recall >= 0.05`
  - Precision-first tie-break.

- Diagnostic candidate (iteration-focused):
  - `accuracy >= 0.84`
  - `closed_precision >= 0.20`
  - Rank by `closed_f1`, then `pr_auc_closed`.

Each selected candidate is confirmed with stronger CV (`5x3` by default).

## Objective During Search
- Optuna optimizes a diagnostic-oriented score:
  - primary: `closed_f1_mean`
  - small secondary weight: `pr_auc_closed_mean`
  - soft penalties for violating diagnostic floors
- This keeps search aligned with closed-performance improvement while discouraging degenerate regions.

## Reproducibility Notes
- Same fixed split policy (`train+val` only for HPO).
- Fixed `random_state` defaults and deterministic sampler seeds per model/mode.
- Optuna does not guarantee identical results to random search; it is intentionally adaptive.

## Outputs
- `hpo_search_trials.csv`
- `hpo_selected_trials_dualgate.csv`
- `hpo_confirm_metrics_dualgate.csv`
- `hpo_run_config_dualgate.json`

## Recommended Run Command
```bash
python src/models_v2/run_hpo_optuna_narrow.py \
  --models lr rf \
  --modes single two-stage \
  --feature-bundle low_plus_medium \
  --n-trials 40 \
  --search-n-splits 5 \
  --search-n-repeats 1 \
  --confirm-n-splits 5 \
  --confirm-n-repeats 3 \
  --decision-threshold 0.5 \
  --output-dir artifacts/hpo_optuna_narrow_pass1
```

## How To Decide If Narrowing Worked
- Compare against weighted pass 1 diagnostic frontier:
  - target improvement in confirmed `closed_f1_mean` and/or `closed_precision_mean`.
- If gain is marginal across one additional narrowed pass, freeze params and move to threshold sweep.
