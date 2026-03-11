# HPO Results Summary (Current Pass)

Note: This baseline run used the legacy gate policy (`accuracy>=0.85`, `closed_precision>=0.30`).
Current protocol now tracks dual gates (production + diagnostic); see `docs/eval_protocol.md`.

Date run: 2026-03-05  
Runner: `src/models_v2/run_hpo_experiments.py`  
Feature bundle: `low_plus_medium`  
Search budget: `40 trials/model`  
Search CV: `5x1`  
Confirm CV: `5x3`  
Decision threshold: `0.5`

## Artifact Sources

- `artifacts/hpo/lr_hpo/hpo_confirm_metrics.csv`
- `artifacts/hpo/lightgbm_hpo/hpo_confirm_metrics.csv`
- `artifacts/hpo/rf_hpo/hpo_confirm_metrics.csv`
- `artifacts/hpo/xgboost_hpo/hpo_confirm_metrics.csv`

## Confirmed Results (Sorted by Closed F1)

| Model | Mode | Accuracy Mean | Closed Precision Mean | Closed Recall Mean | Closed F1 Mean | PR-AUC Closed Mean |
|---|---|---:|---:|---:|---:|---:|
| LR | two-stage | 0.749 | 0.168 | 0.428 | 0.239 | 0.188 |
| LR | single | 0.695 | 0.155 | 0.525 | 0.239 | 0.139 |
| RF | single | 0.857 | 0.236 | 0.237 | 0.232 | 0.189 |
| LightGBM | single | 0.812 | 0.089 | 0.140 | 0.095 | 0.145 |
| LightGBM | two-stage | 0.833 | 0.089 | 0.115 | 0.085 | 0.145 |
| XGBoost | single | 0.883 | 0.113 | 0.040 | 0.059 | 0.144 |
| XGBoost | two-stage | 0.890 | 0.088 | 0.022 | 0.035 | 0.105 |

## First-Pass Hyperparameters Searched (Baseline Runner)

The first HPO pass used random search over the following parameters:

- LR:
  - `C` (log-uniform)
  - `max_iter`
  - `class_weight="balanced"` (fixed in first pass)
- LightGBM:
  - `n_estimators`
  - `learning_rate`
  - `num_leaves`
  - `subsample`
  - `colsample_bytree`
  - `min_child_samples`
  - `reg_lambda`
  - `class_weight="balanced"` (fixed in first pass)
- RandomForest:
  - `n_estimators`
  - `max_depth`
  - `min_samples_leaf`
  - `min_samples_split`
  - `max_features`
  - `class_weight="balanced"` (fixed in first pass)
- XGBoost:
  - `n_estimators`
  - `learning_rate`
  - `max_depth`
  - `subsample`
  - `colsample_bytree`
  - `min_child_weight`
  - `reg_lambda` (no explicit imbalance knob in first pass)

Exact sampled values for each trial are recorded in:
- `artifacts/hpo/*_hpo/hpo_search_trials.csv` (`params_json`)

Script used for this first pass:
- `src/models_v2/run_hpo_experiments.py` (baseline/frozen)

## Gate Status (Current Official Floors)

Current policy floors:
- `accuracy >= 0.85`
- `closed_precision >= 0.30`

Outcome:
- Gate pass count: `0 / 7`
- No model-mode config passed both floors in this HPO pass.
- Selection in each per-model run therefore used fallback policy (`max closed_f1` with shortfall documentation).

## Interpretation

- Best closed-class thresholded performance remains LR (`closed_f1 ~ 0.239`).
- RF is currently the strongest non-LR compromise (`accuracy ~ 0.857`, `closed_precision ~ 0.236`, `closed_f1 ~ 0.232`).
- XGBoost and LightGBM maintain higher accuracy but weak closed recall/F1 at threshold `0.5`.
- The main blocker for gate passing is closed precision floor (`0.30`), not just model family choice.

## Recommended Next Steps (Agreed Plan)

1. Run a second HPO pass with class-imbalance weighting as an explicit tunable knob.
2. Compare run 1 vs run 2 and identify promising parameter regions.
3. Run one final narrowed HPO pass (focused ranges) for this bundle.
4. Freeze shortlisted configs for this bundle.
5. Run decision-threshold sweeps on shortlisted configs.
6. Then proceed to bundle-iteration experiments (repeat HPO lightly for new bundles).

Notes:
- Threshold sweeps happen after the focused HPO pass, not before.
- Official gate status should still be reported using the protocol floors.

## Position in Workflow

This HPO pass happened **before** decision-threshold sweeps, consistent with protocol:
1. model/feature/gate framework fixed
2. per-model HPO
3. threshold sweeps on shortlisted configs

## Current Status Snapshot

- Cross-model HPO pass completed for:
  - `lr` (`single`, `two-stage`)
  - `lightgbm` (`single`, `two-stage`)
  - `rf` (`single`)
  - `xgboost` (`single`, `two-stage`)
- Gate pass count remains `0/7` under current official floors.
- Best confirmed closed-F1 remains LR (`~0.239`), with RF close (`~0.232`).

---

# HPO Results Summary (Weighted Pass 1)

Date run: 2026-03-06  
Runner: `src/models_v2/run_hpo_experiments_weighted.py`  
Feature bundle: `low_plus_medium`  
Search budget: `40 trials/model`  
Search CV: `5x1`  
Confirm CV: `5x3`  
Decision threshold: `0.5`

## Artifact Sources

- `artifacts/hpo_weighted_pass1/hpo_search_trials.csv`
- `artifacts/hpo_weighted_pass1/hpo_selected_trials.csv`
- `artifacts/hpo_weighted_pass1/hpo_confirm_metrics.csv`
- `artifacts/hpo_weighted_pass1/hpo_run_config.json`

## What Changed vs Baseline Pass

Weighted pass 1 introduces explicit class-imbalance tuning knobs:

- LR / RF:
  - `class_weight ∈ {None, "balanced", custom {0: w_closed, 1: 1.0}}`
- LightGBM:
  - `class_weight ∈ {None, "balanced"}`
  - optional `scale_pos_weight`
- XGBoost:
  - `scale_pos_weight`

## Confirmed Results (Sorted by Closed F1)

| Model | Mode | Accuracy Mean | Closed Precision Mean | Closed Recall Mean | Closed F1 Mean | PR-AUC Closed Mean |
|---|---|---:|---:|---:|---:|---:|
| RF | single | 0.865 | 0.247 | 0.219 | 0.230 | 0.179 |
| LR | two-stage | 0.862 | 0.241 | 0.225 | 0.229 | 0.185 |
| LightGBM | single | 0.859 | 0.115 | 0.097 | 0.098 | 0.145 |
| LightGBM | two-stage | 0.834 | 0.086 | 0.115 | 0.084 | 0.154 |
| LR | single | 0.900 | 0.347 | 0.033 | 0.057 | 0.160 |
| XGBoost | single | 0.883 | 0.106 | 0.038 | 0.055 | 0.134 |
| XGBoost | two-stage | 0.892 | 0.098 | 0.022 | 0.036 | 0.098 |

## Gate Status (Current Official Floors)

Current policy floors:
- `accuracy >= 0.85`
- `closed_precision >= 0.30`

Weighted pass 1 confirm outcome:
- Gate pass count: `1 / 7` (LR single)

Important caveat:
- The gate-pass config (`LR single`) achieves this mostly via high open-heavy behavior:
  - closed recall is very low (`~0.033`)
  - closed F1 is low (`~0.057`)
- So this is a technical gate pass under current floors, but not yet a strong closed-detection operating point.

## Gate-Policy Update (After This Run)

To better separate deployability from research ranking, evaluation policy is now documented as dual-gate:

- Production gates (strict, precision-first):
  - `closed_precision >= 0.70`
  - `accuracy >= 0.90`
  - `closed_recall >= 0.05`
- Diagnostic gates (iteration-focused):
  - `closed_precision >= 0.20`
  - `accuracy >= 0.84`
  - rank by `closed_f1`, then `pr_auc_closed`

This weighted pass should be interpreted as pre-update evidence under legacy floors, with dual-gate policy applied in subsequent runs.

## Interpretation

- Weighted tuning improved the frontier and produced two strong closed-oriented candidates:
  - `RF single` and `LR two-stage` both around `closed_f1 ~ 0.23` with `closed_precision ~ 0.24`.
- `LR single` can pass current floors, but with poor closed recall/F1 (not a preferred closed-performance candidate).
- Two-stage remains viable:
  - `LR two-stage` is competitive with RF on closed F1 and PR-AUC(closed), suggesting stage-1 filtering is directionally useful.
- LightGBM/XGBoost remained weaker on thresholded closed metrics in this pass.

## Recommended Next Steps

1. Run one narrowed weighted HPO pass (focused around best RF single and LR two-stage regions).
2. Keep official policy floors unchanged for reporting consistency.
3. After narrowed HPO, run threshold sweeps on shortlisted configs.
4. Re-evaluate gate status and closed-performance tradeoffs.
5. TODO: Split gate definitions into `production` vs `diagnostic` in protocol + runner outputs so deployability decisions are separated from research ranking.

Implementation note:
- Narrow-pass design and rationale are documented in `docs/hpo_optuna_narrow_design.md`.

---

# HPO Results Summary (Optuna Narrow Pass 1)

Date run: 2026-03-06  
Runner: `src/models_v2/run_hpo_optuna_narrow.py`  
Feature bundle: `low_plus_medium`  
Models tuned: `lr` (`single`, `two-stage`), `rf` (`single`)  
Search budget: `40 trials/model-mode`  
Search CV: `5x1`  
Confirm CV: `5x3`  
Decision threshold: `0.5`

## Artifact Sources

- `artifacts/hpo_optuna_narrow_pass1/hpo_search_trials.csv`
- `artifacts/hpo_optuna_narrow_pass1/hpo_selected_trials_dualgate.csv`
- `artifacts/hpo_optuna_narrow_pass1/hpo_confirm_metrics_dualgate.csv`
- `artifacts/hpo_optuna_narrow_pass1/hpo_run_config_dualgate.json`

## Confirmed Results (Sorted by Closed F1)

| Model | Mode | Gate Type | Accuracy Mean | Closed Precision Mean | Closed Recall Mean | Closed F1 Mean | PR-AUC Closed Mean |
|---|---|---|---:|---:|---:|---:|---:|
| LR | two-stage | diagnostic | 0.840 | 0.225 | 0.290 | 0.251 | 0.191 |
| LR | single | diagnostic | 0.852 | 0.229 | 0.242 | 0.233 | 0.194 |
| RF | single | diagnostic | 0.862 | 0.238 | 0.219 | 0.225 | 0.179 |
| LR | single | production | 0.894 | 0.309 | 0.108 | 0.154 | 0.187 |
| RF | single | production | 0.899 | 0.274 | 0.071 | 0.111 | 0.196 |
| LR | two-stage | production | 0.898 | 0.211 | 0.027 | 0.041 | 0.169 |

## Gate Status (Dual-Gate Policy)

- Production gate pass count: `0 / 3`
- Diagnostic gate pass count: `2 / 3`
  - passes: `LR single`, `RF single`
  - near-miss: `LR two-stage` diagnostic (`accuracy ~ 0.8399`)

## Interpretation

- Optuna narrowing improved the LR frontier:
  - `LR two-stage` diagnostic closed F1 improved to `~0.251`
  - `LR single` diagnostic closed F1 improved to `~0.233`
- RF remained competitive but was no longer the top closed-F1 model in this pass.
- No configuration passed strict production gates; this remains a research/iteration stage.

## Current Plan

1. Run LR-only micro Optuna pass to refine around new LR winners:
   - `src/models_v2/run_hpo_optuna_lr_micro.py`
2. If confirm improvement is marginal (for example `< ~0.01` closed-F1 gain), freeze LR hyperparameters.
3. Run decision-threshold sweeps on frozen shortlisted configs.
4. Re-assess production-gate feasibility and decide whether feature-bundle v2 is needed.

## Next-Phase Plan (Ceiling -> Transfer -> Labeling)

1. Establish practical model ceiling for the current frontier:
   - freeze shortlisted configs (`lr two-stage`, `lr single`, `rf single`) after final micro-tuning,
   - run threshold sweeps and keep best diagnostic operating points.
2. Evaluate transfer across datasets:
   - apply frozen pipeline/configs across broad vs concentrated city datasets,
   - compare broad-trained transfer behavior against city-trained behavior.
3. Use transfer findings to diagnose bottleneck:
   - if closed performance remains constrained across these checks, treat label coverage/distribution as the primary bottleneck.
4. Promote labeling pipeline implementation as the next major workstream:
   - hybrid auto-label + targeted review loop to expand reliable labels and improve closed-class performance.

---

# HPO Results Summary (Optuna LR Micro Pass 1)

Date run: 2026-03-08  
Runner: `src/models_v2/run_hpo_optuna_lr_micro.py`  
Feature bundle: `low_plus_medium`  
Models tuned: `lr` (`single`, `two-stage`)  
Search budget: `40 trials/model-mode`  
Search CV: `5x1`  
Confirm CV: `5x3`  
Decision threshold: `0.5`

## Artifact Sources

- `artifacts/hpo_optuna_lr_micro_pass1/hpo_search_trials.csv`
- `artifacts/hpo_optuna_lr_micro_pass1/hpo_selected_trials_dualgate.csv`
- `artifacts/hpo_optuna_lr_micro_pass1/hpo_confirm_metrics_dualgate.csv`
- `artifacts/hpo_optuna_lr_micro_pass1/hpo_run_config_dualgate.json`

## Confirmed Results (Sorted by Closed F1)

| Model | Mode | Gate Type | Accuracy Mean | Closed Precision Mean | Closed Recall Mean | Closed F1 Mean | PR-AUC Closed Mean |
|---|---|---|---:|---:|---:|---:|---:|
| LR | two-stage | diagnostic | 0.860 | 0.251 | 0.257 | 0.251 | 0.190 |
| LR | single | diagnostic | 0.844 | 0.224 | 0.274 | 0.245 | 0.194 |
| LR | two-stage | production | 0.872 | 0.261 | 0.192 | 0.215 | 0.191 |
| LR | single | production | 0.887 | 0.273 | 0.137 | 0.179 | 0.190 |

## Comparison vs Optuna Narrow Pass 1 (Confirm CV Deltas)

`delta = lr_micro - optuna_narrow`

| Model | Mode | Gate Type | Delta Accuracy | Delta Closed Precision | Delta Closed Recall | Delta Closed F1 | Delta PR-AUC Closed |
|---|---|---|---:|---:|---:|---:|---:|
| LR | single | diagnostic | -0.008 | -0.005 | +0.032 | +0.012 | -0.000 |
| LR | two-stage | diagnostic | +0.020 | +0.026 | -0.033 | +0.001 | -0.001 |
| LR | single | production | -0.008 | -0.036 | +0.030 | +0.025 | +0.003 |
| LR | two-stage | production | -0.026 | +0.050 | +0.164 | +0.175 | +0.022 |

## Freeze Decision

- Diagnostic winner did not move materially at the top (`LR two-stage diagnostic` closed-F1: `0.2507 -> 0.2513`, +`0.0005`).
- LR micro improved secondary operating points, but this is now in marginal-gain territory for additional LR HPO.
- Decision: freeze LR hyperparameters and move to featurizer-`k` and threshold optimization.

Frozen shortlisted configs from this pass:
- `lr single diagnostic`: `C=0.031066`, `class_weight={0:4.5,1:1.0}`, `max_iter=2000`
- `lr two-stage diagnostic`: `C=0.018049`, `class_weight={0:5.0,1:1.0}`, `max_iter=1000`
- `lr single production`: `C=0.018703`, `class_weight={0:3.5,1:1.0}`, `max_iter=1000`
- `lr two-stage production`: `C=0.032306`, `class_weight={0:3.5,1:1.0}`, `max_iter=1000`

## Next Steps (Post-Freeze)

1. Tune shared-featurizer `k` controls under frozen LR params:
   - `category_top_k`, `dataset_top_k`, `cluster_top_k`
   - keep CV protocol fixed (`5x3` confirm-style) and evaluate with dual-gate reporting.
2. For top `k` settings, run threshold sweeps (`p_open` threshold grid) per frozen config.
3. Select final diagnostic and production operating points after thresholding.
4. Re-check production gate feasibility; if still not feasible, document shortfall and proceed to feature-bundle v2 / labeling pipeline workstream.

---

# HPO Results Summary (Optuna RF Micro Pass 1)

Date run: 2026-03-08  
Runner: `src/models_v2/run_hpo_optuna_rf_micro.py`  
Feature bundle: `low_plus_medium`  
Models tuned: `rf` (`single`)  
Search budget: `40 trials/model-mode`  
Search CV: `5x1`  
Confirm CV: `5x3`  
Decision threshold: `0.5`

## Artifact Sources

- `artifacts/hpo_optuna_rf_micro_pass1/hpo_search_trials.csv`
- `artifacts/hpo_optuna_rf_micro_pass1/hpo_selected_trials_dualgate.csv`
- `artifacts/hpo_optuna_rf_micro_pass1/hpo_confirm_metrics_dualgate.csv`
- `artifacts/hpo_optuna_rf_micro_pass1/hpo_run_config_dualgate.json`

## Confirmed Results

| Model | Mode | Gate Type | Accuracy Mean | Closed Precision Mean | Closed Recall Mean | Closed F1 Mean | PR-AUC Closed Mean |
|---|---|---|---:|---:|---:|---:|---:|
| RF | single | diagnostic | 0.857 | 0.237 | 0.234 | 0.231 | 0.182 |
| RF | single | production | 0.901 | 0.272 | 0.054 | 0.090 | 0.202 |

## Comparison vs Optuna Narrow Pass 1 (Confirm CV Deltas)

`delta = rf_micro - optuna_narrow`

| Model | Mode | Gate Type | Delta Accuracy | Delta Closed Precision | Delta Closed Recall | Delta Closed F1 | Delta PR-AUC Closed |
|---|---|---|---:|---:|---:|---:|---:|
| RF | single | diagnostic | -0.005 | -0.001 | +0.015 | +0.006 | +0.002 |
| RF | single | production | +0.003 | -0.002 | -0.017 | -0.022 | +0.006 |

## Freeze Decision

- RF micro gave a modest diagnostic improvement (`closed_f1 +0.0056`) with slightly higher recall and similar precision.
- Production-oriented RF point improved accuracy/PR-AUC but regressed on closed recall/F1.
- Decision: freeze RF hyperparameters from this micro pass and proceed to the same phased `k`-then-threshold sweep workflow used for LR.

Frozen shortlisted configs from this pass:
- `rf single diagnostic`: `n_estimators=375`, `max_depth=8`, `min_samples_leaf=7`, `min_samples_split=6`, `max_features=log2`, `class_weight=balanced`
- `rf single production`: `n_estimators=525`, `max_depth=8`, `min_samples_leaf=5`, `min_samples_split=11`, `max_features=log2`, `class_weight={0:4.0,1:1.0}`

## Next Steps (RF Post-Freeze)

1. Run phased `k` sweep (`k_coarse -> k_narrow`) on frozen RF config(s) at fixed threshold (`0.5`).
2. Run threshold sweep on shortlisted RF `k` candidates.
3. Compare finalized RF operating points against finalized LR operating points before final recommendation.

---

# K/Threshold Phased Sweep + Confirm (LR + RF)

Date run: 2026-03-09  
Runner: `src/models_v2/run_k_threshold_sweep.py`  
Phases executed: `k_coarse -> k_narrow -> threshold -> confirm`  
Confirm CV: `5x10` (`random_state=142`)

## Artifact Sources

- `artifacts/k_threshold_sweep_lr_pass1/k_coarse_metrics.csv`
- `artifacts/k_threshold_sweep_lr_pass1/k_narrow_metrics.csv`
- `artifacts/k_threshold_sweep_lr_pass1/threshold_final_best.csv`
- `artifacts/k_threshold_sweep_lr_pass1/threshold_confirm_metrics.csv`
- `artifacts/k_threshold_sweep_rf_pass1/k_coarse_metrics.csv`
- `artifacts/k_threshold_sweep_rf_pass1/k_narrow_metrics.csv`
- `artifacts/k_threshold_sweep_rf_pass1/threshold_final_best.csv`
- `artifacts/k_threshold_sweep_rf_pass1/threshold_confirm_metrics.csv`

## Confirmed Operating Points (Sorted by Closed F1)

| Model | Mode | Gate Type | k (cat,ds,cl) | Threshold | Accuracy Mean | Closed Precision Mean | Closed Recall Mean | Closed F1 Mean | PR-AUC Closed Mean |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| LR | two-stage | diagnostic | (50,5,45) | 0.51 | 0.860 | 0.251 | 0.260 | 0.254 | 0.205 |
| RF | single | diagnostic | (50,5,70) | 0.50 | 0.846 | 0.230 | 0.283 | 0.251 | 0.203 |
| LR | single | diagnostic | (25,5,60) | 0.50 | 0.841 | 0.218 | 0.268 | 0.238 | 0.198 |
| LR | single | production | (45,5,35) | 0.41 | 0.903 | 0.328 | 0.045 | 0.076 | 0.195 |
| RF | single | production | (50,5,70) | 0.48 | 0.907 | 0.388 | 0.033 | 0.060 | 0.215 |
| LR | two-stage | production | (55,5,10) | 0.33 | 0.909 | 0.450 | 0.024 | 0.045 | 0.212 |

## Gate Status (After Confirm)

- Production gate pass count: `0 / 6`
- Diagnostic gate pass count: `6 / 6`

## Interpretation

- Diagnostic frontier is now near-tied between:
  - `LR two-stage` (`closed_f1 ~ 0.2539`)
  - `RF single` (`closed_f1 ~ 0.2514`)
- LR remains the top diagnostic point by a small margin (`~0.0025` closed-F1), while RF diagnostic keeps slightly higher closed recall.
- Production-gate feasibility remains the key blocker:
  - no confirmed point reaches strict production floors (`closed_precision >= 0.70`, `accuracy >= 0.90`, `closed_recall >= 0.05`).
- Current recommendation:
  - keep `LR two-stage` as primary diagnostic winner,
  - keep `RF single` as co-frontier/fallback for recall-sensitive exploration,
  - treat this stage as an iteration milestone, not production-ready.

## Frozen LR Reference Configs

These are the current LR reference operating points after LR micro-HPO plus phased `k`/threshold confirm. Treat them as the frozen LR baseline unless a `v2` LR bundle beats them under the same workflow.

| Mode | Gate Type | Selected Trial | Feature Bundle | Hyperparameters | k (cat,ds,cl) | Threshold | Accuracy Mean | Closed Precision Mean | Closed Recall Mean | Closed F1 Mean | PR-AUC Closed Mean |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| single | production | 14 | `low_plus_medium` | `C=0.018702742958488373`, `class_weight={0:3.5,1:1.0}`, `max_iter=1000`, `solver=lbfgs` | `(45,5,35)` | 0.41 | 0.903 | 0.328 | 0.045 | 0.076 | 0.195 |
| single | diagnostic | 38 | `low_plus_medium` | `C=0.031066049133505004`, `class_weight={0:4.5,1:1.0}`, `max_iter=2000`, `solver=lbfgs` | `(25,5,60)` | 0.50 | 0.841 | 0.218 | 0.268 | 0.238 | 0.198 |
| two-stage | production | 16 | `low_plus_medium` | `C=0.03230600504184561`, `class_weight={0:3.5,1:1.0}`, `max_iter=1000`, `solver=lbfgs` | `(55,5,10)` | 0.33 | 0.909 | 0.450 | 0.024 | 0.045 | 0.212 |
| two-stage | diagnostic | 19 | `low_plus_medium` | `C=0.018049168526244785`, `class_weight={0:5.0,1:1.0}`, `max_iter=1000`, `solver=lbfgs` | `(50,5,45)` | 0.51 | 0.860 | 0.251 | 0.260 | 0.254 | 0.205 |

Notes:
- Hyperparameters come from `artifacts/hpo_optuna_lr_micro_pass1/hpo_selected_trials_dualgate.csv`.
- Final `k` values and thresholds come from `artifacts/k_threshold_sweep_lr_pass1/threshold_confirm_metrics.csv`.
- RF final bundle selection is documented below in the follow-up confirm section.

---

# RF V2 Bundle Follow-Up Confirm (No Spatial Prior)

Date run: 2026-03-10  
Runner: `src/models_v2/run_k_threshold_sweep.py`  
Bundle under test: `v2_rf_single_no_spatial_prior`  
Phases executed: `k_coarse -> k_narrow -> threshold -> confirm`  
Confirm CV: `5x10` (`random_state=142`)

## Artifact Sources

- `artifacts/k_threshold_sweep_rf_no_spatial_prior_pass1/k_coarse_metrics.csv`
- `artifacts/k_threshold_sweep_rf_no_spatial_prior_pass1/k_narrow_metrics.csv`
- `artifacts/k_threshold_sweep_rf_no_spatial_prior_pass1/threshold_final_best.csv`
- `artifacts/k_threshold_sweep_rf_no_spatial_prior_pass1/threshold_confirm_metrics.csv`

## Confirmed RF Winner

| Model | Mode | Gate Type | Feature Bundle | Trial | k (cat,ds,cl) | Threshold | Accuracy Mean | Closed Precision Mean | Closed Recall Mean | Closed F1 Mean | PR-AUC Closed Mean |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| RF | single | diagnostic | `v2_rf_single_no_spatial_prior` | 35 | (25,5,45) | 0.39 | 0.886 | 0.343 | 0.263 | 0.297 | 0.262 |

Frozen RF diagnostic config:
- hyperparameters: `n_estimators=450`, `max_depth=16`, `min_samples_leaf=6`, `min_samples_split=12`, `max_features=log2`, `class_weight=balanced`
- feature bundle: `v2_rf_single_no_spatial_prior`
- `k=(25,5,45)`
- threshold: `0.39`

## Comparison vs Previous Confirmed Leaders

Versus prior confirmed RF diagnostic leader (`low_plus_medium`):
- accuracy: `+0.0398`
- closed precision: `+0.1135`
- closed recall: `-0.0198`
- closed F1: `+0.0456`
- PR-AUC closed: `+0.0588`

Versus prior confirmed overall diagnostic leader (`LR two-stage`, `low_plus_medium`):
- accuracy: `+0.0260`
- closed precision: `+0.0922`
- closed recall: `+0.0035`
- closed F1: `+0.0431`
- PR-AUC closed: `+0.0574`

## Freeze Decision

- `v2_rf_single_no_spatial_prior` is the new confirmed overall diagnostic leader.
- Improvement is material rather than marginal, so further manual bundle refinement is not required for this round.
- Decision: freeze `v2_rf_single_no_spatial_prior` as the RF v2 bundle winner.

## Frozen RF Reference Config

| Mode | Gate Type | Selected Trial | Feature Bundle | Hyperparameters | k (cat,ds,cl) | Threshold | Accuracy Mean | Closed Precision Mean | Closed Recall Mean | Closed F1 Mean | PR-AUC Closed Mean |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| single | diagnostic | 35 | `v2_rf_single_no_spatial_prior` | `n_estimators=450`, `max_depth=16`, `min_samples_leaf=6`, `min_samples_split=12`, `max_features=log2`, `class_weight=balanced` | `(25,5,45)` | 0.39 | 0.886 | 0.343 | 0.263 | 0.297 | 0.262 |

## Next Bundle Iteration Note

- Before creating model-specific v2 bundles, follow:
  - `docs/feature_bundle_v2_conventions.md`
- This defines the feature-selection rules, ablation/evaluation workflow, and stop criteria for the next bundle-optimization round.
