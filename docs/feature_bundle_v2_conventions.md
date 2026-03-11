# Feature Bundle V2 Conventions

## Purpose
Define a consistent, defensible process for building model-specific v2 feature bundles after the current `low_plus_medium` ceiling study.

Scope for this round:
- `v2_lr2` (for LR two-stage)
- `v2_rf_single` (for RF single)
- `v2_rf_single_no_spatial_prior` (RF single follow-up ablation candidate)

## Why These Conventions
- Prevent ad hoc feature cherry-picking.
- Keep per-model ceiling optimization rigorous and comparable.
- Ensure bundle updates are interpretable and reproducible.

## Inputs (Evidence Sources)
Use only these evidence types to propose add/remove decisions:
1. Confirmed performance artifacts from the current ceiling run (`k/threshold/confirm`).
2. Feature-importance outputs in `artifacts/feature_importance/`.
3. Existing feature rationale and policy docs:
   - `docs/feature_rationale.md`
   - `docs/feature_bundle_v2_rationale.md`
   - `docs/eval_protocol.md`
   - `docs/feature_inventory.csv`

## Bundle Construction Rules
1. Model-specific optimization is allowed.
2. Keep policy-compliant schema-native features only.
3. Preserve core anchor features unless evidence clearly supports removal:
   - source coverage/diversity
   - contact/completeness basics
   - key temporal freshness signals
4. Prefer group-level pruning before one-off feature deletion:
   - category OHE group
   - source-dataset OHE group
   - geo-cluster OHE group
   - interaction feature group
5. Avoid pure "top-N importance only" bundle design.
   - Importance is a prior, not a final decision rule.
6. Every inclusion/removal must have a one-line rationale.

## Fairness/Comparability Rules
Hold these fixed across `v2_lr2`, `v2_rf_single`, and `v2_rf_single_no_spatial_prior` evaluations:
- same train/val data policy
- same gate definitions
- same CV protocol for each phase
- same phased runner flow (`k_coarse -> k_narrow -> threshold -> confirm`)
- same reporting metrics and ranking rules

## Evaluation Workflow
1. Draft candidate bundles (`v2_lr2`, `v2_rf_single`, optional `v2_rf_single_no_spatial_prior`).
2. Run quick ablation screen (coarse checks) to remove clearly weak groups/features.
   - Runner: `src/models_v2/run_feature_group_ablation_screen.py`
   - Pattern: baseline + minus-one-group (`spatial`, `category`, `source_provenance`, `temporal`, `completeness_contact`, `interactions`)
   - Use fixed params, fixed `k`, fixed threshold (no full `k/threshold` sweep at this stage).
3. Freeze candidate bundles for full phased sweep.
4. Run phased tuning:
   - `k_coarse`
   - `k_narrow`
   - `threshold`
   - `confirm` (strong CV)
5. Compare against current `low_plus_medium` ceilings.

## Decision Threshold for "Meaningful Improvement"
Treat improvements as meaningful only if they are both:
1. Directionally consistent on primary diagnostic metrics:
   - `closed_f1_mean`
   - `pr_auc_closed_mean`
2. Larger than normal run-to-run/fold variability signal.

If gains are small/mixed and no production-gate movement appears, stop model-side tuning and prioritize labeling pipeline work.

## Documentation Requirements
For each bundle candidate, record:
- feature list
- rationale summary
- run command and output directory
- final chosen `k` and threshold
- confirm metrics (`mean` and `std`)
- comparison delta vs current ceiling

## Exit Criteria for Bundle-V2 Round
Stop the v2 bundle round when one of these is true:
1. No material gain over current ceiling after confirm phase.
2. Gains are marginal and unstable.
3. Production-gate feasibility remains unchanged despite bundle optimization.

At stop, transition focus to data/label quality intervention (labeling pipeline proposal and implementation plan).

## Round Outcome

Current round outcome:
- `v2_lr2` is retained as the LR v2 reference bundle.
- `v2_rf_single_no_spatial_prior` won the RF follow-up confirm run and is frozen as the RF v2 bundle for this round.
- Further manual RF bundle refinement is out of scope unless a new high-signal ablation hypothesis emerges.
