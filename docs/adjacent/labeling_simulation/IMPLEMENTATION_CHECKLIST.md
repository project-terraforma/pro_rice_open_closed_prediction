# Implementation Checklist & Quick Start Guide

## Pre-Deployment Checklist ✅

### Phase 0: Preparation (Before Week 1)

- [ ] **Stakeholder Alignment**
  - [ ] Review PROJECT_SUMMARY.md with leadership
  - [ ] Approve production rollout plan
  - [ ] Designate labeling team (2-3 reviewers minimum)
  - [ ] Set up Slack channel for alerts/updates

- [ ] **Infrastructure Setup**
  - [ ] Create labeling queue system (database table or queue service)
  - [ ] Set up monitoring dashboard (Grafana/Datadog/custom)
  - [ ] Configure automated batch extraction job
  - [ ] Set up model storage (versioning, rollback capability)

- [ ] **Code Preparation**
  - [ ] Clone/copy all model files to production environment
  - [ ] Run validation_suite.py (should see 28/28 tests pass)
  - [ ] Create deployment checklist (this document)
  - [ ] Document model API for inference

- [ ] **Data Preparation**
  - [ ] Freeze test set (never touch during labeling loop)
  - [ ] Create backup of historical labels (if any)
  - [ ] Set up data validation checks (schema, nulls, distributions)
  - [ ] Document feature requirements and transformations

- [ ] **Knowledge Transfer**
  - [ ] Train labeling team on triage categories
  - [ ] Create simple labeling guidelines (closed vs open decision rules)
  - [ ] Set up daily/weekly sync meetings
  - [ ] Document emergency contacts & escalation process

### Phase 1: Pilot Deployment (Week 1-2)

- [ ] **Day 1: Setup & First Batch**
  - [ ] Deploy model code to production environment
  - [ ] Extract first batch of ~400 new records
  - [ ] Run Stage 1 rule filter (should auto-accept ~20%)
  - [ ] Generate review queue (~20 manual reviews, 5% budget)
  - [ ] Notify labeling team to start reviewing

- [ ] **Daily (Week 1): Monitoring**
  - [ ] Track review queue: are reviewers keeping up?
  - [ ] Spot-check 5-10 auto-labeled records (correctness)
  - [ ] Monitor for data quality issues (nulls, invalid fields)
  - [ ] Check error logs for exceptions

- [ ] **End of Batch 1 (Day 5-7)**
  - [ ] Collect all reviewed labels (~20 records)
  - [ ] Merge with auto-labeled (~200 records)
  - [ ] Compute auto-label precision on audit sample (target: >75%)
  - [ ] Retrain model on gold + weighted silver labels
  - [ ] Evaluate on frozen test set (log baseline PR-AUC)
  - [ ] Send metrics report to stakeholders

- [ ] **Batch 2 (Days 8-14)**
  - [ ] Extract next batch
  - [ ] Score with updated model
  - [ ] Triage and generate review queue
  - [ ] Repeat monitoring/collection/retraining cycle
  - [ ] Check for any precision degradation
  - [ ] Solicit reviewer feedback (hard cases? confusion areas?)

- [ ] **Go/No-Go Decision (End Week 2)**
  - [ ] Is auto-label precision > 75%? ✅ = continue
  - [ ] Is review time < 30 min per batch? ✅ = continue
  - [ ] Has reviewer adoption > 80%? ✅ = continue
  - [ ] Any critical issues found? ❌ = fix before expanding

### Phase 2: Scale & Optimize (Week 3-4)

- [ ] **Expand to Additional Datasets**
  - [ ] Repeat pilot process on 2-3 new datasets
  - [ ] Adapt threshold per dataset's closed rate (calibrate)
  - [ ] Monitor for any dataset-specific issues

- [ ] **Automate Batch Processing**
  - [ ] Create nightly job:
    - Extract new records from source table
    - Apply Stage 1 rule filter
    - Score with latest model
    - Generate review queue
    - Send Slack notification to labeling team
  - [ ] Create weekly job:
    - Collect ground truth for reviewed/auto-labeled
    - Retrain model
    - Evaluate test set
    - Log metrics to monitoring dashboard
    - Alert if PR-AUC drops >10%

- [ ] **Threshold Tuning**
  - [ ] For each dataset, compute optimal thresholds:
    - If closed rate < 5%: use 0.90 threshold (more conservative)
    - If closed rate > 20%: use 0.80 threshold (more aggressive)
  - [ ] Test for 1-2 weeks, measure impact
  - [ ] Document final thresholds per dataset

- [ ] **Optimization Decisions**
  - [ ] Evaluate Random vs Uncertainty strategy
    - Run A/B test if possible
    - Switch if Uncertainty shows >15% improvement
  - [ ] Collect reviewer efficiency metrics:
    - Time per review
    - Consensus (agreement between reviewers)
    - Error rate on agreed hard cases

### Phase 3: Production Operations (Month 2+)

- [ ] **Weekly Operations**
  - [ ] Monitor automated batch jobs (did they run?)
  - [ ] Check Slack alerts for any issues
  - [ ] Review metrics dashboard (PR-AUC trend, precision)
  - [ ] Spot-check 5-10 predictions for correctness

- [ ] **Monthly Reviews**
  - [ ] Aggregate metrics across all datasets
  - [ ] Measure cost savings (labels created vs manual effort)
  - [ ] Identify any drift or quality issues
  - [ ] Plan next month's improvements

- [ ] **Quarterly Recalibration**
  - [ ] Retrain model from scratch (fresh start)
  - [ ] Freeze all labeled data (never retrain with test data)
  - [ ] Evaluate on new test set (measure generalization)
  - [ ] Update thresholds if distributions changed
  - [ ] Document any learnings for next quarter

- [ ] **Model Upgrades** (Plan for Month 3+)
  - [ ] After 1,000+ labeled examples per dataset:
    - Try XGBoost (expected +15% improvement)
    - A/B test on same batch
    - If wins, gradually roll out
  - [ ] After 3+ months of operation:
    - Try Uncertainty sampling instead of Random
    - Expected to reduce manual reviews by 20-30%
  - [ ] After 6+ months:
    - Consider ensemble methods (LR + XGB combination)
    - Explore feature engineering improvements

---

## Production Deployment Script

### Quick Start (Run These Commands)

```bash
# 1. Run validation tests
cd /Users/claricepark/Desktop/pro_rice_open_closed_prediction
python src/models_v2/validation_suite.py

# Expected output: "28/28 passed (100%)" ✅

# 2. Run quick comparison benchmark
python src/models_v2/run_quick_comparison.py

# Expected output: Comparison charts saved, summary printed

# 3. Check results
cat artifacts/quick_comparison/summary.csv
cat artifacts/quick_comparison/comparison_charts.png  # View in IDE
```

### Production Configuration Template

Create file: `production_config.yaml`

```yaml
# Label-Coverage Active-Labeling Loop Configuration

deployment:
  environment: production
  start_date: 2024-03-06
  pilot_dataset: restaurants_sf
  scale_timeline: week_3

model:
  type: logistic_regression  # Switch to xgboost at 1k+ labels
  mode: two_stage
  feature_bundle: low_plus_medium
  retrain_frequency: weekly
  save_checkpoints: true
  checkpoint_dir: /path/to/models/checkpoints

triage:
  auto_accept_closed_threshold: 0.85
  auto_accept_open_threshold: 0.15
  review_budget_pct: 0.05  # 5% per batch
  
  # Per-dataset overrides (optional)
  overrides:
    restaurants_sf:
      threshold_closed: 0.80  # Lower (more auto-labels)
    hotels_la:
      threshold_closed: 0.90  # Higher (more conservative)

training:
  gold_weight: 1.0
  silver_weight_formula: "0.4 + 0.4 * confidence"  # Callable in code
  sample_weight_enabled: true
  class_weight: balanced

monitoring:
  metrics_to_track:
    - test_closed_pr_auc
    - auto_label_precision
    - review_queue_length
    - model_inference_time
  alert_thresholds:
    pr_auc_drop_pct: 10  # Alert if drops >10%
    precision_drop_pct: 5  # Alert if drops >5%
    review_backlog_max: 100  # Alert if >100 pending
  dashboard_url: https://monitoring.example.com/label-loop

labeling:
  team_size: 2-3  # Recommended
  guidelines_url: https://docs.example.com/labeling-guide
  escalation_contact: data-science-lead@example.com
  daily_standup: 10:00 AM
  weekly_review: Friday 3:00 PM

evaluation:
  test_set_locked: true
  test_set_size: ~800
  test_set_refresh: quarterly
  holdout_strategy: never_relabel
```

### Batch Processing Job Template

Create file: `batch_processor.py` (nightly scheduled job)

```python
#!/usr/bin/env python3
"""Nightly batch processing job for active-labeling loop."""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    filename='/var/log/label_loop/batch_processor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_batch(source_table, batch_size=400):
    """Extract new records from source."""
    logging.info(f"Extracting batch (size={batch_size})...")
    
    # Query database for new records
    query = f"""
    SELECT id, name, open, sources, websites, ...
    FROM {source_table}
    WHERE created_date >= NOW() - INTERVAL 7 DAY
    AND id NOT IN (SELECT id FROM labeled_records)
    ORDER BY created_date DESC
    LIMIT {batch_size}
    """
    
    batch = pd.read_sql(query, db_connection)
    logging.info(f"Extracted {len(batch)} records")
    return batch

def score_batch(batch, model_path):
    """Score batch with latest model."""
    logging.info("Loading model...")
    model = load_model(model_path)
    
    logging.info("Scoring batch...")
    predictions = model.predict_proba(batch)
    batch['model_score'] = predictions[:, 1]  # P(open)
    batch['model_confidence'] = batch['model_score'].apply(lambda p: max(p, 1-p))
    
    logging.info("Scoring complete")
    return batch

def triage_batch(batch, config):
    """Triage batch into auto-accept/review/defer."""
    logging.info("Triaging batch...")
    
    policy = TriagePolicy(
        t_closed_high=config['triage']['auto_accept_closed_threshold'],
        t_open_low=config['triage']['auto_accept_open_threshold'],
    )
    
    p_closed = 1.0 - batch['model_score']
    routed = policy.route_batch(batch, p_closed)
    
    logging.info(f"Triage results:")
    logging.info(f"  Auto-accept: {(routed['route'] == 'auto_accept').sum()}")
    logging.info(f"  Review: {(routed['route'] == 'review').sum()}")
    logging.info(f"  Defer: {(routed['route'] == 'defer').sum()}")
    
    return routed

def generate_review_queue(routed_batch, budget_pct=0.05):
    """Generate review queue for labeling team."""
    logging.info("Generating review queue...")
    
    review_budget = max(1, int(len(routed_batch) * budget_pct))
    candidates = routed_batch[routed_batch['route'] == 'review'].copy()
    
    if len(candidates) > review_budget:
        # Select top uncertain ones
        candidates['score'] = candidates['uncertainty'] * candidates['impact_score']
        review_queue = candidates.nlargest(review_budget, 'score')
    else:
        review_queue = candidates
    
    logging.info(f"Review queue: {len(review_queue)} records")
    
    # Save to review table
    review_queue.to_sql('review_queue', db_connection, if_exists='append', index=False)
    
    # Send Slack notification
    send_slack_alert(f"Review queue ready: {len(review_queue)} records awaiting review")
    
    return review_queue

def main():
    """Main batch processing pipeline."""
    logging.info("="*80)
    logging.info("BATCH PROCESSING JOB STARTED")
    logging.info("="*80)
    
    try:
        # Load config
        config = load_config('production_config.yaml')
        
        # Extract batch
        batch = extract_batch(config['dataset']['source_table'])
        if len(batch) == 0:
            logging.warning("No new records to process")
            return
        
        # Score with latest model
        batch = score_batch(batch, config['model']['checkpoint_dir'])
        
        # Triage
        routed = triage_batch(batch, config)
        
        # Generate review queue
        generate_review_queue(routed, config['triage']['review_budget_pct'])
        
        # Save auto-labels
        auto_batch = routed[routed['route'] == 'auto_accept'].copy()
        if len(auto_batch) > 0:
            auto_batch.to_sql('auto_labels', db_connection, if_exists='append', index=False)
            logging.info(f"Saved {len(auto_batch)} auto-labels")
        
        logging.info("BATCH PROCESSING JOB COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        logging.error(f"BATCH PROCESSING JOB FAILED: {e}")
        send_slack_alert(f"❌ Batch processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
```

### Weekly Retraining Job Template

```python
#!/usr/bin/env python3
"""Weekly model retraining job."""

import pandas as pd
import logging
from datetime import datetime, timedelta

logging.basicConfig(...)

def get_labeled_records():
    """Get all reviewed & auto-labeled records since last training."""
    query = """
    SELECT id, label, source, confidence
    FROM labeled_records
    WHERE created_date > NOW() - INTERVAL 7 DAY
    AND labeled_date IS NOT NULL
    """
    return pd.read_sql(query, db_connection)

def retrain_model(labeled_data, config, prev_model_path):
    """Retrain model on accumulated labels."""
    
    # Merge with features
    features = get_features(labeled_data['id'].values)
    training_set = features.merge(labeled_data, on='id')
    
    # Create sample weights
    gold_mask = training_set['source'] == 'manual'
    training_set['weight'] = gold_mask.astype(float)  # 1.0 for gold
    training_set.loc[~gold_mask, 'weight'] = (
        0.4 + 0.4 * training_set.loc[~gold_mask, 'confidence']
    )  # 0.4-0.8 for silver
    
    # Train
    model = LogisticRegression(mode="two-stage", feature_bundle="low_plus_medium")
    model.fit(training_set, val_df=None)
    
    # Evaluate
    test_set = load_frozen_test_set()
    metrics = evaluate(model, test_set)
    
    # Check for improvement
    prev_metrics = load_metrics(prev_model_path)
    if metrics['pr_auc'] < prev_metrics['pr_auc'] * 0.9:
        logging.warning(f"Model degraded! PR-AUC: {metrics['pr_auc']} < {prev_metrics['pr_auc']}")
        send_slack_alert("⚠️  Model degraded! Keeping previous version.")
        return None
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"/models/checkpoints/lr_{timestamp}.pkl"
    save_model(model, model_path)
    
    # Log metrics
    log_metrics(metrics, timestamp)
    
    logging.info(f"Retraining complete. New PR-AUC: {metrics['pr_auc']:.4f}")
    send_slack_alert(f"✅ Model retrained. PR-AUC: {metrics['pr_auc']:.4f}")
    
    return model_path

def main():
    """Weekly retraining pipeline."""
    logging.info("="*80)
    logging.info("WEEKLY RETRAINING JOB STARTED")
    logging.info("="*80)
    
    try:
        config = load_config('production_config.yaml')
        
        # Get labeled records
        labeled_data = get_labeled_records()
        logging.info(f"Found {len(labeled_data)} labeled records this week")
        
        if len(labeled_data) < 10:
            logging.info("Insufficient labels for retraining. Skipping.")
            return
        
        # Retrain
        prev_model = get_latest_model()
        new_model_path = retrain_model(labeled_data, config, prev_model)
        
        if new_model_path:
            update_production_model(new_model_path)
            logging.info("Production model updated")
        
        logging.info("WEEKLY RETRAINING JOB COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        logging.error(f"RETRAINING JOB FAILED: {e}")
        send_slack_alert(f"❌ Retraining failed: {e}")
        raise

if __name__ == "__main__":
    main()
```

---

## Monitoring Dashboard Setup

### Key Metrics to Track

1. **Model Performance**
   - Closed PR-AUC (primary)
   - Closed F1, Precision, Recall
   - Model inference time (ms/record)

2. **Labeling Quality**
   - Auto-label precision (audit sample)
   - Inter-reviewer agreement (if multiple reviewers)
   - Error rate on agreed hard cases

3. **Operational Metrics**
   - Review queue length (should stay <100)
   - Review time per record (target: 2-3 min)
   - Batch processing time
   - Model retraining time

4. **Cost Metrics**
   - Total labels created (gold + silver)
   - Manual reviews done
   - Leverage ratio (silver / gold)
   - Cost savings vs manual labeling

### Sample Grafana Dashboard Config

```json
{
  "dashboard": {
    "title": "Label-Coverage Active-Labeling Loop",
    "panels": [
      {
        "title": "Closed PR-AUC (Primary)",
        "type": "graph",
        "targets": [
          {
            "expr": "metric_closed_pr_auc",
            "legendFormat": "PR-AUC"
          }
        ],
        "alert": {
          "condition": "below 0.9 * baseline"
        }
      },
      {
        "title": "Auto-Label Precision",
        "type": "gauge",
        "targets": [
          {
            "expr": "metric_auto_precision"
          }
        ],
        "thresholds": ["0.75", "0.85", "1.0"]
      },
      {
        "title": "Review Queue Length",
        "type": "stat",
        "targets": [
          {
            "expr": "metric_review_queue_length"
          }
        ],
        "alert": {
          "condition": "above 100"
        }
      },
      {
        "title": "Labeling Efficiency",
        "type": "graph",
        "targets": [
          {
            "expr": "metric_leverage_ratio"
          }
        ]
      }
    ]
  }
}
```

---

## Troubleshooting Guide

### Issue: Auto-label precision drops below 75%

**Solution**:
1. Check model retraining logs (any errors?)
2. Lower auto-accept threshold by 0.05 (e.g., 0.85 → 0.80)
3. Manually audit 20-30 failed auto-labels (find pattern)
4. If pattern found: add feature or rule to catch it
5. Monitor for 2 weeks

### Issue: PR-AUC plateaus or decreases

**Solution**:
1. Check for data drift (new data distribution different?)
2. Manually audit high-uncertainty records (model confusion)
3. Increase review budget (sample more uncertain ones)
4. Try Uncertainty sampling instead of Random
5. Consider model upgrade (LR → XGBoost)

### Issue: Review queue growing (reviewers behind)

**Solution**:
1. Reduce batch size (e.g., 400 → 200 records)
2. Reduce review budget (e.g., 5% → 3%)
3. Increase reviewer team size
4. Defer harder batches (data with low confidence)
5. Auto-label more aggressively (lower thresholds)

### Issue: Model training fails or crashes

**Solution**:
1. Check feature extraction logs (nulls? missing values?)
2. Ensure training set has at least 20 records
3. Verify label distribution (both classes present?)
4. Check for duplicate IDs in training data
5. Try increasing max_iter in LogisticRegression
6. Fallback to using previous model (no retraining)

### Issue: Inference is slow (>100ms per record)

**Solution**:
1. Profile feature extraction (usually the bottleneck)
2. Cache features if possible
3. Use batch inference (vectorized) instead of row-by-row
4. Consider lighter feature bundle (low_only instead of low_plus_medium)
5. Plan to optimize in next iteration

---

## Success Criteria Checklist

### Week 1: Foundation
- [ ] Batch 1 auto-label precision > 75%
- [ ] No critical errors in logs
- [ ] Labeling team comfortable with process
- [ ] Review time < 30 min per batch

### Week 2: Validation
- [ ] Batch 2 auto-label precision > 75% (sustained)
- [ ] PR-AUC improving (even if small)
- [ ] No data quality issues found
- [ ] Reviewer feedback collected & addressed

### Week 4: Scale
- [ ] Extended to 3+ datasets
- [ ] Automated jobs running nightly
- [ ] Dashboard monitoring all key metrics
- [ ] Cost savings calculated & verified

### Month 3: Production
- [ ] Consistent auto-label precision > 80%
- [ ] PR-AUC improved 30-50% from baseline
- [ ] 80%+ cost savings vs manual labeling
- [ ] Zero critical incidents

---

**Document Version**: 1.0  
**Last Updated**: 2024-03-05  
**Next Review**: After first production batch
