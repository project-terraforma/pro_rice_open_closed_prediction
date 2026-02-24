# Model Results: Open/Closed Place Prediction

Note: `data/matching_validation/` contains matching-validation datasets (`label=match/no-match`) and is excluded from all open/closed results below.

**Dataset:** 2,397 training samples, 685 validation samples, 343 test samples  
**Class Distribution (Test):** 91% open, 9% closed (severe imbalance)  
**Objective:** Maximize accuracy while minimizing false closed predictions (users thinking places are closed when they're actually open)

Note: LR validation metrics for no-confidence variants were refreshed from current code on Feb 11, 2026 using `src/models/logistic_regression.py --split val`.

---

## Executive Summary - Test Results

| Model | Accuracy | Open Precision | Open Recall | Closed Precision | Closed Recall |
|-------|----------|-----------------|-------------|------------------|---------------|
| **Rules-Only Baseline** | 0.840 | 0.930 | 0.891 | 0.227 | 0.323 |
| **Single-Stage LR (No Confidence)** | 0.793 | 0.941 | 0.824 | 0.214 | 0.484 |
| **Single-Stage LR (Source Confidence)** | 0.682 | 0.943 | 0.692 | 0.158 | 0.581 |
| **Two-Stage LR (No Confidence)** | 0.904 | 0.916 | 0.984 | 0.375 | 0.097 |
| **Two-Stage LR (Source Confidence)** | 0.898 | 0.924 | 0.968 | 0.375 | 0.194 |
| **Random Forest** | 0.837 | 0.935 | 0.881 | 0.245 | 0.387 |
| **Two-Stage LightGBM (No Confidence)** | 0.895 | 0.921 | 0.968 | 0.333 | 0.161 |

| **Less Conservative Two-Stage LR (No Confidence)** | 0.857 | 0.934 | 0.907 | 0.275 | 0.355 |

---

## Detailed Model Metrics

### Rules-Only Baseline

Validation Report:
Open: Precision: 0.932    Recall: 0.923    F1: 0.927
Closed: Precision: 0.304    Recall: 0.333    F1: 0.318
Accuracy: 0.869

Test Report:
Open: Precision: 0.930    Recall: 0.891    F1: 0.910
Closed: Precision: 0.227    Recall: 0.323    F1: 0.267
Accuracy: 0.840

### Single-Stage LR (No Confidence)

Validation Report:
Open: Precision: 0.937    Recall: 0.812    F1: 0.870
Closed: Precision: 0.199    Recall: 0.460    F1: 0.278
Accuracy: 0.780

Test Report:
Open: Precision: 0.941    Recall: 0.824    F1: 0.879
Closed: Precision: 0.214    Recall: 0.484    F1: 0.297
Accuracy: 0.793

### Single-Stage LR (Source Confidence)

Validation Report:
Open: Precision: 0.957    Recall: 0.752    F1: 0.842
Closed: Precision: 0.214    Recall: 0.667    F1: 0.324
Accuracy: 0.745

Test Report:
Open: Precision: 0.943    Recall: 0.692    F1: 0.799
Closed: Precision: 0.158    Recall: 0.581    F1: 0.248
Accuracy: 0.682

### Two-Stage LR (No Confidence)

Validation Report:
Open: Precision: 0.936    Recall: 0.847    F1: 0.889
Closed: Precision: 0.221    Recall: 0.429    F1: 0.292
Accuracy: 0.809

Test Report:
Open: Precision: 0.916    Recall: 0.984    F1: 0.949
Closed: Precision: 0.375    Recall: 0.097    F1: 0.154
Accuracy: 0.904

### Two-Stage LR (Source Confidence)

Validation Report:
Open: Precision: 0.921    Recall: 0.960    F1: 0.940
Closed: Precision: 0.324    Recall: 0.190    F1: 0.240
Accuracy: 0.889

Test Report:
Open: Precision: 0.924    Recall: 0.968    F1: 0.945
Closed: Precision: 0.375    Recall: 0.194    F1: 0.255
Accuracy: 0.898

### Random Forest

Validation Report:
Open: Precision: 0.929    Recall: 0.904    F1: 0.916
Closed: Precision: 0.250    Recall: 0.317    F1: 0.280
Accuracy: 0.850

Test Report:
Open: Precision: 0.935    Recall: 0.881    F1: 0.908
Closed: Precision: 0.245    Recall: 0.387    F1: 0.300
Accuracy: 0.837

### Two-Stage LightGBM (No Confidence)

Validation Report:
Open: Precision: 0.921    Recall: 0.960    F1: 0.940
Closed: Precision: 0.324    Recall: 0.190    F1: 0.240
Accuracy: 0.889

Test Report:
Open: Precision: 0.921    Recall: 0.968    F1: 0.944
Closed: Precision: 0.333    Recall: 0.161    F1: 0.217
Accuracy: 0.895

Notes:
- Single-stage LightGBM performed worse (Accuracy ~0.708, Closed F1 ~0.19).
- A small hyperparameter sweep did not improve closed F1 beyond ~0.21.
