"""Shared metric computation for open/closed prediction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score


def compute_metrics(
    y_true,
    y_pred,
    y_score_open=None,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "open_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "open_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "open_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "closed_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "closed_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "closed_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
    }

    if y_score_open is not None:
        y_score_open = np.asarray(y_score_open)
        y_score_closed = 1.0 - y_score_open
        y_true_closed = (y_true == 0).astype(int)
        metrics["pr_auc_closed"] = float(average_precision_score(y_true_closed, y_score_closed))

    return metrics


def format_metrics_block(metrics: dict[str, float], title: str = "Metrics") -> str:
    lines = [title]
    lines.append(
        "  Open:   Precision: {open_precision:.3f}  Recall: {open_recall:.3f}  F1: {open_f1:.3f}".format(**metrics)
    )
    lines.append(
        "  Closed: Precision: {closed_precision:.3f}  Recall: {closed_recall:.3f}  F1: {closed_f1:.3f}".format(**metrics)
    )
    lines.append("  Accuracy: {accuracy:.3f}".format(**metrics))
    if "pr_auc_closed" in metrics:
        lines.append("  PR-AUC (closed): {pr_auc_closed:.3f}".format(**metrics))
    return "\n".join(lines)
