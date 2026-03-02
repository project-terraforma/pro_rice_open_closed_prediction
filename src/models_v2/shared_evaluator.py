"""Shared evaluator helpers for thresholded and score-based reporting."""

from __future__ import annotations

import numpy as np

try:
    from shared_metrics import compute_metrics, format_metrics_block
except ImportError:  # pragma: no cover - package import fallback
    from .shared_metrics import compute_metrics, format_metrics_block


def evaluate_predictions(y_true, y_pred, y_score_open=None, model_name: str = "Model") -> dict:
    metrics = compute_metrics(y_true=y_true, y_pred=y_pred, y_score_open=y_score_open)
    print()
    print(format_metrics_block(metrics, title=f"{model_name} - Results:"))
    return {"model": model_name, **metrics}


def evaluate_from_scores(y_true, y_score_open, threshold: float = 0.5, model_name: str = "Model") -> dict:
    y_score_open = np.asarray(y_score_open)
    y_pred = (y_score_open >= threshold).astype(int)
    metrics = evaluate_predictions(y_true, y_pred, y_score_open=y_score_open, model_name=model_name)
    metrics["threshold"] = float(threshold)
    return metrics
