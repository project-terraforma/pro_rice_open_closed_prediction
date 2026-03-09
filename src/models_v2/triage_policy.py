"""
Triage policy: route batch records into auto-accept, review-queue, or defer.

This module scores confidence and impact, then applies deterministic routing
to identify high-confidence cases vs. uncertain cases needing review.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class TriagePolicy:
    """Route records into auto-accept, review-queue, or defer based on model confidence."""
    
    def __init__(
        self,
        t_closed_high: float = 0.85,
        t_open_low: float = 0.15,
        uncertainty_method: str = "inverse_entropy",
        impact_method: str = "uniform",
    ):
        """
        Initialize triage policy.
        
        Args:
            t_closed_high: Threshold to auto-accept closed (p_closed >= threshold)
            t_open_low: Threshold to auto-accept open (p_closed <= threshold)
            uncertainty_method: 'inverse_entropy' or 'symmetric'
            impact_method: 'uniform' or 'category_weight'
        """
        self.t_closed_high = t_closed_high
        self.t_open_low = t_open_low
        self.uncertainty_method = uncertainty_method
        self.impact_method = impact_method
    
    def compute_uncertainty(self, p_closed: np.ndarray) -> np.ndarray:
        """
        Compute uncertainty score for each record.
        
        Higher = more uncertain (closer to decision boundary).
        
        Args:
            p_closed: Model probability of closed (0-1)
        
        Returns:
            Uncertainty scores (0-1)
        """
        if self.uncertainty_method == "inverse_entropy":
            # Entropy-based: max entropy at 0.5
            eps = 1e-7
            p_closed = np.clip(p_closed, eps, 1 - eps)
            entropy = -(p_closed * np.log(p_closed) + (1 - p_closed) * np.log(1 - p_closed))
            # Normalize entropy to [0, 1] (max entropy at 0.5 is ln(2))
            max_entropy = np.log(2)
            uncertainty = entropy / max_entropy
        else:  # symmetric
            # Distance from decision boundary
            uncertainty = 1.0 - 2 * np.abs(p_closed - 0.5)
        
        return uncertainty
    
    def compute_impact(
        self,
        batch_df: pd.DataFrame,
        method: str = "uniform",
    ) -> np.ndarray:
        """
        Compute impact score for each record.
        
        Higher = more important to label.
        
        Args:
            batch_df: Batch dataframe
            method: 'uniform' or 'category_weight'
        
        Returns:
            Impact scores (0-1, normalized)
        """
        if method == "uniform":
            return np.ones(len(batch_df)) / len(batch_df)
        
        elif method == "category_weight":
            # Rare categories get higher weight
            category_counts = batch_df['category'].value_counts(normalize=True)
            # Inverse frequency: rare = high weight
            category_weight = 1.0 / (1 + category_counts)
            impact = batch_df['category'].map(category_weight).fillna(0.5)
            impact = impact / impact.sum()  # Normalize
            return impact.values
        
        else:
            return np.ones(len(batch_df)) / len(batch_df)
    
    def route_batch(
        self,
        batch_df: pd.DataFrame,
        p_closed: np.ndarray,
        allow_rule_agreement: bool = True,
    ) -> pd.DataFrame:
        """
        Route batch into auto-accept, review, defer.
        
        Args:
            batch_df: Batch dataframe
            p_closed: Model probability of closed (1D array)
            allow_rule_agreement: If True, can use simple heuristics
        
        Returns:
            Routed dataframe with new columns:
            - p_closed, uncertainty, impact_score
            - route ('auto_accept', 'review', 'defer')
            - auto_label (predicted label for auto_accept only)
            - confidence_bin ('high', 'medium', 'low')
        """
        result = batch_df.copy()
        result['p_closed'] = p_closed
        
        # Compute uncertainty and impact
        uncertainty = self.compute_uncertainty(p_closed)
        impact = self.compute_impact(result, method=self.impact_method)
        
        result['uncertainty'] = uncertainty
        result['impact_score'] = impact
        
        # Routing logic
        routes = []
        auto_labels = []
        confidence_bins = []
        
        for i in range(len(result)):
            p = p_closed[i]
            unc = uncertainty[i]
            
            # Auto-accept closed (high confidence closed prediction)
            if p >= self.t_closed_high:
                routes.append('auto_accept')
                auto_labels.append(0)  # closed
                confidence_bins.append('high')
            
            # Auto-accept open (high confidence open prediction)
            elif p <= self.t_open_low:
                routes.append('auto_accept')
                auto_labels.append(1)  # open
                confidence_bins.append('high')
            
            # Medium confidence: uncertain
            elif unc > 0.6:
                routes.append('review')
                auto_labels.append(None)
                confidence_bins.append('low')
            
            # Low uncertainty but not extreme confidence
            else:
                routes.append('defer')
                auto_labels.append(None)
                confidence_bins.append('medium')
        
        result['route'] = routes
        result['auto_label'] = auto_labels
        result['confidence_bin'] = confidence_bins
        
        return result


def print_triage_summary(routed_df: pd.DataFrame) -> None:
    """Print triage results."""
    print(f"\nTriage Results:")
    print(f"  Auto-accept: {(routed_df['route'] == 'auto_accept').sum()} ({(routed_df['route'] == 'auto_accept').mean():.1%})")
    print(f"  Review queue: {(routed_df['route'] == 'review').sum()} ({(routed_df['route'] == 'review').mean():.1%})")
    print(f"  Defer: {(routed_df['route'] == 'defer').sum()} ({(routed_df['route'] == 'defer').mean():.1%})")
    
    # Auto-accept breakdown
    auto_accept_mask = routed_df['route'] == 'auto_accept'
    if auto_accept_mask.sum() > 0:
        auto_closed = (routed_df[auto_accept_mask]['auto_label'] == 0).sum()
        auto_open = (routed_df[auto_accept_mask]['auto_label'] == 1).sum()
        print(f"    Auto-closed: {auto_closed}, Auto-open: {auto_open}")


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, '/Users/claricepark/Desktop/pro_rice_open_closed_prediction/src/models_v2')
    
    from build_sim_batches import build_sim_batches, load_and_prepare_data
    
    data_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/project_c_samples.csv"
    train_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/train_split.parquet"
    val_path = "/Users/claricepark/Desktop/pro_rice_open_closed_prediction/data/project_c_samples/val_split.parquet"
    
    df = load_and_prepare_data(data_path, train_path, val_path)
    batches, _ = build_sim_batches(df, n_batches=3, seed=42)
    
    # Mock predictions
    policy = TriagePolicy(t_closed_high=0.85, t_open_low=0.15)
    p_closed = np.random.uniform(0, 1, len(batches[0]))
    routed = policy.route_batch(batches[0], p_closed)
    
    print_triage_summary(routed)
