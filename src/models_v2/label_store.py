"""
Label store: maintains cumulative gold and silver labels across batches.
"""

import pandas as pd
import numpy as np
from typing import Set, Dict
from datetime import datetime


class LabelStore:
    """Manages gold and silver labels across simulation rounds."""
    
    def __init__(self):
        """Initialize label store."""
        self.gold_labels = {}  # id -> {label, confidence, source, timestamp}
        self.silver_labels = {}  # id -> {label, confidence, source, timestamp}
        self.deferred_ids = set()
    
    def add_gold_labels(
        self,
        ids: np.ndarray,
        labels: np.ndarray,
        confidence: float = 1.0,
        source: str = "review",
    ) -> None:
        """
        Add reviewed (gold) labels.
        
        Args:
            ids: Record IDs
            labels: True labels (0=closed, 1=open)
            confidence: Confidence in these labels (0-1)
            source: Label source ('manual_review', 'api', etc.)
        """
        timestamp = datetime.now().isoformat()
        for id_, label in zip(ids, labels):
            self.gold_labels[id_] = {
                'label': label,
                'confidence': confidence,
                'source': source,
                'timestamp': timestamp,
            }
            # Remove from silver if it was there
            self.silver_labels.pop(id_, None)
            self.deferred_ids.discard(id_)
    
    def add_silver_labels(
        self,
        ids: np.ndarray,
        labels: np.ndarray,
        confidence: np.ndarray,
        source: str = "auto",
    ) -> None:
        """
        Add auto-labeled (silver) labels.
        
        Args:
            ids: Record IDs
            labels: Predicted labels (0=closed, 1=open)
            confidence: Confidence in each label (0-1 array)
            source: Label source ('auto_high_confidence', etc.)
        """
        timestamp = datetime.now().isoformat()
        for id_, label, conf in zip(ids, labels, confidence):
            # Only add if not already in gold
            if id_ not in self.gold_labels:
                self.silver_labels[id_] = {
                    'label': label,
                    'confidence': conf,
                    'source': source,
                    'timestamp': timestamp,
                }
    
    def add_deferred(self, ids: np.ndarray) -> None:
        """Mark records as deferred (no label assigned this round)."""
        for id_ in ids:
            if id_ not in self.gold_labels and id_ not in self.silver_labels:
                self.deferred_ids.add(id_)
    
    def get_labeled_ids(self) -> Set:
        """Get all records that have been labeled (gold or silver)."""
        return set(self.gold_labels.keys()) | set(self.silver_labels.keys())
    
    def get_training_data(self) -> pd.DataFrame:
        """
        Combine gold and silver labels into training dataset.
        
        Returns:
            DataFrame with columns: id, label, weight (1.0 for gold, 0.4-0.8 for silver)
        """
        records = []
        
        # Gold labels (weight = 1.0)
        for id_, info in self.gold_labels.items():
            records.append({
                'id': id_,
                'label': info['label'],
                'weight': 1.0,
                'confidence': info['confidence'],
                'source': info['source'],
            })
        
        # Silver labels (weight proportional to confidence)
        for id_, info in self.silver_labels.items():
            # Map confidence to weight: 0.5 -> 0.4, 1.0 -> 0.8
            weight = 0.4 + 0.4 * info['confidence']
            records.append({
                'id': id_,
                'label': info['label'],
                'weight': weight,
                'confidence': info['confidence'],
                'source': info['source'],
            })
        
        return pd.DataFrame(records)
    
    def get_stats(self) -> Dict:
        """Get label store statistics."""
        gold_df = pd.DataFrame(self.gold_labels).T
        silver_df = pd.DataFrame(self.silver_labels).T
        
        stats = {
            'total_labeled': len(self.gold_labels) + len(self.silver_labels),
            'n_gold': len(self.gold_labels),
            'n_silver': len(self.silver_labels),
            'n_deferred': len(self.deferred_ids),
            'gold_open_pct': gold_df['label'].mean() if len(gold_df) > 0 else None,
            'silver_open_pct': silver_df['label'].mean() if len(silver_df) > 0 else None,
        }
        return stats
    
    def print_stats(self) -> None:
        """Print label store statistics."""
        stats = self.get_stats()
        print(f"\nLabel Store Status:")
        print(f"  Gold labels: {stats['n_gold']}")
        if stats['n_gold'] > 0:
            print(f"    Open: {int(stats['gold_open_pct'] * stats['n_gold'])} ({stats['gold_open_pct']:.1%})")
        print(f"  Silver labels: {stats['n_silver']}")
        if stats['n_silver'] > 0:
            print(f"    Open: {int(stats['silver_open_pct'] * stats['n_silver'])} ({stats['silver_open_pct']:.1%})")
        print(f"  Deferred: {stats['n_deferred']}")
        print(f"  Total labeled: {stats['total_labeled']}")


if __name__ == "__main__":
    # Test label store
    store = LabelStore()
    
    # Add some gold labels
    store.add_gold_labels(
        ids=np.array(['a', 'b', 'c']),
        labels=np.array([0, 1, 0]),
        confidence=1.0,
        source='manual_review',
    )
    
    # Add some silver labels
    store.add_silver_labels(
        ids=np.array(['d', 'e', 'f']),
        labels=np.array([1, 0, 1]),
        confidence=np.array([0.9, 0.7, 0.8]),
        source='auto_high_confidence',
    )
    
    store.print_stats()
    
    training_data = store.get_training_data()
    print(f"\nTraining data:\n{training_data}")
