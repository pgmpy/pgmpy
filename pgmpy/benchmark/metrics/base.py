"""Abstract base class for evaluation metrics.

All concrete metrics inherit from BaseMetric and implement:
- compute(pred_dag, true_dag) — Evaluate prediction against ground truth
- get_name() — Return metric name
"""

from abc import ABC, abstractmethod
import networkx as nx


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""
    
    def __init__(self, name: str):
        """
        Initialize metric.
        
        Parameters
        ----------
        name : str
            Descriptive name of the metric
        """
        self.name = name
    
    @abstractmethod
    def compute(self, pred_dag: nx.DiGraph, true_dag: nx.DiGraph):
        """
        Compute metric value.
        
        Parameters
        ----------
        pred_dag : nx.DiGraph
            Predicted DAG
        true_dag : nx.DiGraph
            Ground truth DAG
            
        Returns
        -------
        float or dict
            Metric value(s)
        """
        pass
    
    def get_name(self) -> str:
        """Get metric name."""
        return self.name

# NOTE: Direct imports from this file are no longer recommended.
# Use pgmpy.benchmark module instead (see __init__.py)

