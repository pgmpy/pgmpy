"""
Metrics engine for the benchmark framework.

Provides standard evaluation metrics for causal graph recovery:
    - SHD (Structural Hamming Distance)
    - Precision, Recall, F1
    - SID (Structural Intervention Distance)
    - Orientation F1
    - Runtime
"""

from typing import Callable, Dict, Any, List
import networkx as nx
import numpy as np

from pgmpy.benchmark.base import BaseMetric, MetricResult


class SHDMetric(BaseMetric):
    """Structural Hamming Distance between two graphs."""
    
    def compute(
        self,
        estimated_dag: nx.DiGraph,
        ground_truth_dag: nx.DiGraph,
    ) -> MetricResult:
        """
        Compute SHD (sum of missing, extra, and reversed edges).
        
        Parameters
        ----------
        estimated_dag : nx.DiGraph
            Estimated graph.
        ground_truth_dag : nx.DiGraph
            Ground truth graph.
            
        Returns
        -------
        MetricResult
            SHD value and breakdown.
        """
        # Get edges from both graphs
        true_edges = set(ground_truth_dag.edges())
        est_edges = set(estimated_dag.edges())
        
        # Calculate differences
        missing = len(true_edges - est_edges)
        extra = len(est_edges - true_edges)
        
        # Check for reversed edges (u,v) in true but (v,u) in est
        reversed_count = 0
        for u, v in true_edges - est_edges:
            if (v, u) in est_edges:
                reversed_count += 1
        
        shd = missing + extra + reversed_count
        
        return MetricResult(
            name="SHD",
            value=float(shd),
            metadata={
                "missing": missing,
                "extra": extra,
                "reversed": reversed_count,
                "total_true_edges": len(true_edges),
                "total_est_edges": len(est_edges),
            },
        )
    
    def get_name(self) -> str:
        """Return metric name."""
        return "SHD"


class PrecisionRecallMetric(BaseMetric):
    """Edge-level precision, recall, and F1 score."""
    
    def compute(
        self,
        estimated_dag: nx.DiGraph,
        ground_truth_dag: nx.DiGraph,
    ) -> MetricResult:
        """
        Compute edge-level precision, recall, and F1.
        
        Parameters
        ----------
        estimated_dag : nx.DiGraph
            Estimated graph.
        ground_truth_dag : nx.DiGraph
            Ground truth graph.
            
        Returns
        -------
        MetricResult
            Precision, recall, and F1 values.
        """
        true_edges = set(ground_truth_dag.edges())
        est_edges = set(estimated_dag.edges())
        
        # True positives: edges in both
        tp = len(true_edges & est_edges)
        
        # False positives: edges in est but not in true
        fp = len(est_edges - true_edges)
        
        # False negatives: edges in true but not in est
        fn = len(true_edges - est_edges)
        
        # Compute precision, recall, F1 (avoid division by zero)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        
        return MetricResult(
            name="PrecisionRecall",
            value=f1,  # Use F1 as primary
            metadata={
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            },
        )
    
    def get_name(self) -> str:
        """Return metric name."""
        return "PrecisionRecall"


class OrientationMetric(BaseMetric):
    """Orientation F1: F1 restricted to correctly identified edges."""
    
    def compute(
        self,
        estimated_dag: nx.DiGraph,
        ground_truth_dag: nx.DiGraph,
    ) -> MetricResult:
        """
        Compute orientation F1: among correctly detected edges,
        what fraction had the correct direction?
        
        Parameters
        ----------
        estimated_dag : nx.DiGraph
            Estimated graph.
        ground_truth_dag : nx.DiGraph
            Ground truth graph.
            
        Returns
        -------
        MetricResult
            Orientation F1 and details.
        """
        true_edges = set(ground_truth_dag.edges())
        est_edges = set(estimated_dag.edges())
        
        # Correctly directed edges
        correct_orientations = len(true_edges & est_edges)
        
        # Edges with correct endpoints but wrong direction
        # (u,v) in true but (v,u) in est_edges
        wrong_orientations = 0
        for u, v in true_edges:
            if (v, u) in est_edges:
                wrong_orientations += 1
        
        # F1 for orientation: among edges that could have been oriented
        denom = correct_orientations + wrong_orientations
        orientation_recall = (
            correct_orientations / denom if denom > 0 else 0.0
        )
        
        return MetricResult(
            name="OrientationF1",
            value=orientation_recall,
            metadata={
                "correct_orientations": correct_orientations,
                "wrong_orientations": wrong_orientations,
                "orientation_recall": orientation_recall,
            },
        )
    
    def get_name(self) -> str:
        """Return metric name."""
        return "OrientationF1"


class SIDMetric(BaseMetric):
    """Structural Intervention Distance (simplified version)."""
    
    def compute(
        self,
        estimated_dag: nx.DiGraph,
        ground_truth_dag: nx.DiGraph,
    ) -> MetricResult:
        """
        Simplified SID: count causal relationships that differ.
        
        Full SID requires computing interventional distributions,
        so this is a proxy based on graph structure.
        
        Parameters
        ----------
        estimated_dag : nx.DiGraph
            Estimated graph.
        ground_truth_dag : nx.DiGraph
            Ground truth graph.
            
        Returns
        -------
        MetricResult
            SID value (simplified).
        """
        # For each node, check if descendants match
        differences = 0
        total_comparisons = 0
        
        all_nodes = set(ground_truth_dag.nodes()) & set(estimated_dag.nodes())
        
        for node in all_nodes:
            true_desc = set(nx.descendants(ground_truth_dag, node))
            est_desc = set(nx.descendants(estimated_dag, node)) if node in estimated_dag else set()
            
            # Symmetric difference in descendants
            diff = len(true_desc ^ est_desc)
            differences += diff
            total_comparisons += len(all_nodes)
        
        sid = differences / total_comparisons if total_comparisons > 0 else 0.0
        
        return MetricResult(
            name="SID",
            value=float(sid),
            metadata={"differences": differences, "comparisons": total_comparisons},
        )
    
    def get_name(self) -> str:
        """Return metric name."""
        return "SID"


class RuntimeMetric(BaseMetric):
    """Placeholder for runtime measurement (computed externally)."""
    
    def __init__(self, runtime: float = 0.0):
        self.runtime = runtime
    
    def compute(
        self,
        estimated_dag: nx.DiGraph,
        ground_truth_dag: nx.DiGraph,
    ) -> MetricResult:
        """Return pre-recorded runtime."""
        return MetricResult(
            name="Runtime",
            value=self.runtime,
            metadata={"seconds": self.runtime},
        )
    
    def get_name(self) -> str:
        """Return metric name."""
        return "Runtime"


class MetricsRegistry:
    """Registry and factory for metrics."""
    
    _metrics: Dict[str, Callable[[], BaseMetric]] = {}
    
    @classmethod
    def register(cls, name: str, factory: Callable[[], BaseMetric]):
        """
        Register a new metric.
        
        Parameters
        ----------
        name : str
            Metric name.
        factory : callable
            Callable that returns a BaseMetric instance.
        """
        cls._metrics[name] = factory
    
    @classmethod
    def get(cls, name: str) -> BaseMetric:
        """
        Get a metric by name.
        
        Parameters
        ----------
        name : str
            Metric name.
            
        Returns
        -------
        BaseMetric
            Metric instance.
        """
        if name not in cls._metrics:
            raise ValueError(f"Unknown metric: {name}")
        return cls._metrics[name]()
    
    @classmethod
    def list_metrics(cls) -> List[str]:
        """List all registered metrics."""
        return list(cls._metrics.keys())


# Register default metrics
MetricsRegistry.register("shd", SHDMetric)
MetricsRegistry.register("precision_recall", PrecisionRecallMetric)
MetricsRegistry.register("orientation_f1", OrientationMetric)
MetricsRegistry.register("sid", SIDMetric)
MetricsRegistry.register("runtime", lambda: RuntimeMetric(0.0))


# Export convenience functions
def shd(estimated_dag: nx.DiGraph, ground_truth_dag: nx.DiGraph) -> MetricResult:
    """Compute SHD metric."""
    return SHDMetric().compute(estimated_dag, ground_truth_dag)


def precision_recall(
    estimated_dag: nx.DiGraph, ground_truth_dag: nx.DiGraph
) -> MetricResult:
    """Compute precision/recall/F1 metrics."""
    return PrecisionRecallMetric().compute(estimated_dag, ground_truth_dag)


def orientation_f1(
    estimated_dag: nx.DiGraph, ground_truth_dag: nx.DiGraph
) -> MetricResult:
    """Compute orientation F1 metric."""
    return OrientationMetric().compute(estimated_dag, ground_truth_dag)


def sid(estimated_dag: nx.DiGraph, ground_truth_dag: nx.DiGraph) -> MetricResult:
    """Compute SID metric."""
    return SIDMetric().compute(estimated_dag, ground_truth_dag)
