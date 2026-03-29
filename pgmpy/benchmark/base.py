"""
Base classes and interfaces for the benchmark framework.

Defines abstract base classes that all simulators, metrics, and methods must conform to.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import networkx as nx


@dataclass
class SimulationOutput:
    """Output container for data simulators."""
    
    dag: nx.DiGraph
    """Ground truth directed acyclic graph."""
    
    data: pd.DataFrame
    """Generated observational data from the DAG."""
    
    params: Dict[str, Any]
    """Simulator parameters used to generate this data."""
    
    seed: Optional[int] = None
    """Random seed used (if deterministic)."""


class BaseSimulator(ABC):
    """
    Abstract base class for data simulators.
    
    All simulators must implement the `simulate()` method which returns
    a (DAG, DataFrame) pair with ground truth structure.
    """
    
    @abstractmethod
    def simulate(self, seed: Optional[int] = None) -> SimulationOutput:
        """
        Generate synthetic data from a causal model.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
            
        Returns
        -------
        SimulationOutput
            Contains ground truth DAG and observational data.
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return simulator configuration parameters."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return a descriptive name for this simulator."""
        pass


@dataclass
class MetricResult:
    """Output container for a single metric evaluation."""
    
    name: str
    """Metric name (e.g., 'SHD', 'precision')."""
    
    value: float
    """Computed metric value."""
    
    metadata: Dict[str, Any]
    """Optional additional context (e.g., components of the metric)."""


class BaseMetric(ABC):
    """
    Abstract base class for evaluation metrics.
    
    All metrics take (estimated_DAG, ground_truth_DAG) and return a MetricResult.
    """
    
    @abstractmethod
    def compute(
        self,
        estimated_dag: nx.DiGraph,
        ground_truth_dag: nx.DiGraph,
    ) -> MetricResult:
        """
        Compute the metric.
        
        Parameters
        ----------
        estimated_dag : nx.DiGraph
            Estimated causal DAG from algorithm output.
        ground_truth_dag : nx.DiGraph
            True ground-truth DAG.
            
        Returns
        -------
        MetricResult
            The computed metric value and metadata.
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the metric name."""
        pass
