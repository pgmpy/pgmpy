"""Abstract base class for data simulators.

All concrete simulators inherit from BaseSimulator and implement:
- simulate(seed) — Generate synthetic data
- get_name() — Return simulator name
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import networkx as nx
import pandas as pd


@dataclass
class SimulationOutput:
    """Output from a simulator."""
    
    dag: nx.DiGraph
    data: pd.DataFrame
    metadata: dict


class BaseSimulator(ABC):
    """Abstract base class for data simulators."""
    
    def __init__(self, n_nodes: int, n_samples: int, seed: Optional[int] = None):
        """
        Initialize simulator.
        
        Parameters
        ----------
        n_nodes : int
            Number of nodes in the DAG
        n_samples : int
            Number of samples to generate
        seed : int, optional
            Random seed for reproducibility
        """
        if n_nodes < 1:
            raise ValueError(f"n_nodes must be >= 1, got {n_nodes}")
        if n_samples < 0:
            raise ValueError(f"n_samples must be >= 0, got {n_samples}")
        
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        self.seed = seed
    
    @abstractmethod
    def simulate(self, seed: Optional[int] = None) -> SimulationOutput:
        """
        Simulate synthetic causal data.
        
        Parameters
        ----------
        seed : int, optional
            Random seed
            
        Returns
        -------
        SimulationOutput
            Generated DAG and data
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get simulator name."""
        pass

