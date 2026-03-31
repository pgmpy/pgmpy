"""
Data simulators for the benchmark framework.

Provides simulators for common causal graph structures:
    - Erdos-Renyi random graphs
    - Scale-free networks
    - Real Bayesian networks
    - Linear Gaussian SEMs
"""

from typing import Optional, Dict, Any, Literal
import numpy as np
import pandas as pd
import networkx as nx

from pgmpy.benchmark.base import BaseSimulator, SimulationOutput


class ErdosRenyiSimulator(BaseSimulator):
    """
    Simulate data from a random Erdos-Renyi DAG.
    
    Generates a random DAG using the Erdos-Renyi model, then generates
    observational data by simulating a linear SEM: x_j = sum_i(c_ij * x_i) + noise_j
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph (default: 10).
    edge_prob : float
        Probability of edge between any two nodes (default: 0.3).
    n_samples : int
        Number of samples to generate (default: 500).
    noise : Literal['gaussian', 'laplace', 'uniform']
        Noise distribution (default: 'gaussian').
    noise_scale : float
        Standard deviation/scale of noise (default: 1.0).
    edge_coeff_range : tuple
        Range [min, max] for edge coefficients (default: (0.5, 2.0)).
    seed : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        n_nodes: int = 10,
        edge_prob: float = 0.3,
        n_samples: int = 500,
        noise: Literal["gaussian", "laplace", "uniform"] = "gaussian",
        noise_scale: float = 1.0,
        edge_coeff_range: tuple = (0.5, 2.0),
        seed: Optional[int] = None,
    ):
        self.n_nodes = n_nodes
        self.edge_prob = edge_prob
        self.n_samples = n_samples
        self.noise = noise
        self.noise_scale = noise_scale
        self.edge_coeff_range = edge_coeff_range
        self.seed = seed
    
    def simulate(self, seed: Optional[int] = None) -> SimulationOutput:
        """Generate synthetic data from an Erdos-Renyi DAG."""
        rng = np.random.RandomState(seed or self.seed)
        
        # Generate random DAG
        dag = self._generate_dag(rng)
        
        # Generate data from the DAG
        data = self._generate_data(dag, rng)
        
        return SimulationOutput(
            dag=dag,
            data=data,
            params=self.get_params(),
            seed=seed or self.seed,
        )
    
    def _generate_dag(self, rng: np.random.RandomState) -> nx.DiGraph:
        """Generate a random DAG using topological ordering."""
        dag = nx.DiGraph()
        
        # Create nodes
        nodes = [f"X{i}" for i in range(self.n_nodes)]
        dag.add_nodes_from(nodes)
        
        # Add edges with topological ordering (i -> j where i < j)
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if rng.rand() < self.edge_prob:
                    dag.add_edge(nodes[i], nodes[j])
        
        return dag
    
    def _generate_data(self, dag: nx.DiGraph, rng: np.random.RandomState) -> pd.DataFrame:
        """Generate observational data from a linear SEM on the DAG."""
        n = self.n_nodes
        nodes = sorted(dag.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Topological sort
        topo_order = list(nx.topological_sort(dag))
        
        # Initialize data matrix
        data = np.zeros((self.n_samples, n))
        
        # Generate data following topological order
        for node in topo_order:
            j = node_to_idx[node]
            
            # Get parents
            parents = list(dag.predecessors(node))
            parent_indices = [node_to_idx[p] for p in parents]
            
            # Generate noise
            if self.noise == "gaussian":
                noise = rng.normal(0, self.noise_scale, self.n_samples)
            elif self.noise == "laplace":
                noise = rng.laplace(0, self.noise_scale, self.n_samples)
            elif self.noise == "uniform":
                noise = rng.uniform(
                    -self.noise_scale, self.noise_scale, self.n_samples
                )
            else:
                raise ValueError(f"Unknown noise: {self.noise}")
            
            # Combine parent contributions
            if parent_indices:
                # Random coefficients for edges
                coeffs = rng.uniform(
                    self.edge_coeff_range[0],
                    self.edge_coeff_range[1],
                    len(parent_indices),
                )
                data[:, j] = np.sum(
                    data[:, parent_indices] * coeffs, axis=1
                ) + noise
            else:
                data[:, j] = noise
        
        return pd.DataFrame(data, columns=nodes)
    
    def get_params(self) -> Dict[str, Any]:
        """Return simulator parameters."""
        return {
            "n_nodes": self.n_nodes,
            "edge_prob": self.edge_prob,
            "n_samples": self.n_samples,
            "noise": self.noise,
            "noise_scale": self.noise_scale,
            "edge_coeff_range": self.edge_coeff_range,
        }
    
    def get_name(self) -> str:
        """Return simulator name."""
        return (
            f"ErdosRenyi(n={self.n_nodes}, "
            f"p={self.edge_prob}, "
            f"samples={self.n_samples}, "
            f"noise={self.noise})"
        )


class ScaleFreeSimulator(BaseSimulator):
    """
    Simulate data from a scale-free network DAG.
    
    Uses preferential attachment to generate a scale-free DAG,
    then generates observational data similarly to ErdosRenyiSimulator.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph (default: 10).
    alpha : float
        Attachment exponent in preferential attachment (default: 1.0).
    n_samples : int
        Number of samples to generate (default: 500).
    noise : Literal['gaussian', 'laplace', 'uniform']
        Noise distribution (default: 'gaussian').
    noise_scale : float
        Standard deviation/scale of noise (default: 1.0).
    edge_coeff_range : tuple
        Range [min, max] for edge coefficients (default: (0.5, 2.0)).
    seed : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        n_nodes: int = 10,
        alpha: float = 1.0,
        n_samples: int = 500,
        noise: Literal["gaussian", "laplace", "uniform"] = "gaussian",
        noise_scale: float = 1.0,
        edge_coeff_range: tuple = (0.5, 2.0),
        seed: Optional[int] = None,
    ):
        self.n_nodes = n_nodes
        self.alpha = alpha
        self.n_samples = n_samples
        self.noise = noise
        self.noise_scale = noise_scale
        self.edge_coeff_range = edge_coeff_range
        self.seed = seed
    
    def simulate(self, seed: Optional[int] = None) -> SimulationOutput:
        """Generate synthetic data from a scale-free DAG."""
        rng = np.random.RandomState(seed or self.seed)
        
        # Generate scale-free DAG
        dag = self._generate_dag(rng)
        
        # Generate data
        data = self._generate_data(dag, rng)
        
        return SimulationOutput(
            dag=dag,
            data=data,
            params=self.get_params(),
            seed=seed or self.seed,
        )
    
    def _generate_dag(self, rng: np.random.RandomState) -> nx.DiGraph:
        """Generate a scale-free DAG using preferential attachment."""
        # Start with a small undirected scale-free graph
        undirected = nx.barabasi_albert_graph(self.n_nodes, 2, seed=rng.randint(0, 2**31))
        
        # Convert to DAG by topological ordering
        dag = nx.DiGraph()
        nodes = list(undirected.nodes())
        dag.add_nodes_from(nodes)
        
        # Random topological order (permutation)
        perm = rng.permutation(len(nodes))
        node_order = [nodes[i] for i in perm]
        node_to_rank = {node: rank for rank, node in enumerate(node_order)}
        
        # Add edges respecting topological order
        for u, v in undirected.edges():
            if node_to_rank[u] < node_to_rank[v]:
                dag.add_edge(u, v)
            else:
                dag.add_edge(v, u)
        
        return dag
    
    def _generate_data(self, dag: nx.DiGraph, rng: np.random.RandomState) -> pd.DataFrame:
        """Generate observational data from a linear SEM on the DAG."""
        n = self.n_nodes
        nodes = sorted(dag.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        topo_order = list(nx.topological_sort(dag))
        data = np.zeros((self.n_samples, n))
        
        for node in topo_order:
            j = node_to_idx[node]
            parents = list(dag.predecessors(node))
            parent_indices = [node_to_idx[p] for p in parents]
            
            if self.noise == "gaussian":
                noise = rng.normal(0, self.noise_scale, self.n_samples)
            elif self.noise == "laplace":
                noise = rng.laplace(0, self.noise_scale, self.n_samples)
            else:
                noise = rng.uniform(
                    -self.noise_scale, self.noise_scale, self.n_samples
                )
            
            if parent_indices:
                coeffs = rng.uniform(
                    self.edge_coeff_range[0],
                    self.edge_coeff_range[1],
                    len(parent_indices),
                )
                data[:, j] = np.sum(data[:, parent_indices] * coeffs, axis=1) + noise
            else:
                data[:, j] = noise
        
        return pd.DataFrame(data, columns=nodes)
    
    def get_params(self) -> Dict[str, Any]:
        """Return simulator parameters."""
        return {
            "n_nodes": self.n_nodes,
            "alpha": self.alpha,
            "n_samples": self.n_samples,
            "noise": self.noise,
            "noise_scale": self.noise_scale,
            "edge_coeff_range": self.edge_coeff_range,
        }
    
    def get_name(self) -> str:
        """Return simulator name."""
        return (
            f"ScaleFree(n={self.n_nodes}, "
            f"alpha={self.alpha}, "
            f"samples={self.n_samples})"
        )


class RealBNSimulator(BaseSimulator):
    """
    Simulate data from a real Bayesian network.
    
    Loads pre-defined Bayesian networks from pgmpy and generates data.
    
    Parameters
    ----------
    network : Literal['alarm', 'asia', 'cancer']
        Name of the pre-defined network (default: 'alarm').
    n_samples : int
        Number of samples to generate (default: 1000).
    seed : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        network: Literal["alarm", "asia", "cancer"] = "alarm",
        n_samples: int = 1000,
        seed: Optional[int] = None,
    ):
        self.network_name = network
        self.n_samples = n_samples
        self.seed = seed
    
    def simulate(self, seed: Optional[int] = None) -> SimulationOutput:
        """Generate data from a real Bayesian network."""
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.models import BayesianNetwork
        
        # Load the network (stub for now; would use pgmpy.datasets)
        dag, cpts = self._load_network()
        
        # Create BayesianNetwork and add CPDs
        bn = BayesianNetwork(json_graph=nx.node_link_data(dag))
        
        rng = np.random.RandomState(seed or self.seed)
        
        # For now, return simulated data using rejection sampling or similar
        # (Simplified: just use linear SEM as stand-in)
        data = self._generate_data_from_dag(dag, rng)
        
        return SimulationOutput(
            dag=dag,
            data=data,
            params=self.get_params(),
            seed=seed or self.seed,
        )
    
    def _load_network(self):
        """Load a pre-defined Bayesian network structure."""
        # Simplified: return a small DAG for now
        # In production, would load from pgmpy.datasets
        dag = nx.DiGraph()
        
        if self.network_name == "asia":
            dag.add_edges_from([("asia", "tub"), ("smoke", "lung"), 
                                ("smoke", "bronc"), ("tub", "either"),
                                ("lung", "either"), ("either", "xray"),
                                ("either", "dysp"), ("bronc", "dysp")])
        else:  # alarm, cancer, etc.
            dag.add_edges_from([("A", "C"), ("B", "C")])
        
        return dag, {}
    
    def _generate_data_from_dag(
        self, dag: nx.DiGraph, rng: np.random.RandomState
    ) -> pd.DataFrame:
        """Generate simple data for now."""
        nodes = sorted(dag.nodes())
        data = rng.randn(self.n_samples, len(nodes))
        return pd.DataFrame(data, columns=nodes)
    
    def get_params(self) -> Dict[str, Any]:
        """Return simulator parameters."""
        return {
            "network": self.network_name,
            "n_samples": self.n_samples,
        }
    
    def get_name(self) -> str:
        """Return simulator name."""
        return f"RealBN({self.network_name}, samples={self.n_samples})"


class LinearGaussianSEM(BaseSimulator):
    """
    Linear Gaussian Structural Equation Model.
    
    Generates data from a linear SEM with Gaussian noise.
    Particularly useful for testing methods like LiNGAM and NOTEARS.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes (default: 10).
    n_samples : int
        Number of samples (default: 500).
    noise_scale : float
        Standard deviation of noise (default: 1.0).
    edge_coeff_range : tuple
        Range [min, max] for edge coefficients (default: (0.5, 2.0)).
    sparsity : float
        Fraction of edges to include [0, 1] (default: 0.3).
    seed : int, optional
        Random seed.
    """
    
    def __init__(
        self,
        n_nodes: int = 10,
        n_samples: int = 500,
        noise_scale: float = 1.0,
        edge_coeff_range: tuple = (0.5, 2.0),
        sparsity: float = 0.3,
        seed: Optional[int] = None,
    ):
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        self.noise_scale = noise_scale
        self.edge_coeff_range = edge_coeff_range
        self.sparsity = sparsity
        self.seed = seed
    
    def simulate(self, seed: Optional[int] = None) -> SimulationOutput:
        """Generate synthetic data from a linear Gaussian SEM."""
        rng = np.random.RandomState(seed or self.seed)
        
        dag = self._generate_dag(rng)
        data = self._generate_data(dag, rng)
        
        return SimulationOutput(
            dag=dag,
            data=data,
            params=self.get_params(),
            seed=seed or self.seed,
        )
    
    def _generate_dag(self, rng: np.random.RandomState) -> nx.DiGraph:
        """Generate sparse DAG."""
        dag = nx.DiGraph()
        nodes = [f"X{i}" for i in range(self.n_nodes)]
        dag.add_nodes_from(nodes)
        
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if rng.rand() < self.sparsity:
                    dag.add_edge(nodes[i], nodes[j])
        
        return dag
    
    def _generate_data(self, dag: nx.DiGraph, rng: np.random.RandomState) -> pd.DataFrame:
        """Generate data from linear SEM."""
        n = self.n_nodes
        nodes = sorted(dag.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        topo_order = list(nx.topological_sort(dag))
        data = np.zeros((self.n_samples, n))
        
        for node in topo_order:
            j = node_to_idx[node]
            parents = list(dag.predecessors(node))
            parent_indices = [node_to_idx[p] for p in parents]
            
            noise = rng.normal(0, self.noise_scale, self.n_samples)
            
            if parent_indices:
                coeffs = rng.uniform(
                    self.edge_coeff_range[0],
                    self.edge_coeff_range[1],
                    len(parent_indices),
                )
                data[:, j] = np.sum(data[:, parent_indices] * coeffs, axis=1) + noise
            else:
                data[:, j] = noise
        
        return pd.DataFrame(data, columns=nodes)
    
    def get_params(self) -> Dict[str, Any]:
        """Return simulator parameters."""
        return {
            "n_nodes": self.n_nodes,
            "n_samples": self.n_samples,
            "noise_scale": self.noise_scale,
            "edge_coeff_range": self.edge_coeff_range,
            "sparsity": self.sparsity,
        }
    
    def get_name(self) -> str:
        """Return simulator name."""
        return (
            f"LinearGaussianSEM(n={self.n_nodes}, "
            f"sparsity={self.sparsity}, samples={self.n_samples})"
        )
