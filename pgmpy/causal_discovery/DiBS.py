import networkx as nx
import numpy as np
import pandas as pd
import torch

from typing import Callable, Tuple, List

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery


class DiBS(_BaseCausalDiscovery):
    """
    TODO: description and documentation here:
    Causal discovery using Differentiable Bayesian Structure Learning (DiBS).

    Detailed description here of how it works.

    Parameters
    ----------
    TODO: parameters here:
    example_param : type, default=None
        Small description of parameter. Supported instances:

        - Category 1: '...', '...'
        - Category 2: '...', '...'

    Attributes
    ----------
    TODO: attributes here similar to parameters:
    example_attribute : type, default=None

    Examples
    --------
    Simulate some data to use for causal discovery:

    TODO: put some example code here and the outputs

    References
    ----------
    .. [1] Lorch, Lars, Jonas Rothfuss, Bernhard Schölkopf, and Andreas
       Krause. "DiBS: Differentiable Bayesian Structure Learning."
       Advances in Neural Information Processing Systems, 2021.
    .. [2] Official implementation: larslorch/dibs, GitHub repository.
    """

    def __init__(
        self,
        n_particles: int = 30,
        n_steps: int = 2000,
        learning_rate: float = 5e-3,
        edge_prob_threshold: float = 0.1,
        kernel: str = "frobenius",
        kernel_bandwidth: float | str = "median",
        alpha_linear: float = 0.05,
        beta_linear: float = 1.0,
        latent_dim: int = 32,
    ):
        self.n_particles = n_particles
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.edge_prob_threshold = edge_prob_threshold # Used for summarization of the graphs later.
        self.kernel = kernel
        self.kernel_bandwidth = kernel_bandwidth
        self.alpha = lambda t: alpha_linear * t
        self.beta = lambda t: beta_linear * t
        self.latent_dim = latent_dim # dimension of each U_i and V_i.
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")

    def _initialize_particles(
        self,
        n_nodes: int,
    ) -> torch.Tensor:
        n = self.n_particles

        # TODO: maybe divide by sqrt of latent_dim so that every row in U and V have unit variance.
        # last dim is self.latent_dim * 2 since Z = [U, V] and each U_i and V_i are of dim self.latent_dim.
        # Ignore acyclicity part in prior as no (efficient) sampler exists.
        return torch.randn((n, n_nodes, self.latent_dim * 2))


    def _compute_particle_scores(
        self,
        X: pd.DataFrame,
        particles: torch.Tensor,
    ) -> torch.Tensor:
        pass


    def _get_kernel(
        self,
        particles: torch.Tensor,
    ):
        kernel_name = self.kernel
        bandwidth = self.kernel_bandwidth

        if bandwidth == "median":
            flat_particles = particles.view(particles.shape[0], -1)

            # compute all pairwise distances between particles
            distances = torch.cdist(flat_particles, flat_particles)

            # compute median distance between particles, excluding the diagonal
            bandwidth = torch.median(distances[~ torch.eye(distances.shape[0], dtype=bool)])

        if kernel_name == "frobenius":
            def frobenius_kernel(z1: torch.Tensor, z2: torch.Tensor) -> torch.float64:
                diff = torch.linalg.matrix_norm(z1 - z2, ord='fro') ** 2
                return torch.exp(- diff / bandwidth)
            return frobenius_kernel


    def _svgd_increment(
        self,
        scores: torch.Tensor,
        particles: torch.Tensor,
    ) -> torch.Tensor:
        kernel = self._get_kernel(particles)



    def _run_inference(
        self,
        X: pd.DataFrame,
    ):
        """
        Implements algorithm 1.

        Parameters
        ----------
        X

        Returns
        -------

        """
        X_t = torch.tensor(X.to_numpy(), device=self.device)
        n_nodes = X.shape[1]
        particles = self._initialize_particles(n_nodes).to(self.device)

        for t in range(self.n_steps):
            # estimate score grad_Z log p(Z | D)
            scores = self._compute_particle_scores(X_t, particles)

            # run svgd update: Z_new = Z_old + eta_t phi_t(Z_old)
            particles = torch.add(particles, self._svgd_increment(scores, particles))

        # compute G_infty(Z):
        U, V = torch.chunk(particles, 2, dim=2)
        graphs_infty = ((U @ V) > 0) * ~ torch.eye(n_nodes, dtype=bool).to(self.device)
        return graphs_infty


    def _sample_graphs(
        self,
        nodes,
    ):
        """
        Turn latent graphs into actual graphs.

        Parameters
        ----------
        nodes

        Returns
        -------

        """
        pass

    def _summarize_graphs(
        self,
        graph_samples,
    ):
        """
        Aggregate sampled graphs. Then use threshold for final summary graph.

        Parameters
        ----------
        graph_samples

        Returns
        -------

        """
        pass

    def _fit(self, X: pd.DataFrame):
        # TODO: Add logic to learn the causal graph from the data X. Methods from mixin classes can be used here if
        #       applicable.

        # TODO: After learning the causal graph, assign the learned graph to self.causal_graph_ attribute. Can be an
        #       instance of pgmpy.base.DAG, PDAG, MAG, PAG, or ADMG, depending on the algorithm or the hyperparameters.
        self.causal_graph_ = None

        # TODO: Additionally, assign the adjacency matrix of the learned graph to self.adjacency_matrix_ attribute.
        self.adjacency_matrix_ = None

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()

        return self