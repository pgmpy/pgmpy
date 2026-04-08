import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import logsigmoid
from torch.func import grad
from math import log

from typing import Callable, Tuple, List

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
        log_likelihood: Callable | str = "todo", # TODO
        learning_rate: float = 5e-3, # TODO: together with RMSProp schedule, or possibly adam.
        edge_prob_threshold: float = 0.1,
        kernel: str = "frobenius",
        kernel_bandwidth: float | str = "median",
        grad_estimator_z: str = "score",
        baseline: float = 0.0,
        alpha_linear: float = 0.05,
        beta_linear: float = 1.0,
        latent_dim: int = 32,
        n_grad_mc_samples: int = 128,
        n_acyclicity_mc_samples: int = 32,
        latent_prior_std: float = 1.0,
        tau: float = 1.0,
    ):
        self.n_particles = n_particles
        self.n_steps = n_steps
        self.log_likelihood = log_likelihood
        self.learning_rate = learning_rate
        self.edge_prob_threshold = edge_prob_threshold # Used for summarization of the graphs later.
        self.kernel = kernel
        self.kernel_bandwidth = kernel_bandwidth
        self.grad_estimator_z = grad_estimator_z
        self.baseline = baseline
        self.alpha = lambda t: alpha_linear * t
        self.beta = lambda t: beta_linear * t
        self.latent_dim = latent_dim # dimension of each U_i and V_i.
        self.n_grad_mc_samples = n_grad_mc_samples
        self.n_acyclicity_mc_samples = n_acyclicity_mc_samples
        self.latent_prior_std = latent_prior_std
        self.tau = tau
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")


    def _initialize_particles(
        self,
        n_nodes: int,
    ) -> torch.Tensor:
        n = self.n_particles

        # TODO: maybe divide by sqrt of latent_dim so that every row in U and V have unit variance.
        # last dim is self.latent_dim * 2 since Z = [U, V] and each U_i and V_i are of dim self.latent_dim.
        # Ignore acyclicity part in prior as no (efficient) sampler exists.
        return torch.randn((n, n_nodes, self.latent_dim * 2))


    def _grad_z_likelihood_score_function(
        self,
        X_t: torch.Tensor,
        particles: torch.Tensor,
        t: int,
    ):
        """
        Use eq. 14 to compute the ratio that is the second term in RHS of eq. 9:
        Parameters
        ----------
        X_t
        particles
        t

        Returns
        -------

        """
        n_samples = self.n_grad_mc_samples
        p, n_nodes, _ = particles.shape
        particles = particles.detach().requires_grad_(True)

        # Draw some hard graph samples:
        U, V = particles.chunk(2, dim=-1)
        soft_graphs = torch.sigmoid(self.alpha(t) * U @ V.transpose(-1, -2))
        hard_graph_samples = (torch.rand((p, n_samples, n_nodes, n_nodes), device=soft_graphs.device) < soft_graphs.unsqueeze(1)).type(torch.int32)

        # get rid of self-loops:
        hard_graph_samples = hard_graph_samples * (1.0 - torch.eye(soft_graphs.shape[-1], device=soft_graphs.device, dtype=soft_graphs.dtype))

        # compute the likelihood:
        if self.log_likelihood is None:
            # TODO: implement a likelihood function so that the user doesn't have to.
            self.log_likelihood = lambda x: x

        # TODO:
        log_likelihood_fn = self.log_likelihood
        log_likelihood = log_likelihood_fn(X_t, hard_graph_samples) # 2d tensor now.

        # function that computes log_p(G|Z), see eq. 6
        def log_p(G, Z):
            U, V = Z.chunk(2, dim=-1)
            scores = self.alpha(t) * (U @ V.transpose(-1, -2))
            mask = 1.0 - torch.eye(scores.shape[-1], device=scores.device, dtype=scores.dtype)

            G = G.to(scores.dtype)
            log_p1 = logsigmoid(scores) * mask
            log_p0 = logsigmoid(-scores) * mask
            return (G * log_p1 + (1 - G) * log_p0).sum(dim=(-1, -2))

        grad_log_p = grad(log_p, argnums=1)
        vectorized_grad_log_p = torch.vmap(
            torch.vmap(
                grad_log_p,
                in_dims=(0, None), # vectorize over the samples
            ),
            in_dims=(0, 0), # vectorize over the particles
        )

        # grad_z p(G|z) for each particle and each of the earlier samples: [particles.shape[0], S, *particles.shape[1:]]
        grads = vectorized_grad_log_p(hard_graph_samples, particles)

        # grad_z E_p(G|z)[ p(D|G) ], used for numerator in eq. 9
        weights = (log_likelihood.exp() - self.baseline)
        grad_z = (weights[..., None, None] * grads).mean(dim=1)

        # E_p(G|z)[ p(D|G) ], used for denominator in eq. 9
        expec_pgz = (torch.logsumexp(log_likelihood, dim=1) - log(n_samples)).exp()

        return grad_z / expec_pgz[:, None, None]



    def _grad_z_likelihood_gumbel(self):
        # TODO
        pass


    def _make_likelihood_grad_estimator(self, name: str):
        if name == "score":
            return self._grad_z_likelihood_score_function
        elif name == "reparam":
            return self._grad_z_likelihood_gumbel
        raise ValueError(f"Unknown grad estimator: {name}")


    def _compute_particle_posterior_scores(
        self,
        X_t: torch.Tensor,
        particles: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """
        # todo: computes grad_z log p(z|D) vectorized over the particles.
        Parameters
        ----------
        X
        particles
        t

        Returns
        -------

        """
        alpha, beta = self.alpha(t), self.beta(t)

        # compute eq. (A.34): grad_z log p(z) = -beta grad_z E_p(G|Z)[ h(G) ] - 1/sigma**2 Z:
        particles = particles.detach().requires_grad_(True)
        U, V = particles.chunk(2, dim=-1)

        graph_probs = torch.sigmoid(alpha * U @ V.transpose(-1, -2)) # compute the soft graph
        d = graph_probs.shape[-1]
        eye = torch.eye(d, device=graph_probs.device, dtype=graph_probs.dtype)
        graph_probs = graph_probs * (1.0 - eye.unsqueeze(0)) # with masked diagonal to prevent self loops

        # Compute the expectation:
        def h(G):
            d = G.shape[-1]
            id = torch.eye(d, device=G.device, dtype=G.dtype)
            return torch.trace(torch.linalg.matrix_power(id + G / d, d)) - d

        acyclicity_soft = torch.vmap(h)(graph_probs) # It is assumed E_p(G|Z)[ h(G) ] is approximated by h(G_soft)
        softgraph_score, = torch.autograd.grad(acyclicity_soft.sum(), particles)

        log_prior_score = -beta * softgraph_score - particles / (self.latent_prior_std ** 2)

        # Now compute the second term of eq. 9 by using eq 12.
        strategy = self._make_likelihood_grad_estimator(self.grad_estimator_z)
        ratio = strategy(X_t, particles, t) # second term in eq. 9

        score = log_prior_score + ratio
        return score

    def _get_kernel(
            self,
            particles: torch.Tensor,
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
        kernel_name = self.kernel
        bandwidth = self.kernel_bandwidth

        flat_particles = particles.reshape(particles.shape[0], -1)

        if bandwidth == "median":
            distances = torch.cdist(flat_particles, flat_particles) ** 2

            mask = ~torch.eye(
                distances.shape[0],
                dtype=torch.bool,
                device=distances.device,
            )
            bandwidth = torch.median(distances[mask]).clamp_min(1e-8)

        if kernel_name == "frobenius":
            interactions = flat_particles @ flat_particles.T
            self_interactions = interactions.diagonal()

            # using ||A-B||_F^2 = ||A||_F^2 + ||B||_F^2 - 2<A,B>_F
            norm_diff = (
                    self_interactions.unsqueeze(1)
                    + self_interactions.unsqueeze(0)
                    - 2 * interactions
            )

            def k(x, y):
                # flatten particles:
                x, y = x.flatten(), y.flatten()
                return torch.exp(- ((x - y) ** 2).sum() / bandwidth)

            return torch.exp(-norm_diff / bandwidth), k
        raise ValueError(f"Unknown kernel name: {kernel_name}")


    def _svgd_increment(
        self,
        scores: torch.Tensor,
        particles: torch.Tensor,
    ) -> torch.Tensor:
        M = particles.shape[0]
        kernel_mat, kernel_fn = self._get_kernel(particles)

        # from definition of phi in line 5 of algorithm 1.
        # https://arxiv.org/pdf/2105.11839

        # k(z_k, *) grad_{z_k} log p(z_k | D)
        # but also vectorized over m
        driving_term = torch.einsum('ij,i...->j...', kernel_mat, scores)

        # grad_{z_k} k(z_k, *)
        # also vectorized over m
        grad_first = grad(kernel_fn, argnums=0)

        # for some fixed particle z_m as the second argument, vectorize over the first argument (z_k)
        grad_over_k = torch.vmap(
            grad_first,
            in_dims=(0, None),
        )

        # vectorize over second argument (z_m)
        grad_over_m = torch.vmap(
            grad_over_k,
            in_dims=(None, 0),
        )
        repulsive_term = grad_over_m(particles, particles).sum(dim=1)

        return (driving_term + repulsive_term) / M



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
        X_t = torch.tensor(X.to_numpy(), device=self.device, dtype=torch.float32)
        n_nodes = X.shape[1]
        particles = torch.nn.Parameter(self._initialize_particles(n_nodes).to(self.device))
        optimizer = torch.optim.RMSprop([particles], lr=self.learning_rate, maximize=True)

        for t in range(self.n_steps):
            optimizer.zero_grad()
            # estimate score grad_Z log p(Z | D)
            scores = self._compute_particle_posterior_scores(X_t, particles, t).detach()

            # run svgd update: Z_new = Z_old + eta_t phi_t(Z_old)
            particles.grad = self._svgd_increment(scores, particles).detach()
            optimizer.step()

        # compute G_infty(Z):
        U, V = torch.chunk(particles.detach(), 2, dim=2)
        graphs_infty = ((U @ V.transpose(-1, -2)) > 0) * ~ torch.eye(n_nodes, dtype=torch.bool, device=self.device)
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