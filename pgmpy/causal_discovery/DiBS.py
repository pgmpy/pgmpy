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
        log_likelihood: Callable | None = None,
        learning_rate: float = 5e-3,
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
        if log_likelihood is None:
            self.log_likelihood = self._lgbn_log_likelihood
        else:
            self.log_likelihood = log_likelihood
        self.learning_rate = learning_rate
        self.edge_prob_threshold = edge_prob_threshold # Used for summarization of the graphs later.
        self.kernel = kernel
        self.kernel_bandwidth = kernel_bandwidth
        self.grad_estimator_z = grad_estimator_z
        self.baseline = baseline
        self.alpha = lambda t: alpha_linear * (t + 1)
        self.beta = lambda t: beta_linear * (t + 1)
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


    def _lgbn_log_likelihood(
        self,
        data: torch.Tensor,
        graph: torch.Tensor,
    ):
        """
        Compute a local linear-Gaussian log-likelihood score for one graph or a batch
        of graphs.

        Parameters
        ----------
        data : torch.Tensor
            Shape (n_samples, n_nodes). Each column is a variable and each row is an
            observation.
        graph : torch.Tensor
            Shape (n_nodes, n_nodes) or (batch..., n_nodes, n_nodes).
            Binary adjacency matrix with convention graph[parent, child] = 1.

        Returns
        -------
        torch.Tensor
            Scalar if a single graph is passed, else shape graph.shape[:-2].
        """
        if data.ndim != 2:
            raise ValueError(f"`data` must have shape (n_samples, n_nodes). Got {data.shape}")
        if graph.ndim < 2 or graph.shape[-1] != graph.shape[-2]:
            raise ValueError(f"`graph` must have shape (..., n_nodes, n_nodes). Got {graph.shape}")

        n_samples, n_nodes = data.shape
        if graph.shape[-1] != n_nodes:
            raise ValueError(
                f"Mismatch: data has {n_nodes} variables but graph has {graph.shape[-1]} nodes."
            )

        single_graph = graph.ndim == 2
        if single_graph:
            graph = graph.unsqueeze(0)

        batch_shape = graph.shape[:-2]
        graphs = graph.reshape(-1, n_nodes, n_nodes)

        data = data.to(graphs.device)
        dtype = data.dtype
        device = data.device

        # Prevent self loops from participating in the regressions.
        eye = torch.eye(n_nodes, device=device, dtype=graphs.dtype)
        graphs = graphs * (1 - eye)

        scores = []

        for g in graphs:
            total_ll = torch.zeros((), device=device, dtype=dtype)

            for child in range(n_nodes):
                # parents = {i : i -> child}
                parent_mask = g[:, child].bool()
                parent_mask[child] = False

                y = data[:, child]  # (n_samples,)

                if parent_mask.any():
                    X_par = data[:, parent_mask]  # (n_samples, n_parents)
                    design = torch.cat(
                        [torch.ones((n_samples, 1), device=device, dtype=dtype), X_par],
                        dim=1,
                    )
                else:
                    # Intercept-only model when there are no parents.
                    design = torch.ones((n_samples, 1), device=device, dtype=dtype)

                # Least-squares fit for the local linear Gaussian model.
                beta = torch.linalg.lstsq(design, y.unsqueeze(1)).solution  # (p+1, 1)
                resid = y.unsqueeze(1) - design @ beta  # (n_samples, 1)
                rss = (resid.squeeze(1) ** 2).sum()

                # MLE of Gaussian noise variance.
                sigma2 = (rss / n_samples).clamp_min(1e-8)

                # Local Gaussian log-likelihood with MLE plug-in.
                local_ll = -0.5 * n_samples * (torch.log(2 * torch.pi * sigma2) + 1.0)
                total_ll = total_ll + local_ll

            scores.append(total_ll)

        scores = torch.stack(scores).reshape(batch_shape)
        return scores[0] if single_graph else scores


    def _initialize_particles(
        self,
        n_nodes: int,
    ) -> torch.Tensor:
        n = self.n_particles

        # TODO: maybe divide by sqrt of latent_dim so that every row in U and V have unit variance.
        # last dim is self.latent_dim * 2 since Z = [U, V] and each U_i and V_i are of dim self.latent_dim.
        # Ignore acyclicity part in prior as no (efficient) sampler exists.
        return self.latent_prior_std * torch.randn((n, n_nodes, self.latent_dim * 2))


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
        hard_graph_samples = (
                torch.rand((p, n_samples, n_nodes, n_nodes), device=soft_graphs.device)
                < soft_graphs.unsqueeze(1)
        ).to(soft_graphs.dtype)

        # get rid of self-loops:
        hard_graph_samples = hard_graph_samples * (1.0 - torch.eye(soft_graphs.shape[-1], device=soft_graphs.device, dtype=soft_graphs.dtype))

        # compute the log likelihood:
        log_likelihood_fn = self.log_likelihood
        ll = log_likelihood_fn(X_t, hard_graph_samples) # 2d tensor now.

        # function that computes log_p(G|Z), see eq. 6
        def log_p(G, Z):
            U, V = Z.chunk(2, dim=-1)
            scores = self.alpha(t) * (U @ V.transpose(-1, -2))
            mask = 1.0 - torch.eye(scores.shape[-1], device=scores.device, dtype=scores.dtype)

            G = G.to(scores.dtype)
            log_p1 = logsigmoid(scores) * mask
            log_p0 = logsigmoid(-scores) * mask
            return (G * log_p1 + (1 - G) * log_p0).sum(dim=(-1, -2))

        vectorized_grad_log_p = torch.vmap(
            torch.vmap(
                grad(log_p, argnums=1),
                in_dims=(0, None), # vectorize over the samples
            ),
            in_dims=(0, 0), # vectorize over the particles
        )

        # grad_z log p(G|z) for each particle and each of the earlier samples: [particles.shape[0], S, *particles.shape[1:]]
        grads = vectorized_grad_log_p(hard_graph_samples, particles)

        # using a stable rewrite of the ratio in eq. 9 using eq. 14:
        weights = torch.softmax(ll, dim=1)
        first_term = (weights[..., None, None] * grads).sum(dim=1)

        b = self.baseline
        if b == 0:
            return first_term

        b = torch.tensor(b, device=grads.device, dtype=grads.dtype)
        second_term = torch.sign(b) * torch.exp(torch.log(torch.abs(b)) - torch.logsumexp(ll, dim=1))[..., None, None] * grads.sum(dim=1)
        return first_term - second_term


    def _grad_z_likelihood_gumbel(
        self,
        X_t: torch.Tensor,
        particles: torch.Tensor,
        t: int,
    ):
        """
        # todo: implementation of eq. 12.
        Parameters
        ----------
        X_t
        particles
        t

        Returns
        -------

        """

        # Use inverse transform sampling to get samples from logistic distribution with
        # location 0 and scale 1.
        # To this, end, use that the quantile function is given by Q(p) = log p / (1-p)
        # See https://en.wikipedia.org/wiki/Logistic_distribution#Quantile_function
        n_nodes = X_t.shape[1]
        eps = torch.finfo(particles.dtype).eps
        uniform_samples = torch.rand(
            (particles.shape[0], self.n_grad_mc_samples, n_nodes, n_nodes),
            device=particles.device,
            dtype=particles.dtype,
        ).clamp(eps, 1 - eps)
        logistic_samples = torch.log(uniform_samples / (1 - uniform_samples))


        # todo: discuss correctness of eq. 12 with ankur; is the chain rule applied correctly?
        # todo: for the time being, use a custom stable rewrite and use autodiff.

        def graph_tau(L, Z):
            # equation 13:
            U, V = Z.chunk(2, dim=-1)
            interactions = U @ V.transpose(-1, -2)
            graph_taus = torch.sigmoid(self.tau * (L + self.alpha(t) * interactions))
            graph_taus = graph_taus * (1 - torch.eye(n_nodes, device=graph_taus.device, dtype=graph_taus.dtype))
            return graph_taus

        marginal_log_likelihood = lambda g: self.log_likelihood(X_t, g)
        composition = lambda l, z: marginal_log_likelihood(graph_tau(l, z))

        # using grad_z f = f * grad_z log f:
        # grad_z marginal_log_likelihood(G_tau(L, Z))
        gradz_marg_ll = torch.vmap(
            torch.vmap(grad(composition, argnums=1), in_dims=(0, None)),
            in_dims=(0, 0),
        )(logistic_samples, particles)

        # log_likelihood(G_tau(L, Z))
        log_marg_likelihoods = torch.vmap(
            torch.vmap(composition, in_dims=(0, None)),
            in_dims=(0, 0),
        )(logistic_samples, particles)

        # Stable computation of
        #   [sum_m exp(log_marg_likelihoods_m) * gradz_marg_ll_m] /
        #   [sum_m exp(log_marg_likelihoods_m)]
        weights = torch.softmax(log_marg_likelihoods, dim=1)
        ratio = (weights[..., None, None] * gradz_marg_ll).sum(dim=1)

        return ratio


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
        Computes grad_Z log p(Z | D) for all particles.

        This matches the authors' DiBS prior-gradient structure more closely:
            grad_Z log p(Z)
            = - beta(t) * grad_Z E_{p(G|Z)}[ h(G) ]
              - Z / sigma_z^2
              [+ optional extra graph-prior term, if implemented]

        For the acyclicity term, use a Gumbel-softmax / Concrete
        reparameterization estimator with `n_acyclicity_mc_samples`.
        """
        alpha, beta = self.alpha(t), self.beta(t)

        particles = particles.detach().requires_grad_(True)
        p, d, _ = particles.shape
        device = particles.device
        dtype = particles.dtype

        eye = torch.eye(d, device=device, dtype=dtype)

        def h(G: torch.Tensor) -> torch.Tensor:
            """
            Acyclicity surrogate:
                tr((I + G / d)^d) - d

            Supports G of shape (..., d, d).
            Returns shape (...,).
            """
            I = torch.eye(d, device=G.device, dtype=G.dtype)
            M = I + G / d

            flat_M = M.reshape(-1, d, d)
            vals = torch.stack(
                [torch.trace(torch.linalg.matrix_power(A, d)) - d for A in flat_M],
                dim=0,
            )
            return vals.reshape(G.shape[:-2])

        def soft_graph_from_latent(Z: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
            """
            Gumbel-softmax / Concrete sample:
                sigmoid(tau * (eps + alpha(t) * U V^T))
            with zero diagonal.
            Z:   (d, 2k)
            eps: (d, d)
            returns: (d, d)
            """
            U, V = Z.chunk(2, dim=-1)  # (d, k), (d, k)
            scores = U @ V.transpose(-1, -2)  # (d, d)
            G_soft = torch.sigmoid(self.tau * (eps + alpha * scores))
            return G_soft * (1.0 - eye)

        def constraint_gumbel(single_z: torch.Tensor, single_eps: torch.Tensor) -> torch.Tensor:
            """
            h(G_tau(eps, z))
            """
            G_soft = soft_graph_from_latent(single_z, single_eps)
            return h(G_soft)

        def grad_constraint_gumbel(single_z: torch.Tensor) -> torch.Tensor:
            """
            Reparameterization estimator for
                grad_Z E_{p(G|Z)} [ h(G) ]
            using Logistic(0,1) noise and n_acyclicity_mc_samples MC samples.
            """
            eps_u = torch.rand(
                (self.n_acyclicity_mc_samples, d, d),
                device=device,
                dtype=dtype,
            )
            finfo = torch.finfo(dtype)
            eps_u = eps_u.clamp(min=finfo.eps, max=1.0 - finfo.eps)
            eps = torch.log(eps_u) - torch.log1p(-eps_u)  # Logistic(0,1)

            grad_fn = grad(constraint_gumbel, argnums=0)
            mc_grads = torch.vmap(grad_fn, in_dims=(None, 0))(single_z, eps)
            return mc_grads.mean(dim=0)

        # Batched prior score:
        #   - beta * grad E[h(G)] - Z / sigma^2
        grad_expected_h = torch.vmap(grad_constraint_gumbel, in_dims=0)(particles)
        log_prior_score = -beta * grad_expected_h - particles / (self.latent_prior_std ** 2)

        # Likelihood-ratio / reparam estimator for the likelihood term
        strategy = self._make_likelihood_grad_estimator(self.grad_estimator_z)
        ratio = strategy(X_t, particles, t)

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

        self._graph_particle_samples = graphs_infty.detach().cpu()

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