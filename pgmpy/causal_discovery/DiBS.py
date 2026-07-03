from __future__ import annotations
from collections.abc import Callable

import networkx as nx
import numpy as np
import pandas as pd

try:
    import torch
    from torch.func import grad
    from torch.nn.functional import logsigmoid

    _HAS_TORCH = True
except ImportError:
    torch = None
    grad = None
    logsigmoid = None
    _HAS_TORCH = False

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.global_vars import config


class DiBS(BaseCausalDiscovery):
    """
    Causal discovery using Differentiable Bayesian Structure Learning (DiBS).

    DiBS represents directed graphs through continuous latent variables. For each
    particle, every node has two latent vectors, commonly denoted U_i and V_i.
    The probability of an edge i -> j is parameterized by the sigmoid of the
    inner product U_i^T V_j, scaled by an annealing coefficient alpha(t). This
    allows Bayesian structure learning over directed graphs to be approximated
    with gradient-based inference in a continuous latent space.

    This implementation maintains multiple particles over latent graph
    representations and updates them using Stein Variational Gradient Descent
    (SVGD). The posterior score combines a graph likelihood term, an acyclicity
    prior term, and a Gaussian latent prior. After inference, each latent
    particle is converted into a hard graph using the limiting edge rule
    U_i^T V_j > 0. The resulting graph samples are summarized into marginal edge
    probabilities and a final acyclic summary graph.

    By default, the class uses a local linear-Gaussian Bayesian-network score:
    each variable is regressed on its selected parents, and the Gaussian
    log-likelihood is evaluated with maximum-likelihood estimates of the local
    regression coefficients and noise variances. A custom log-likelihood
    callable can be supplied for other data types or structural assumptions.

    Parameters
    ----------
    n_particles : int, default=30
        Number of SVGD particles used to approximate the posterior over latent
        graph representations. Larger values provide a richer posterior
        approximation but increase memory use and computation time.
    n_steps : int, default=2000
        Number of SVGD optimization steps.
    log_likelihood : Callable or None, default=None
        Custom graph log-likelihood function. The callable should accept
        ``(data, graph)`` and return a scalar or batched tensor of
        log-likelihood values. If None, a local linear-Gaussian likelihood is
        used.
    learning_rate : float, default=5e-3
        Learning rate used by the RMSprop optimizer for the SVGD particle
        updates.
    edge_prob_threshold : float, default=0.0
        Minimum posterior edge probability required for an edge to be considered
        during summary-graph construction.
    grad_estimator_z : {"score", "reparam"}, default="score"
        Gradient estimator used for the likelihood contribution. ``"score"``
        uses a score-function estimator; ``"reparam"`` uses a Gumbel-softmax /
        Concrete relaxation.
    baseline : float, default=0.0
        Optional baseline used in the score-function likelihood-gradient
        estimator.
    alpha_linear : float, default=0.05
        Linear coefficient for the edge-probability sharpness schedule
        ``alpha(t) = alpha_linear * (t + 1)``.
    beta_linear : float, default=1.0
        Linear coefficient for the acyclicity-penalty schedule
        ``beta(t) = beta_linear * (t + 1)``.
    latent_dim : int, default=32
        Dimension of each node-level latent vector U_i and V_i.
    n_grad_mc_samples : int, default=128
        Number of Monte Carlo graph samples used to estimate the likelihood
        gradient for each particle.
    n_acyclicity_mc_samples : int, default=32
        Number of Monte Carlo samples used to estimate the gradient of the
        expected acyclicity penalty.
    latent_prior_std : float, default=1.0
        Standard deviation of the isotropic Gaussian prior over latent
        variables.
    tau : float, default=1.0
        Temperature-like scale used in the Gumbel-softmax / Concrete graph
        relaxation.

    Attributes
    ----------
    causal_graph_ : nx.DiGraph
        Final summarized causal graph learned after calling ``fit``.
    edge_probs_ : pd.DataFrame
        Empirical posterior edge probabilities estimated from the final graph
        particles.
    adjacency_matrix_ : pd.DataFrame
        Binary adjacency matrix of ``causal_graph_``.
    n_features_in_ : int
        Number of variables in the fitted data.
    feature_names_in_ : list
        Column names of the fitted pandas DataFrame.
    device_ : torch.device
        Device used for tensor computations.
    _graph_particle_samples : torch.Tensor
        Hard adjacency matrices obtained from the final latent particles.

    Examples
    --------
    Simulate a three-variable linear causal chain and fit DiBS:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.causal_discovery.DiBS import DiBS
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> A = rng.normal(size=n)
    >>> B = 2.0 * A + rng.normal(scale=0.1, size=n)
    >>> C = -1.5 * B + rng.normal(scale=0.1, size=n)
    >>> X = pd.DataFrame({"A": A, "B": B, "C": C})
    >>> dibs = DiBS(n_particles=20, n_steps=50, edge_prob_threshold=0.5)
    >>> _ = dibs.fit(X)
    >>> dibs.adjacency_matrix_
    >>> dibs.edge_probs_

    The learned graph is available as a NetworkX directed graph:

    >>> dibs.causal_graph_.edges()

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
        edge_prob_threshold: float = 0.0,
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
        self.edge_prob_threshold = edge_prob_threshold  # Used for summarization of the graphs later.
        self.grad_estimator_z = grad_estimator_z
        self.baseline = baseline
        self.alpha_linear = alpha_linear
        self.beta_linear = beta_linear
        self.latent_dim = latent_dim  # dimension of each U_i and V_i.
        self.n_grad_mc_samples = n_grad_mc_samples
        self.n_acyclicity_mc_samples = n_acyclicity_mc_samples
        self.latent_prior_std = latent_prior_std
        self.tau = tau

    def _alpha(self, t: int):
        return self.alpha_linear * (t + 1)

    def _beta(self, t: int):
        return self.beta_linear * (t + 1)

    def _lgbn_log_likelihood(
        self,
        data: torch.Tensor,
        graph: torch.Tensor,
    ):
        """
        Compute local linear-Gaussian log-likelihood for one graph or a batch.

        Parameters
        ----------
        data : torch.Tensor
            Shape ``(n_samples, n_nodes)``.
        graph : torch.Tensor
            Shape ``(n_nodes, n_nodes)`` or ``(..., n_nodes, n_nodes)`` with
            convention ``graph[parent, child] = 1``.

        Returns
        -------
        torch.Tensor
            Scalar for a single graph; otherwise shape ``graph.shape[:-2]``.
        """
        n_samples, n_nodes = data.shape

        single_graph = graph.ndim == 2
        if single_graph:
            graph = graph.unsqueeze(0)

        batch_shape = graph.shape[:-2]
        graphs = graph.reshape(-1, n_nodes, n_nodes)

        data = data.to(graphs.device)
        dtype = data.dtype
        device = data.device

        # Ignore self-loops in local regressions.
        eye = torch.eye(n_nodes, device=device, dtype=graphs.dtype)
        graphs = graphs * (1 - eye)

        scores = []
        for g in graphs:
            total_ll = torch.zeros((), device=device, dtype=dtype)

            for child in range(n_nodes):
                parent_mask = g[:, child].bool()
                parent_mask[child] = False

                y = data[:, child]

                if parent_mask.any():
                    X_par = data[:, parent_mask]
                    design = torch.cat(
                        [torch.ones((n_samples, 1), device=device, dtype=dtype), X_par],
                        dim=1,
                    )
                else:
                    design = torch.ones((n_samples, 1), device=device, dtype=dtype)

                beta = torch.linalg.lstsq(design, y.unsqueeze(1)).solution
                resid = y.unsqueeze(1) - design @ beta
                rss = (resid.squeeze(1) ** 2).sum()

                sigma2 = (rss / n_samples).clamp_min(1e-8)
                local_ll = -0.5 * n_samples * (torch.log(2 * torch.pi * sigma2) + 1.0)
                total_ll = total_ll + local_ll

            scores.append(total_ll)

        scores = torch.stack(scores).reshape(batch_shape)
        return scores[0] if single_graph else scores

    def _grad_z_likelihood_score_function(
        self,
        X_t: torch.Tensor,
        particles: torch.Tensor,
        t: int,
    ):
        """
        Estimate the likelihood contribution to the latent posterior score.

        This method uses a score-function estimator for the gradient of the
        marginal graph likelihood with respect to the latent variables. For each
        particle, hard graphs are sampled from the Bernoulli edge distribution
        induced by the latent variables. The sampled graphs are scored by the graph
        log-likelihood, and the resulting normalized likelihood weights are used to
        average gradients of ``log p(G | Z)``.

        Parameters
        ----------
        X_t : torch.Tensor
            Data tensor of shape ``(n_samples, n_nodes)``.
        particles : torch.Tensor
            Current latent particles of shape
            ``(n_particles, n_nodes, 2 * latent_dim)``.
        t : int
            Current inference step, used by the alpha schedule.

        Returns
        -------
        torch.Tensor
            Estimated likelihood-gradient contribution with the same shape as
            ``particles``.
        """
        n_samples = self.n_grad_mc_samples
        p, n_nodes, _ = particles.shape
        particles = particles.detach().requires_grad_(True)

        # Draw some hard graph samples:
        U, V = particles.chunk(2, dim=-1)
        soft_graphs = torch.sigmoid(self._alpha(t) * U @ V.transpose(-1, -2))
        hard_graph_samples = (
            torch.rand((p, n_samples, n_nodes, n_nodes), device=soft_graphs.device) < soft_graphs.unsqueeze(1)
        ).to(soft_graphs.dtype)

        # get rid of self-loops:
        hard_graph_samples = hard_graph_samples * (
            1.0 - torch.eye(soft_graphs.shape[-1], device=soft_graphs.device, dtype=soft_graphs.dtype)
        )

        # compute the log likelihood:
        log_likelihood_fn = self._log_likelihood_fn
        ll = log_likelihood_fn(X_t, hard_graph_samples)  # 2d tensor now.

        # function that computes log_p(G|Z), see eq. 6
        def log_p(G, Z):
            U, V = Z.chunk(2, dim=-1)
            scores = self._alpha(t) * (U @ V.transpose(-1, -2))
            mask = 1.0 - torch.eye(scores.shape[-1], device=scores.device, dtype=scores.dtype)

            G = G.to(scores.dtype)
            log_p1 = logsigmoid(scores) * mask
            log_p0 = logsigmoid(-scores) * mask
            return (G * log_p1 + (1 - G) * log_p0).sum(dim=(-1, -2))

        vectorized_grad_log_p = torch.vmap(
            torch.vmap(
                grad(log_p, argnums=1),
                in_dims=(0, None),  # vectorize over the samples
            ),
            in_dims=(0, 0),  # vectorize over the particles
        )

        # grad_z log p(G|z) for each particle and each of the earlier samples:
        # # [particles.shape[0], S, *particles.shape[1:]]
        grads = vectorized_grad_log_p(hard_graph_samples, particles)

        # using a stable rewrite of the ratio in eq. 9 using eq. 14:
        weights = torch.softmax(ll, dim=1)
        first_term = (weights[..., None, None] * grads).sum(dim=1)

        b = self.baseline
        if b == 0:
            return first_term

        b = torch.tensor(b, device=grads.device, dtype=grads.dtype)
        second_term = (
            torch.sign(b)
            * torch.exp(torch.log(torch.abs(b)) - torch.logsumexp(ll, dim=1))[..., None, None]
            * grads.sum(dim=1)
        )

        return first_term - second_term

    def _grad_z_likelihood_gumbel(
        self,
        X_t: torch.Tensor,
        particles: torch.Tensor,
        t: int,
    ):
        """
        Estimate the likelihood contribution using a Gumbel-softmax relaxation.

        This method samples Logistic noise and constructs differentiable relaxed
        adjacency matrices using the Concrete/Gumbel-sigmoid transformation. The
        graph likelihood is evaluated on these relaxed graphs and differentiated
        with respect to the latent variables using automatic differentiation.

        Parameters
        ----------
        X_t : torch.Tensor
            Data tensor of shape ``(n_samples, n_nodes)``.
        particles : torch.Tensor
            Current latent particles of shape
            ``(n_particles, n_nodes, 2 * latent_dim)``.
        t : int
            Current inference step, used by the alpha schedule.

        Returns
        -------
        torch.Tensor
            Estimated likelihood-gradient contribution with the same shape as
            ``particles``.
        """

        # Use inverse transform sampling to get samples from logistic distribution with
        # location 0 and scale 1.
        # To this, end, use that the quantile function is given by Q(p) = log p / (1-p)
        # See https://en.wikipedia.org/wiki/Logistic_distribution#Quantile_function
        n_nodes = self.n_features_in_
        eps = torch.finfo(particles.dtype).eps
        uniform_samples = torch.rand(
            (particles.shape[0], self.n_grad_mc_samples, n_nodes, n_nodes),
            device=particles.device,
            dtype=particles.dtype,
        ).clamp(eps, 1 - eps)
        logistic_samples = torch.log(uniform_samples / (1 - uniform_samples))

        def graph_tau(L, Z):
            # equation 13:
            U, V = Z.chunk(2, dim=-1)
            interactions = U @ V.transpose(-1, -2)
            graph_taus = torch.sigmoid(self.tau * (L + self._alpha(t) * interactions))
            graph_taus = graph_taus * (1 - torch.eye(n_nodes, device=graph_taus.device, dtype=graph_taus.dtype))
            return graph_taus

        marginal_log_likelihood = lambda g: self._log_likelihood_fn(X_t, g)
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
        """
        Return the likelihood-gradient estimator specified by name.

        Parameters
        ----------
        name : {"score", "reparam"}
            Name of the gradient estimator.

        Returns
        -------
        Callable
            Method implementing the requested estimator.

        Raises
        ------
        ValueError
            If ``name`` is not a supported estimator.
        """
        if name == "score":
            return self._grad_z_likelihood_score_function
        elif name == "reparam":
            return self._grad_z_likelihood_gumbel

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
        alpha, beta = self._alpha(t), self._beta(t)

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
        grad_expected_h = torch.vmap(grad_constraint_gumbel, in_dims=0, randomness="different")(particles)
        log_prior_score = -beta * grad_expected_h - particles / (self.latent_prior_std**2)

        # Likelihood-ratio / reparam estimator for the likelihood term
        strategy = self._make_likelihood_grad_estimator(self.grad_estimator_z)
        ratio = strategy(X_t, particles, t)

        score = log_prior_score + ratio
        return score

    def _get_kernel(
        self,
        particles: torch.Tensor,
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
        """
        Compute the SVGD kernel matrix and corresponding pairwise kernel function.

        Parameters
        ----------
        particles : torch.Tensor
            Current latent particles of shape
            ``(n_particles, n_nodes, 2 * latent_dim)``.

        Returns
        -------
        kernel_mat : torch.Tensor
            Pairwise kernel matrix of shape ``(n_particles, n_particles)``.
        kernel_fn : Callable
            Function computing the kernel value between two individual particles.
        """

        flat_particles = particles.reshape(particles.shape[0], -1)

        distances = torch.cdist(flat_particles, flat_particles) ** 2

        mask = ~torch.eye(
            distances.shape[0],
            dtype=torch.bool,
            device=distances.device,
        )
        bandwidth = torch.median(distances[mask]).clamp_min(1e-8)

        interactions = flat_particles @ flat_particles.T
        self_interactions = interactions.diagonal()

        # using ||A-B||_F^2 = ||A||_F^2 + ||B||_F^2 - 2<A,B>_F
        norm_diff = self_interactions.unsqueeze(1) + self_interactions.unsqueeze(0) - 2 * interactions

        def k(x, y):
            # flatten particles:
            x, y = x.flatten(), y.flatten()
            return torch.exp(-((x - y) ** 2).sum() / bandwidth)

        return torch.exp(-norm_diff / bandwidth), k

    def _svgd_increment(
        self,
        scores: torch.Tensor,
        particles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the SVGD update direction for all particles.

        The update consists of a driving term, which moves particles toward regions
        of high posterior density, and a repulsive term, which encourages diversity
        among particles.

        Parameters
        ----------
        scores : torch.Tensor
            Estimated posterior score ``grad_Z log p(Z | D)`` for each particle.
        particles : torch.Tensor
            Current latent particles.

        Returns
        -------
        torch.Tensor
            SVGD update direction with the same shape as ``particles``.
        """

        M = particles.shape[0]
        kernel_mat, kernel_fn = self._get_kernel(particles)

        # from definition of phi in line 5 of algorithm 1.
        # https://arxiv.org/pdf/2105.11839

        # k(z_k, *) grad_{z_k} log p(z_k | D)
        # but also vectorized over m
        driving_term = torch.einsum("ij,i...->j...", kernel_mat, scores)

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

    def _fit(self, X: pd.DataFrame):
        """
        Fit the DiBS causal discovery model to observational data.

        This method runs latent-particle inference, converts the final particles
        into graph samples, summarizes those samples into posterior edge
        probabilities, and stores the final causal graph and adjacency matrix.

        Parameters
        ----------
        X : pd.DataFrame
            Observational data where columns are variables and rows are samples.

        Returns
        -------
        self : DiBS
            Fitted estimator.
        """

        if not _HAS_TORCH:
            raise ImportError(
                "DiBS requires the optional dependency 'torch'. "
                "Install with `pip install pgmpy[torch]` or `pip install torch`."
            )

        # Extended part of the __init__ to ensure sklearn backwards compatibility:
        self._log_likelihood_fn = self._lgbn_log_likelihood if self.log_likelihood is None else self.log_likelihood
        config.set_backend("torch")
        self.dtype_ = config.get_dtype()
        self.device_ = config.get_device()
        #################################################################################

        if self.grad_estimator_z not in ["score", "reparam"]:
            raise ValueError(f"Unknown grad estimator: {self.grad_estimator_z}. Must be one of ['score', 'reparam'].")

        # Run SVGD inference over latent graph particles.
        # The input data are converted to a torch tensor, latent particles are
        # initialized, and the particles are updated for ``n_steps`` iterations. After
        # optimization, each particle is converted into a hard adjacency matrix using
        # the limiting edge rule ``U_i^T V_j > 0`` with the diagonal set to zero.
        X_t = torch.tensor(X.to_numpy(dtype=np.float64), device=self.device_, dtype=self.dtype_)
        n_nodes = self.n_features_in_

        # initialize particles:
        # TODO: maybe divide by sqrt of latent_dim so that every row in U and V have unit variance.
        # last dim is self.latent_dim * 2 since Z = [U, V] and each U_i and V_i are of dim self.latent_dim.
        # Ignore acyclicity part in prior as no (efficient) sampler exists.
        particles = self.latent_prior_std * torch.randn((self.n_particles, n_nodes, self.latent_dim * 2))
        particles = torch.nn.Parameter(particles.to(self.device_, dtype=self.dtype_))

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
        graphs_infty = ((U @ V.transpose(-1, -2)) > 0) * ~torch.eye(n_nodes, dtype=torch.bool, device=self.device_)

        self._graph_particle_samples = graphs_infty.detach().cpu()

        # Convert stored particle adjacency matrices into NetworkX directed graphs.
        nodes = self.feature_names_in_
        graph_samples = self._graph_particle_samples.detach().cpu().numpy()
        sampled_graphs = []
        for adj in graph_samples:
            graph = nx.DiGraph()
            graph.add_nodes_from(nodes)
            src_idx, dst_idx = np.where(adj)
            graph.add_edges_from((nodes[i], nodes[j]) for i, j in zip(src_idx, dst_idx) if i != j)
            sampled_graphs.append(graph)

        self.graph_samples_ = sampled_graphs

        # Pick one sampled graph (temporary behavior), then coerce to a DAG.
        chosen_graph = self.graph_samples_[0] if len(self.graph_samples_) else nx.DiGraph()
        chosen_graph.add_nodes_from(nodes)

        # Greedy cycle removal by edge deletion order (simple temporary fallback).
        dag_graph = nx.DiGraph()
        dag_graph.add_nodes_from(chosen_graph.nodes())
        for u, v in chosen_graph.edges():
            if not nx.has_path(dag_graph, v, u):
                dag_graph.add_edge(u, v)

        self.causal_graph_ = DAG()
        self.causal_graph_.add_nodes_from(dag_graph.nodes())
        self.causal_graph_.add_edges_from(dag_graph.edges())

        self.adjacency_matrix_ = pd.DataFrame(
            nx.to_numpy_array(self.causal_graph_, nodelist=nodes, dtype=int, weight=None),
            index=nodes,
            columns=nodes,
        )
        self.edge_probs_ = self.adjacency_matrix_.astype(float)

        return self
