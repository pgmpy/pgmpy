from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Generator, Hashable
from itertools import combinations, permutations

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skbase.utils.dependencies import _safe_import
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    adjusted_mutual_info_score,
    mutual_info_score,
    normalized_mutual_info_score,
)
from sklearn.utils.validation import check_is_fitted, validate_data
from tqdm.auto import tqdm

from pgmpy import config, logger
from pgmpy.base import DAG, UndirectedGraph
from pgmpy.ci_tests import IndependenceMatch, get_ci_test
from pgmpy.independencies import Independencies
from pgmpy.metrics import get_metrics
from pgmpy.structure_score import BaseStructureScore

torch = _safe_import("torch")


class BaseCausalDiscovery(BaseEstimator):
    """
    Base class for all causal discovery estimators in pgmpy.

    Sets the sklearn tags and defines a method to check the input data for fitting.
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.categorical = True
        tags.input_tags.allow_nan = False
        tags.input_tags.positive_only = False
        tags.target_tags.required = False
        return tags

    def _check_fit_data(self, X):
        """Check the input data for fitting the causal discovery algorithm.

        Parameters
        ----------
        X: pd.DataFrame
            The data to fit the causal discovery algorithm on.
        """
        n_samples, n_features = X.shape

        if n_features == 0:
            raise ValueError(f"0 feature(s) (shape={X.shape}) while a minimum of 1 is required.")
        if n_samples < 2:
            raise ValueError(f"n_samples = {n_samples}, at least 2 are required.")

        # Handle cases like complex data, sparse arrays etc. first
        validate_data(
            self,
            X=X,
            dtype=None,
            accept_sparse=False,
            ensure_all_finite=True,
            reset=True,
        )

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
            self.feature_names_in_ = X.columns

        if not all([isinstance(x, Hashable) for x in X.values.flat]):
            raise TypeError("argument must be a string, number, or hashable object.")

        self.n_features_in_ = len(X.columns)
        return X

    def fit(self, X: pd.DataFrame, y=None):
        """Fit data (`X`) to a causal graph. The method
        calls the `_fit` method, which must be implemented separately in any causal
        discovery algorithm inheriting from `BaseCausalDiscovery`.
        """
        X = self._check_fit_data(X)
        return self._fit(X)

    def score(
        self,
        X=None,
        true_graph=None,
        metric=None,
    ):
        """
        Method to calculate the score of the fitted causal graph.

        The score can be calculated either against a dataset (`X`) or against a ground truth model (`true_graph`).
        Hence, only one of the two parameters should be provided. Depending on whether `X` is provided or
        `true_graph`, the `metric` should be chosen accordingly.

        Parameters
        ----------
        X : pandas.DataFrame, optional
            Test data used for scoring the learned causal model. If provided, `metric` should be a metric that
            can operate on data. You can find all such metrics using: `pgmpy.metrics.get_metrics(requires_data=True)`

        true_graph : pgmpy.base.DAG, optional
            The true model graph for scoring the learned causal model. If provided, `metric` should be a metric
            that compares graphs. You can find all such metrics using:
            `pgmpy.metrics.get_metrics(requires_true_graph=True)`

        metric : str or pgmpy.metrics._Base.*Metric instance, optional
            Method to be used for calculating the score. If ``None``, a default metric appropriate for the
            provided argument (`X` or `true_graph`) will be selected internally.

        Returns
        -------
        score : float or other type
            The calculated score of the learned causal graph according to the specified scoring method. The exact
            return type depends on the chosen metric and may be a float, pandas.DataFrame, tuple, or another
            metric-specific type.

        Examples
        --------
        >>> from pgmpy.causal_discovery import PC
        >>> from pgmpy.metrics import get_metrics
        >>> from pgmpy.datasets import load_dataset
        >>> dataset = load_dataset("lead")
        >>> data = dataset.data
        >>> dag = PC(return_type="dag").fit(data)
        >>> score = dag.score(X=data, metric="correlation_score")
        """
        check_is_fitted(self, "causal_graph_")

        # Case 1: When data is provided.
        if X is not None:
            validate_data(
                self,
                X=X,
                dtype=None,
                accept_sparse=False,
                ensure_all_finite=True,
                reset=False,
            )
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])

            if metric is None:
                scoring_class = get_metrics(requires_data=True, is_default=True)[0]
                metric = scoring_class()

            elif isinstance(metric, str):
                scoring_class = get_metrics(name=metric)
                if len(scoring_class) == 0:
                    raise ValueError(f"No scoring method found with name: {metric}")

                metric = scoring_class[0]()

            return metric.evaluate(X, self.causal_graph_)

        # Case 2: When true graph is provided.
        elif true_graph is not None:
            if metric is None:
                scoring_class = get_metrics(requires_true_graph=True, is_default=True)
                metric = scoring_class[0]()
            elif isinstance(metric, str):
                scoring_class = get_metrics(name=metric)
                if len(scoring_class) == 0:
                    raise ValueError(f"No scoring method found with name: {metric}")

                metric = scoring_class[0]()

            return metric.evaluate(true_causal_graph=true_graph, est_causal_graph=self.causal_graph_)
        else:
            raise ValueError("Either `X` or `true_graph` needs to be specified")


class _BaseDAGMAMixin:
    """
    Mixin class implementing shared acyclicity constraint, optimization, and graph reconstruction logic for DAGMA and
    its variants.
    """

    def _resolve_device_and_dtype(self) -> tuple:
        """
        Queries the global pgmpy configurations to resolve the PyTorch device and tensor float precision (mapping
        string representations to torch.dtype objects).

        If the global backend is ``"numpy"``, it is automatically switched to ``"torch"`` (DAGMA requires PyTorch).
        """
        if config.get_backend() == "numpy":
            config.set_backend("torch")
        device = config.get_device()
        dtype_str = config.get_dtype()
        if isinstance(dtype_str, str):
            dtype = getattr(torch, dtype_str)
        else:
            dtype = dtype_str
        return device, dtype

    def _log_det_barrier(self, W, s: float):
        """
        Computes the log-determinant acyclicity barrier function:
        h(W) = -log det(sI - W o W) + d log s

        Returns
        -------
        is_cyclic : bool
            True if the matrix violates the M-matrix domain (sign <= 0).
        h : torch.Tensor or None
            The computed barrier value if acyclic, otherwise None.
        """
        d = W.shape[0]
        I = torch.eye(d, device=W.device, dtype=W.dtype)
        M = s * I - W * W

        sign, logdet = torch.slogdet(M)
        if sign <= 0:
            return True, None

        h = -logdet + d * math.log(s)
        return False, h

    def _convert_to_dag(self, W: np.ndarray, feature_names: list, w_threshold: float, return_type: str):
        """
        Thresholds the estimated weight matrix and converts it into a pgmpy DAG or CPDAG.
        """
        W_thresh = np.where(np.abs(W) > w_threshold, W, 0)
        # Zero out diagonal entries — self-loops are never part of a DAG
        np.fill_diagonal(W_thresh, 0)
        dag = nx.from_pandas_adjacency(
            pd.DataFrame(W_thresh, index=feature_names, columns=feature_names),
            create_using=nx.DiGraph,
        )

        # Break any residual cycles by removing the smallest-|weight| edge in each cycle.
        # DAGMANonlinear can produce small numerical cycles at low iteration counts;
        # DAGMALinear with proper convergence should not trigger this path. The guard
        # is kept here (shared mixin) because both variants call _convert_to_dag.
        while True:
            try:
                cycle = nx.find_cycle(dag)
            except nx.NetworkXNoCycle:
                break
            # Remove the edge with the smallest absolute weight in the cycle
            min_edge = min(cycle, key=lambda e: abs(W_thresh[feature_names.index(e[0]), feature_names.index(e[1])]))
            dag.remove_edge(*min_edge)

        if return_type == "dag":
            return DAG(dag)
        elif return_type == "cpdag":
            return DAG(dag).to_pdag()
        else:
            raise ValueError(f"return_type must be 'dag' or 'cpdag', got {return_type}")

    def _optimize(
        self,
        W_tensor,
        optimizer_cls,
        optimizer_kwargs: dict,
        objective_fn,
        mu_init: float,
        mu_factor: float,
        max_iter: int,
        inner_iter: int = 1,
        gradient_fn=None,
        s_schedule=None,
        warm_iter=None,
        lr=None,
        tol=1e-6,
        checkpoint=1000,
    ) -> np.ndarray:
        """
        Unified optimization loop executing the dual-loop DAGMA optimization
        with domain-violation recovery and optional analytical gradients.

        When ``gradient_fn`` is provided, gradients are computed analytically and
        injected into ``W_param.grad`` — eliminating autograd backward pass overhead.
        Any ``torch.optim.Optimizer`` works because all optimizers read ``param.grad``
        the same way.

        When ``gradient_fn`` is None, falls back to autograd (``loss.backward()``).

        Domain-violation recovery (two-level):
        - **Inner level:** When a step violates the M-matrix domain, the bad step is
          undone, learning rate is halved, and the step is re-applied with the smaller
          lr. If lr drops too low, signals failure to the outer level.
        - **Outer level:** When the inner loop fails, W is rolled back to the
          pre-iteration checkpoint, lr is halved, and s is loosened (+0.1).
          Retries until success or lr < 1e-16 (gives up gracefully).

        Parameters
        ----------
        W_tensor : torch.Tensor
            Initial weight matrix.
        optimizer_cls : type
            Uninstantiated PyTorch optimizer class.
        optimizer_kwargs : dict
            Keyword arguments for the optimizer constructor.
        objective_fn : callable
            ``fn(W, mu, s) -> loss`` — computes scalar loss (with or without graph).
        mu_init : float
            Initial central path parameter.
        mu_factor : float
            Decay factor for mu (mu *= mu_factor each outer iteration).
        max_iter : int
            Number of outer iterations (T).
        inner_iter : int, optional (default=1)
            Number of inner optimization steps per outer iteration.
            When ``warm_iter`` is provided, this applies only to the final stage.
        gradient_fn : callable or None, optional (default=None)
            ``fn(W, mu, s) -> (grad, is_valid)`` — analytical gradient + domain check.
            When None, autograd ``loss.backward()`` is used.
        s_schedule : list of float or None, optional (default=None)
            s values per outer iteration. If None, defaults to ``[1.0] * max_iter``.
        warm_iter : int or None, optional (default=None)
            Inner steps for non-final outer iterations. If None, ``inner_iter`` is
            used for all stages (no warm/final split).
        lr : float or None, optional (default=None)
            Initial learning rate for retry halving. If None, extracted from
            ``optimizer_kwargs['lr']`` or defaults to 0.0003.
        tol : float, optional (default=1e-6)
            Relative tolerance for convergence early-stop.
        checkpoint : int, optional (default=1000)
            Frequency (in inner steps) of convergence checks.

        Returns
        -------
        np.ndarray
            Optimized weight matrix.
        """
        mu = mu_init
        W_current = W_tensor.detach().clone()
        use_analytical = gradient_fn is not None

        # Resolve learning rate for retry halving
        if lr is None:
            lr = optimizer_kwargs.get("lr", 0.0003)
        lr_initial = lr

        # Resolve s-schedule
        if s_schedule is None:
            s_schedule = [1.0] * max_iter
        elif len(s_schedule) < max_iter:
            s_schedule = list(s_schedule) + [s_schedule[-1]] * (max_iter - len(s_schedule))

        # Resolve warm/final iteration split
        _warm_iter = warm_iter if warm_iter is not None else inner_iter
        _final_iter = inner_iter

        # Detect L-BFGS — requires step(closure) API
        is_lbfgs = optimizer_cls is torch.optim.LBFGS

        for t in range(max_iter):
            s = s_schedule[t]
            inner_iters = _final_iter if t == max_iter - 1 else _warm_iter
            lr_current = lr_initial
            success = False

            while not success:
                # Checkpoint W before this attempt (for outer-level rollback)
                W_checkpoint = W_current.clone()

                W_param = torch.nn.Parameter(W_current.clone().requires_grad_(not use_analytical))
                opt_kw = dict(optimizer_kwargs)
                opt_kw["lr"] = lr_current
                optimizer = optimizer_cls([W_param], **opt_kw)

                obj_prev = 1e16
                success = True  # assume success unless proven otherwise

                for i in range(1, inner_iters + 1):
                    if use_analytical:
                        # --- Analytical gradient path ---
                        with torch.no_grad():
                            grad, is_valid = gradient_fn(W_param.data, mu, s)

                        if not is_valid:
                            # Domain violation: M-matrix has negative inv entries
                            success = False
                            break

                        W_param.grad = grad
                        optimizer.step()

                        # Convergence check (loss computed without graph)
                        if i % checkpoint == 0 or i == inner_iters:
                            with torch.no_grad():
                                obj_new = objective_fn(W_param.data, mu, s).item()
                            if abs((obj_prev - obj_new) / (abs(obj_prev) + 1e-16)) <= tol:
                                break
                            obj_prev = obj_new

                    elif is_lbfgs:
                        # --- L-BFGS autograd path (step-with-closure API) ---
                        def closure():
                            optimizer.zero_grad()
                            loss = objective_fn(W_param, mu, s)
                            if loss.item() >= 1e9:  # barrier violation sentinel
                                return loss
                            loss.backward()
                            return loss

                        loss = optimizer.step(closure)
                        if loss.item() >= 1e9:
                            success = False
                            break

                        if i % checkpoint == 0 or i == inner_iters:
                            obj_new = loss.item()
                            if abs((obj_prev - obj_new) / (abs(obj_prev) + 1e-16)) <= tol:
                                break
                            obj_prev = obj_new

                    else:
                        # --- Standard autograd path (Adam, SGD, etc.) ---
                        optimizer.zero_grad()
                        loss = objective_fn(W_param, mu, s)

                        if loss.item() >= 1e9:  # barrier violation sentinel
                            success = False
                            break

                        loss.backward()
                        optimizer.step()

                        # Convergence check
                        if i % checkpoint == 0 or i == inner_iters:
                            obj_new = loss.item()
                            if abs((obj_prev - obj_new) / (abs(obj_prev) + 1e-16)) <= tol:
                                break
                            obj_prev = obj_new

                if not success:
                    # Outer-level recovery: rollback W, halve lr, loosen s
                    W_current = W_checkpoint
                    lr_current *= 0.5
                    s = s + 0.1
                    s_schedule[t] = s
                    if lr_current < 1e-16:
                        # Give up gracefully — accept current checkpoint
                        success = True
                else:
                    W_current = W_param.detach().clone()

            mu *= mu_factor

        return W_current.cpu().numpy()


class _ConstraintMixin:
    """
    Base class for all constraint-based causal discovery estimators.
    """

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        independencies: Independencies = None,
    ):
        """Fit data (`X`) and independence relations (optional) to a causal graph. The method
        calls the `_fit` method, which must be implemented separately in any causal
        discovery algorithm inheriting from `BaseConstraintCausalDiscovery`.
        """
        X = self._check_fit_data(X)
        return self._fit(X, independencies)

    def _build_skeleton(
        self,
        data,
        independencies=None,
        variant: str = "stable",
        ci_test: str | Callable | None = None,
        significance_level: float = 0.01,
        max_cond_vars: int = 5,
        expert_knowledge=None,
        n_jobs: int = -1,
        show_progress: bool = True,
        **kwargs,
    ) -> tuple[UndirectedGraph, dict[tuple[str, str], set[str]]]:
        """
        Estimates a graph skeleton (UndirectedGraph) from a set of independencies
        using (the first part of) the PC algorithm.

        The independencies can either be provided as an instance of the
        `Independencies`-class or by passing a decision function that decides any
        conditional independency assertion. Returns a tuple `(skeleton, separating_sets)`.

        If an Independencies-instance is passed, the contained IndependenceAssertions
        have to admit a faithful BN representation. This is the case if
        they are obtained as a set of d-separations of some Bayesian network or
        if the independence assertions are closed under the semi-graphoid axioms.
        Otherwise, the procedure may fail to identify the correct structure.

        Parameters
        ----------
        variant: str (one of "orig", "stable", "parallel")
            The variant of PC algorithm to run.
                "orig": The original PC algorithm. Might not give the same
                        results in different runs but does less independence
                        tests compared to stable.
                "stable": Gives the same result in every run but does needs to
                        do more statistical independence tests.
                "parallel": Parallel version of PC Stable. Can run on multiple
                        cores with the same result on each run.

        ci_test: str or fun
            The statistical test to use for testing conditional independence in
            the dataset. If `str` values should be one of:
                "independence_match": If using this option, an additional parameter
                        `independencies` must be specified.
                "chi_square": Uses the Chi-Square independence test. This works
                        only for discrete datasets.
                "pearsonr": Uses the partial correlation based on pearson
                        correlation coefficient to test independence. This works
                        only for continuous datasets.
                "g_sq": G-test. Works only for discrete datasets.
                "log_likelihood": Log-likelihood test. Works only for discrete dataset.
                "freeman_tuckey": Freeman Tuckey test. Works only for discrete dataset.
                "modified_log_likelihood": Modified Log Likelihood test. Works only for discrete variables.
                "neyman": Neyman test. Works only for discrete variables.
                "cressie_read": Cressie Read test. Works only for discrete variables.

        significance_level: float (default: 0.01)
            The statistical tests use this value to compare with the p-value of
            the test to decide whether the tested variables are independent or
            not. Different tests can treat this parameter differently:
                1. Chi-Square: If p-value > significance_level, it assumes that the
                    independence condition satisfied in the data.
                2. pearsonr: If p-value > significance_level, it assumes that the
                    independence condition satisfied in the data.

        max_cond_vars: int (default: 5)
            The maximum number of variables to condition on while testing
            independence.

        expert_knowledge: pgmpy.causal_discovery.ExpertKnowledge instance
            Expert knowledge to be used with the algorithm.

        n_jobs: int (default: -1)
            The number of jobs to run in parallel.

        show_progress: bool (default: True)
            If True, shows a progress bar while running the algorithm.


        Returns
        -------
        skeleton: UndirectedGraph
            An estimate for the undirected graph skeleton of the BN underlying the data.

        separating_sets: dict
            A dict containing for each pair of not directly connected nodes a
            separating set ("witnessing set") of variables that makes them
            conditionally independent. (needed for edge orientation procedures)

        References
        ----------
        - :footcite:t:`neapolitan_2009` (Section 10.1.2, Algorithm 10.2, page 550).
        - :footcite:t:`koller_friedman_2009` (Section 3.4.2.1, page 85, Algorithm 3.3).
        """
        # Initialize initial values and structures.
        lim_neighbors = 0
        separating_sets = dict()
        if independencies is not None:
            ci_test = IndependenceMatch(independencies=independencies)
        else:
            ci_test = get_ci_test(test=ci_test, data=data)

        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(total=max_cond_vars)
            pbar.set_description("Working for n conditional variables: 0")

        if variant == "parallel":
            parallel_pool = Parallel(n_jobs=n_jobs, prefer="threads")

        variables = list(data.columns.values)

        # Step 1: Initialize a fully connected undirected graph
        graph = nx.complete_graph(n=variables, create_using=nx.Graph)
        if expert_knowledge is None:
            temporal_ordering, required_edges, forbidden_edges = {}, set(), set()
        else:
            temporal_ordering = expert_knowledge.temporal_ordering_
            required_edges = expert_knowledge.required_edges_
            forbidden_edges = expert_knowledge.forbidden_edges_

        # Remove edges that are forbidden in both directions. Directed forbidden are enforced as orientations after the
        # skeleton is learned.
        graph.remove_edges_from([(u, v) for (u, v) in forbidden_edges if (v, u) in forbidden_edges])

        # Exit condition: 1. If all the nodes in graph has less than `lim_neighbors` neighbors.
        #             or  2. `lim_neighbors` is greater than `max_conditional_variables`.
        while not all([len(list(graph.neighbors(var))) < lim_neighbors for var in variables]):
            # Step 2: Iterate over the edges and find a conditioning set of
            # size `lim_neighbors` which makes u and v independent.
            if variant == "orig":
                for u, v in graph.edges():
                    if (u, v) not in required_edges:
                        for separating_set in self._get_potential_sepsets(
                            u, v, temporal_ordering, graph, lim_neighbors
                        ):
                            # If a conditioning set exists remove the edge, store the separating set
                            # and move on to finding conditioning set for next edge.
                            if ci_test(
                                u,
                                v,
                                separating_set,
                                significance_level=significance_level,
                            ):
                                separating_sets[frozenset((u, v))] = separating_set
                                graph.remove_edge(u, v)
                                break

            elif variant == "stable":
                neighbors = {node: set(graph.neighbors(node)) for node in variables}
                edges_to_remove = []
                # In case of stable, precompute neighbors as this is the stable algorithm.
                for u, v in graph.edges():
                    if (u, v) not in required_edges:
                        sep_vars = set()
                        found_independence = False
                        for separating_set in self._get_potential_sepsets(
                            u, v, temporal_ordering, graph, lim_neighbors, neighbors=neighbors
                        ):
                            if ci_test(
                                u,
                                v,
                                separating_set,
                                significance_level=significance_level,
                            ):
                                found_independence = True
                                sep_vars.update(separating_set)
                        if found_independence:
                            separating_sets[frozenset((u, v))] = tuple(sorted(sep_vars, key=repr))
                            edges_to_remove.append((u, v))
                graph.remove_edges_from(edges_to_remove)

            elif variant == "parallel":

                def _parallel_fun(u, v):
                    sep_vars = set()
                    found_independence = False
                    for separating_set in self._get_potential_sepsets(u, v, temporal_ordering, graph, lim_neighbors):
                        if ci_test(
                            u,
                            v,
                            separating_set,
                            significance_level=significance_level,
                        ):
                            found_independence = True
                            sep_vars.update(separating_set)
                    if found_independence:
                        return (u, v), tuple(sorted(sep_vars, key=repr))

                results = parallel_pool(
                    delayed(_parallel_fun)(u, v) for (u, v) in graph.edges() if (u, v) not in required_edges
                )
                for result in results:
                    if result is not None:
                        (u, v), sep_set = result
                        graph.remove_edge(u, v)
                        separating_sets[frozenset((u, v))] = sep_set

            else:
                raise ValueError(f"variant must be one of (orig, stable, parallel). Got: {variant}")

            # Step 3: After iterating over all the edges, expand the search space by increasing the size
            #         of conditioning set by 1.
            if lim_neighbors >= max_cond_vars:
                logger.info("Reached maximum number of allowed conditional variables. Exiting")
                break
            lim_neighbors += 1

            if show_progress and config.SHOW_PROGRESS:
                pbar.update(1)
                pbar.set_description(f"Working for n conditional variables: {lim_neighbors}")

        if show_progress and config.SHOW_PROGRESS:
            pbar.update(max_cond_vars - lim_neighbors)
            pbar.close()

        return graph, separating_sets

    @staticmethod
    def _get_potential_sepsets(
        u: Hashable,
        v: Hashable,
        temporal_ordering: dict[Hashable, int],
        graph: UndirectedGraph,
        lim_neighbors: int,
        neighbors: dict[Hashable, set[Hashable]] = None,
    ) -> set[tuple]:
        """
        Return the temporally consistent candidate separating sets of `u`, `v`.

        Each candidate is a sorted tuple of size ``lim_neighbors`` drawn from
        the (temporally filtered) neighbors of ``u`` or ``v``. Sorting both
        input sets makes ``combinations`` yield lex-sorted tuples, so the
        union of the two halves deduplicates directly via set semantics.

        Parameters
        ----------
        u: variable
            The node whose neighbors are being considered for separating set.

        v: variable
            The node along with u whose separating set is being calculated.

        temporal_ordering: dict
            The temporal ordering of variables according to prior knowledge.

        graph: UndirectedGraph
            The graph where separating sets are being calculated for the edges.

        lim_neighbors: int
            The maximum number of neighbours (conditioning variables) for u, v.

        Returns
        -------
        sepsets: set[tuple]
            Unique candidate separating sets of size ``lim_neighbors``.
        """

        if neighbors is not None:
            separating_set_u = neighbors[u].copy()
            separating_set_v = neighbors[v].copy()
        else:
            separating_set_u = set(graph.neighbors(u)).copy()
            separating_set_v = set(graph.neighbors(v)).copy()
        separating_set_u.discard(v)
        separating_set_v.discard(u)

        if temporal_ordering != dict():
            max_order = min(temporal_ordering[u], temporal_ordering[v])
            for neigh in list(separating_set_u):
                if temporal_ordering[neigh] > max_order:
                    separating_set_u.discard(neigh)

            for neigh in list(separating_set_v):
                if temporal_ordering[neigh] > max_order:
                    separating_set_v.discard(neigh)

        sorted_u = sorted(separating_set_u, key=repr)
        sorted_v = sorted(separating_set_v, key=repr)
        return set(combinations(sorted_u, lim_neighbors)) | set(combinations(sorted_v, lim_neighbors))


class _ScoreMixin:
    """
    Base class for all score-based causal discovery estimators.

    Score-based causal discovery algorithms (e.g., HillClimbSearch, GES) work by
    searching through the space of possible DAGs and scoring each candidate structure
    using a scoring function (e.g., BIC, K2, BDeu).
    """

    def _legal_operations_dag(
        self,
        model: DAG,
        scoring_method: BaseStructureScore,
        tabu_list: deque[tuple[str, tuple[Hashable, Hashable]]],
        max_indegree: int,
        forbidden_edges: list[tuple[Hashable, Hashable]],
        required_edges: list[tuple[Hashable, Hashable]],
    ) -> Generator[tuple[tuple[str, tuple[Hashable, Hashable]], float]]:
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. For details on scoring
        see Koller & Friedman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered. A list of
        edges can optionally be passed as `forbidden_edges` or `required_edges` to exclude those
        edges or to force them to be present in the model, respectively.
        """

        # Step 0: Pre-compute structures that are constant
        tabu_list = set(tabu_list)
        descendants = {v: nx.descendants(model, v) for v in model.nodes()}
        edges = list(model.edges())
        edge_set = set(edges)
        reverse_edge_set = {(Y, X) for (X, Y) in edges}

        parents_cache = {v: tuple(model.get_parents(v)) for v in model.nodes()}
        current_score = {v: scoring_method.local_score(v, parents_cache[v]) for v in model.nodes()}

        prior_add = scoring_method.structure_prior_ratio("+")
        prior_remove = scoring_method.structure_prior_ratio("-")
        prior_flip = scoring_method.structure_prior_ratio("flip")

        # Step 1: Get all legal operations for adding edges. Sort the iteration order for reproducible runs.
        potential_new_edges = sorted(set(permutations(self.variables_, 2)) - edge_set - reverse_edge_set)

        for X, Y in potential_new_edges:
            # Adding X->Y creates a cycle iff Y already reaches X.
            if X in descendants[Y]:
                continue
            operation = ("+", (X, Y))
            if (operation not in tabu_list) and ((X, Y) not in forbidden_edges):
                old_parents = parents_cache[Y]
                new_parents = old_parents + (X,)
                if len(new_parents) <= max_indegree:
                    score_delta = scoring_method.local_score(Y, new_parents) - current_score[Y] + prior_add
                    yield (operation, score_delta)

        # Step 2: Get all legal operations for removing edges
        for X, Y in edges:
            operation = ("-", (X, Y))
            if (operation not in tabu_list) and ((X, Y) not in required_edges):
                old_parents = parents_cache[Y]
                new_parents = tuple(var for var in old_parents if var != X)
                score_delta = scoring_method.local_score(Y, new_parents) - current_score[Y] + prior_remove
                yield (operation, score_delta)

        # Step 3: Get all legal operations for flipping edges
        for X, Y in edges:
            # Flipping X->Y to Y->X creates a cycle iff some other child of X
            # already reaches Y (a path X -> C -> ... -> Y with C != Y).
            if any(Y in descendants[c] for c in model.successors(X) if c != Y):
                continue
            operation = ("flip", (X, Y))
            if (
                ((operation not in tabu_list) and ("flip", (Y, X)) not in tabu_list)
                and ((X, Y) not in required_edges)
                and ((Y, X) not in forbidden_edges)
            ):
                new_X_parents = parents_cache[X] + (Y,)
                new_Y_parents = tuple(var for var in parents_cache[Y] if var != X)
                if len(new_X_parents) <= max_indegree:
                    score_delta = (
                        scoring_method.local_score(X, new_X_parents)
                        + scoring_method.local_score(Y, new_Y_parents)
                        - current_score[X]
                        - current_score[Y]
                        + prior_flip
                    )
                    yield (operation, score_delta)


class _TreeSearchMixin:
    """
    Mixin class providing shared functionality for tree-based causal discovery
    algorithms (Chow-Liu and TAN).

    Provides static helpers for resolving the ``edge_weights_fn`` argument,
    computing pairwise edge weights, and constructing a directed spanning tree
    (DAG) from a weight matrix.  Both :class:`ChowLiu` and :class:`TAN` inherit
    from this mixin so that the shared logic lives in exactly one place.
    """

    _EDGE_WEIGHT_FNS = {
        "mutual_info": mutual_info_score,
        "adjusted_mutual_info": adjusted_mutual_info_score,
        "normalized_mutual_info": normalized_mutual_info_score,
    }

    @staticmethod
    def _resolve_edge_weights_fn(edge_weights_fn):
        """
        Resolve ``edge_weights_fn`` to a callable of the form ``fn(array, array)``.

        Accepts either one of the string shorthands in
        :attr:`_EDGE_WEIGHT_FNS` or a callable (returned unchanged). Anything
        else raises ``ValueError``.
        """
        if callable(edge_weights_fn):
            return edge_weights_fn
        try:
            return _TreeSearchMixin._EDGE_WEIGHT_FNS[edge_weights_fn]
        except (KeyError, TypeError):
            raise ValueError(
                f"edge_weights_fn should be one of {list(_TreeSearchMixin._EDGE_WEIGHT_FNS)}, "
                f"or a callable of the form fn(array, array). Got: {edge_weights_fn!r}"
            )

    @staticmethod
    def _get_weights(data, edge_weights_fn="mutual_info", n_jobs=-1, show_progress=True):
        """
        Compute the pairwise edge weight matrix.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe where each column represents one variable.

        edge_weights_fn : str or callable, default="mutual_info"
            Method to use for computing edge weights. Options are:

            - ``"mutual_info"``: Mutual Information Score.
            - ``"adjusted_mutual_info"``: Adjusted Mutual Information Score.
            - ``"normalized_mutual_info"``: Normalized Mutual Information Score.
            - A callable of the form ``fn(array, array) -> float``.

        n_jobs : int, default=-1
            Number of jobs to run in parallel. ``-1`` means use all processors.

        show_progress : bool, default=True
            If ``True``, shows a progress bar.

        Returns
        -------
        weights : np.ndarray, shape (n_columns, n_columns)
            Symmetric matrix where ``weights[i, j]`` is the edge weight between
            variable *i* and variable *j*.
        """
        edge_weights_fn = _TreeSearchMixin._resolve_edge_weights_fn(edge_weights_fn)

        n_vars = len(data.columns)
        pbar = combinations(data.columns, 2)
        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(pbar, total=(n_vars * (n_vars - 1) / 2), desc="Building tree")

        vals = Parallel(n_jobs=n_jobs)(delayed(edge_weights_fn)(data.loc[:, u], data.loc[:, v]) for u, v in pbar)
        weights = np.zeros((n_vars, n_vars))
        indices = np.triu_indices(n_vars, k=1)
        weights[indices] = vals
        weights.T[indices] = vals
        return weights

    @staticmethod
    def _create_tree_and_dag(weights, columns, root_node):
        """
        Build a DAG by computing the maximum spanning tree from a weight matrix
        and directing all edges away from ``root_node`` via BFS.

        Parameters
        ----------
        weights : np.ndarray, shape (n_columns, n_columns)
            Symmetric matrix where each element represents an edge weight.

        columns : list or array-like
            Names of the columns (and rows) of the weight matrix.

        root_node : str, int, or any hashable python object
            The root node of the tree structure.

        Returns
        -------
        model : pgmpy.base.DAG
            The estimated DAG rooted at ``root_node``.
        """
        T = nx.maximum_spanning_tree(
            nx.from_pandas_adjacency(
                pd.DataFrame(weights, index=columns, columns=columns),
                create_using=nx.Graph,
            )
        )
        D = nx.bfs_tree(T, root_node)
        return DAG(D)
