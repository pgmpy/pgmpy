from __future__ import annotations

import warnings
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base import DAG
from pgmpy.datasets._base import BaseSimulatedDataset

_SUPPORTED_FUNCTION_TYPES = {"sine_add", "polynomial", "sigmoid_add"}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class AdditiveNoiseModel(BaseSimulatedDataset):
    """
    Simulated dataset from a random Additive Noise Model.

    Generates data according to the model:

    .. math::

        X_j = f_j(\\text{Pa}(X_j)) + N_j

    where :math:`f_j` are non-linear functions and :math:`N_j` are independent
    noise terms (Hoyer et al., 2008).

    Parameters accepted via keyword arguments in :func:`load_dataset`:

    * **dag** (DAG, optional) – A user-provided DAG.  When given,
      ``n_nodes`` and ``edge_prob`` are ignored with a warning.
    * **n_nodes** (int, default 5) – Number of variables in the random DAG.
    * **edge_prob** (float, default 0.5) – Probability of an edge between any
      two topologically ordered nodes in the random DAG.
    * **noise** (BaseDistribution, optional) – A ``skpro`` distribution
      instance for noise terms.  When ``None``, standard Gaussian noise is
      used.  ``skpro`` must be installed separately.
    * **function_type** (str or callable, default ``"sine_add"``) – The
      non-linear function applied to parent values.  Supported strings:
      ``"sine_add"``, ``"polynomial"``, ``"sigmoid_add"``.  A callable must
      accept an array of shape ``(n_samples, n_parents)`` and return an
      array of shape ``(n_samples,)``.
    * **noise_scale** (float, default 1.0) – Multiplier for the default
      Gaussian noise.  Cannot be used together with a custom ``noise``
      distribution.
    * **weight_range** (tuple of float, default ``(0.5, 2.0)``) – Range
      for randomly sampled coefficients in preset functions.  Ignored when
      ``function_type`` is a callable.

    References
    ----------
    Hoyer, P., Janzing, D., Mooij, J. M., Peters, J., & Schölkopf, B. (2008).
    Nonlinear causal discovery with additive noise models. *NeurIPS*.
    """

    _tags = {
        "name": "anm",
        "has_ground_truth": True,
        "is_continuous": True,
    }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_dag(
        dag: DAG | None,
        n_nodes: int,
        edge_prob: float,
        seed: int | None,
    ) -> DAG:
        """Return a user-provided DAG or generate a random one.

        Parameters
        ----------
        dag : DAG or None
            A user-provided DAG.  When given, *n_nodes* and *edge_prob* are
            ignored and a ``UserWarning`` is emitted.
        n_nodes : int
            Number of nodes for the random DAG (ignored if *dag* is given).
        edge_prob : float
            Edge probability for the random DAG (ignored if *dag* is given).
        seed : int or None
            Seed forwarded to ``DAG.get_random``.

        Returns
        -------
        DAG
        """
        if dag is not None:
            if not isinstance(dag, DAG):
                raise TypeError(f"dag must be a pgmpy.base.DAG instance, got {type(dag).__name__}.")
            if not nx.is_directed_acyclic_graph(dag):
                raise ValueError("dag must be an acyclic directed graph.")
            if n_nodes != 5 or edge_prob != 0.5:
                warnings.warn(
                    "dag was provided; n_nodes and edge_prob are ignored.",
                    UserWarning,
                    stacklevel=4,
                )
            return dag

        if not isinstance(n_nodes, int) or n_nodes < 1:
            raise ValueError(f"n_nodes must be a positive integer, got {n_nodes!r}.")
        if not 0 <= edge_prob <= 1:
            raise ValueError(f"edge_prob must be between 0 and 1, got {edge_prob!r}.")

        return DAG.get_random(n_nodes=n_nodes, edge_prob=edge_prob, seed=seed)

    @staticmethod
    def _make_function(
        function_type: str | Any,
        n_parents: int,
        rng: np.random.Generator,
        weight_range: tuple[float, float],
    ):
        """Return a callable ``f(parent_data) -> signal`` for one node.

        Parameters
        ----------
        function_type : str or callable
            Preset name or user-provided callable.
        n_parents : int
            Number of parents for the current node.
        rng : numpy.random.Generator
            Seeded RNG for sampling random coefficients.
        weight_range : tuple of float
            ``(low, high)`` range for coefficient magnitudes.

        Returns
        -------
        callable
            ``f(parent_data: ndarray (n_samples, n_parents)) -> ndarray (n_samples,)``
        """
        if callable(function_type):
            return function_type

        low, high = weight_range
        weights = rng.uniform(low, high, size=n_parents)
        signs = rng.choice([-1, 1], size=n_parents)
        weights = weights * signs

        if function_type == "sine_add":
            return lambda x, w=weights: (w * np.sin(x)).sum(axis=1)
        elif function_type == "polynomial":
            degrees = rng.choice([2, 3], size=n_parents)
            return lambda x, w=weights, d=degrees: (w * np.power(x, d)).sum(axis=1)
        elif function_type == "sigmoid_add":
            return lambda x, w=weights: (w * _sigmoid(x)).sum(axis=1)
        else:
            raise ValueError(
                f"Unknown function_type '{function_type}'. Supported: {sorted(_SUPPORTED_FUNCTION_TYPES)}."
            )

    @staticmethod
    def _sample_noise(
        noise: Any,
        n_samples: int,
        noise_scale: float,
        rng: np.random.Generator,
        noise_seed: int | None,
    ) -> np.ndarray:
        """Sample additive noise for one node.

        Parameters
        ----------
        noise : BaseDistribution or None
            ``None`` uses default Gaussian; otherwise calls ``dist.sample()``.
        n_samples : int
            Number of noise samples to generate.
        noise_scale : float
            Multiplier applied only when *noise* is ``None``.
        rng : numpy.random.Generator
            Seeded RNG for the default Gaussian path.
        noise_seed : int or None
            Per-node seed for the ``skpro`` path (sets global
            ``np.random.seed``).  Must be unique per node to ensure
            independent noise draws.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
        """
        if noise is None:
            return rng.normal(0, 1, size=n_samples) * noise_scale

        # skpro distribution path — save/restore global RNG state so
        # the simulator never leaks side effects to the caller.
        state = np.random.get_state()  # noqa: NPY002
        try:
            if noise_seed is not None:
                np.random.seed(noise_seed)  # noqa: NPY002
            samples = noise.sample(n_samples=n_samples)
        finally:
            np.random.set_state(state)  # noqa: NPY002
        return np.asarray(samples).flatten()

    @staticmethod
    def _validate_params(
        n_samples: int | None,
        function_type: str | Any,
        weight_range: tuple[float, float],
        noise: Any,
        noise_scale: float,
    ) -> None:
        """Validate simulator parameters, raising on invalid combinations."""
        # n_samples
        if n_samples is not None and (not isinstance(n_samples, int) or n_samples < 1):
            raise ValueError(f"n_samples must be a positive integer or None, got {n_samples!r}.")

        # noise_scale
        if noise_scale <= 0:
            raise ValueError(f"noise_scale must be positive, got {noise_scale!r}.")

        # function_type
        if not callable(function_type) and not isinstance(function_type, str):
            raise TypeError(f"function_type must be a string or callable, got {type(function_type).__name__}.")
        if isinstance(function_type, str) and function_type not in _SUPPORTED_FUNCTION_TYPES:
            raise ValueError(
                f"Unknown function_type '{function_type}'. Supported: {sorted(_SUPPORTED_FUNCTION_TYPES)}."
            )

        # weight_range
        if not isinstance(weight_range, (tuple, list)) or len(weight_range) != 2:
            raise ValueError(f"weight_range must be a (low, high) tuple of length 2, got {weight_range!r}.")
        if weight_range[0] <= 0 or weight_range[1] <= 0:
            raise ValueError("weight_range bounds must be positive.")
        if weight_range[0] > weight_range[1]:
            raise ValueError("weight_range lower bound must be <= upper bound.")

        # noise and noise_scale conflict
        if noise is not None and noise_scale != 1.0:
            raise ValueError(
                "noise_scale cannot be used with a custom noise distribution. "
                "Set the scale via the distribution's own parameters."
            )

        # Check skpro availability when custom noise is provided
        if noise is not None:
            try:
                from skpro.distributions.base import BaseDistribution
            except ImportError:
                raise ImportError(
                    "skpro is required for custom noise distributions. Install with: pip install skpro"
                ) from None
            if not isinstance(noise, BaseDistribution):
                raise TypeError(f"noise must be a skpro BaseDistribution instance or None, got {type(noise).__name__}.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def load_dataframe(
        cls,
        n_samples: int | None = None,
        seed: int | None = None,
        dag: DAG | None = None,
        n_nodes: int = 5,
        edge_prob: float = 0.5,
        noise: Any = None,
        function_type: str | Any = "sine_add",
        noise_scale: float = 1.0,
        weight_range: tuple[float, float] = (0.5, 2.0),
    ) -> pd.DataFrame:
        """Generate data from a random Additive Noise Model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate.  Defaults to 1000.
        seed : int, optional
            Random seed for reproducible graph and data generation.
        dag : DAG, optional
            A user-provided DAG.  When given, ``n_nodes`` and ``edge_prob``
            are ignored.
        n_nodes : int, optional
            Number of variables in the generated DAG.
        edge_prob : float, optional
            Probability of an edge between any two topologically ordered nodes.
        noise : BaseDistribution, optional
            A ``skpro`` distribution for additive noise.  When ``None``,
            standard Gaussian noise is used.
        function_type : str or callable, optional
            Non-linearity applied to parent values.  Supported preset
            strings: ``"sine_add"``, ``"polynomial"``, ``"sigmoid_add"``.
            A callable must accept ``(n_samples, n_parents)`` and return
            ``(n_samples,)``.
        noise_scale : float, optional
            Multiplier for the default Gaussian noise.  Cannot be used
            together with a custom *noise* distribution.
        weight_range : tuple of float, optional
            ``(low, high)`` range for random coefficients in preset
            functions.  Ignored for custom callables.

        Returns
        -------
        pd.DataFrame
        """
        cls._validate_params(n_samples, function_type, weight_range, noise, noise_scale)

        graph = cls._get_dag(dag, n_nodes, edge_prob, seed)
        rng = np.random.default_rng(seed)
        topo_order = list(nx.topological_sort(graph))
        n = n_samples if n_samples is not None else 1000

        data = pd.DataFrame(0.0, index=range(n), columns=list(graph.nodes()))

        for node in topo_order:
            parents = list(graph.predecessors(node))
            # Derive a unique per-node seed so that custom skpro noise
            # draws are independent across nodes but still reproducible.
            noise_seed = int(rng.integers(0, 2**31)) if seed is not None else None
            noise_vals = cls._sample_noise(noise, n, noise_scale, rng, noise_seed)

            if not parents:
                data[node] = noise_vals
            else:
                parent_data = data[parents].values
                func = cls._make_function(function_type, len(parents), rng, weight_range)
                result = func(parent_data)

                if result.shape != (n,):
                    raise ValueError(f"function_type callable must return array of shape ({n},), got {result.shape}.")
                data[node] = result + noise_vals

        return data

    @classmethod
    def load_ground_truth(
        cls,
        seed: int | None = None,
        dag: DAG | None = None,
        n_nodes: int = 5,
        edge_prob: float = 0.5,
        **kwargs,
    ) -> DAG:
        """Return the ground-truth DAG of the generated ANM.

        Parameters
        ----------
        seed : int, optional
            Must match the seed used in ``load_dataframe`` to get the
            corresponding graph.
        dag : DAG, optional
            A user-provided DAG.  When given, ``n_nodes`` and ``edge_prob``
            are ignored.
        n_nodes : int, optional
            Number of variables in the generated DAG.
        edge_prob : float, optional
            Probability of an edge between any two topologically ordered nodes.
        **kwargs
            Absorbed for call-signature compatibility with ``load_dataset()``.
            The graph structure is independent of ``noise``,
            ``function_type``, ``noise_scale``, and ``weight_range``.

        Returns
        -------
        pgmpy.base.DAG
        """
        graph = cls._get_dag(dag, n_nodes, edge_prob, seed)
        gt = DAG()
        gt.add_nodes_from(graph.nodes())
        gt.add_edges_from(graph.edges())
        return gt
