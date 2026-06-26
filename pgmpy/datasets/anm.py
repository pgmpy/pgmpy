from __future__ import annotations

import warnings
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base import DAG
from pgmpy.datasets._base import BaseSimulatedDataset

_DEFAULT_FUNCTIONS = (np.sin, np.cos, np.tanh)


class AdditiveNoiseModel(BaseSimulatedDataset):
    """Simulated dataset from a random Additive Noise Model.

    Generates data according to the model:

    .. math::

        X_j = f_j(\\text{Pa}(X_j)) + N_j

    where :math:`f_j` are non-linear functions and :math:`N_j` are
    independent noise terms :cite:p:`hoyer_2008`.

    Parameters accepted via keyword arguments in :func:`load_dataset`:

    Parameters
    ----------
    dag : DAG, optional
        A user-provided DAG.  When given, ``n_nodes`` and
        ``edge_prob`` are ignored with a warning.
    n_nodes : int, optional (default 5)
        Number of variables in the random DAG.
    edge_prob : float, optional (default 0.5)
        Probability of an edge between any two topologically
        ordered nodes in the random DAG.
    noise : BaseDistribution, optional
        A ``skpro`` distribution instance for noise terms.
        When ``None``, standard normal :math:`\\mathcal{N}(0, 1)`
        noise is used.  ``skpro`` must be installed separately.
    function_type : set, tuple, or list of callable, optional
        Non-linear functions to randomly apply to parent
        values.  Each callable must accept and return a numpy
        array.  Default is ``(np.sin, np.cos, np.tanh)``.
    weight_range : tuple of float, optional (default ``(-1, 1)``)
        Range ``(low, high)`` for randomly sampled edge
        coefficients.

    References
    ----------
    - :cite:p:`hoyer_2008`
    """

    _tags = {
        "name": "anm",
        "has_ground_truth": True,
        "is_continuous": True,
    }

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
            A user-provided DAG.  When given, *n_nodes* and
            *edge_prob* are ignored and a warning is emitted.
        n_nodes : int
            Number of nodes for the random DAG.
        edge_prob : float
            Edge probability for the random DAG.
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
    def _sample_noise(
        noise: Any,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample additive noise for one node.

        When *noise* is ``None``, samples are drawn from a
        standard normal distribution using the isolated *rng*
        generator (no global state is touched).  When a custom
        ``skpro`` distribution is provided, ``noise.sample()``
        is called directly.

        Parameters
        ----------
        noise : BaseDistribution or None
            ``None`` uses standard normal :math:`N(0, 1)`;
            otherwise calls ``noise.sample()``.
        n_samples : int
            Number of noise samples to generate.
        rng : numpy.random.Generator
            Seeded RNG for the default Gaussian path.

        Returns
        -------
        numpy.ndarray of shape ``(n_samples,)``
        """
        if noise is None:
            return rng.normal(0, 1, size=n_samples)

        samples = noise.sample(n_samples=n_samples)
        return np.asarray(samples).flatten()

    @staticmethod
    def _validate_params(
        n_samples: int | None,
        function_type: set | tuple | list,
        weight_range: tuple[float, float],
        noise: Any,
    ) -> None:
        """Validate simulator parameters.

        Checks *n_samples* (positive int or None),
        *function_type* (non-empty collection of callables),
        *weight_range* (two-element tuple with low <= high),
        and *noise* (skpro BaseDistribution or None).
        """
        if n_samples is not None and (not isinstance(n_samples, int) or n_samples < 1):
            raise ValueError(f"n_samples must be a positive integer or None, got {n_samples!r}.")

        # function_type must be a collection of callables
        if not isinstance(function_type, (set, tuple, list)):
            raise TypeError(
                f"function_type must be a set, tuple, or list of callables, got {type(function_type).__name__}."
            )
        if len(function_type) == 0:
            raise ValueError("function_type must contain at least one callable.")
        for fn in function_type:
            if not callable(fn):
                raise TypeError(f"All items in function_type must be callable, got {type(fn).__name__}.")

        # weight_range
        if not isinstance(weight_range, (tuple, list)) or len(weight_range) != 2:
            raise ValueError(f"weight_range must be a (low, high) tuple of length 2, got {weight_range!r}.")
        if weight_range[0] > weight_range[1]:
            raise ValueError("weight_range lower bound must be <= upper bound.")

        # Check skpro availability when custom noise is given
        if noise is not None:
            from skpro.distributions.base import BaseDistribution

            if not isinstance(noise, BaseDistribution):
                raise TypeError(f"noise must be a skpro BaseDistribution instance or None, got {type(noise).__name__}.")

    @classmethod
    def load_dataframe(
        cls,
        n_samples: int | None = None,
        seed: int | None = None,
        dag: DAG | None = None,
        n_nodes: int = 5,
        edge_prob: float = 0.5,
        noise: Any = None,
        function_type: set | tuple | list = _DEFAULT_FUNCTIONS,
        weight_range: tuple[float, float] = (-1, 1),
    ) -> pd.DataFrame:
        """Generate data from a random Additive Noise Model.

        For each child node :math:`X_j` with parents
        :math:`\\text{Pa}(X_j)`, a function is randomly chosen
        from *function_type*, applied element-wise to each
        parent column, multiplied by a random weight sampled
        from *weight_range*, summed, and combined with additive
        noise.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate.  Defaults to 1000.
        seed : int, optional
            Random seed for reproducible graph and data
            generation.
        dag : DAG, optional
            A user-provided DAG.  When given, ``n_nodes`` and
            ``edge_prob`` are ignored.
        n_nodes : int, optional
            Number of variables in the generated DAG.
        edge_prob : float, optional
            Edge probability for the random DAG.
        noise : BaseDistribution, optional
            A ``skpro`` distribution for additive noise.  When
            ``None``, standard normal :math:`\\mathcal{N}(0, 1)`
            noise is used.
        function_type : set, tuple, or list of callable, optional
            Non-linear functions to randomly apply to each
            parent column.  Default is
            ``(np.sin, np.cos, np.tanh)``.
        weight_range : tuple of float, optional
            ``(low, high)`` range for random edge coefficients.
            Default is ``(-1, 1)``.

        Returns
        -------
        pd.DataFrame
        """
        cls._validate_params(n_samples, function_type, weight_range, noise)

        graph = cls._get_dag(dag, n_nodes, edge_prob, seed)
        rng = np.random.default_rng(seed)
        topo_order = list(nx.topological_sort(graph))
        n = n_samples if n_samples is not None else 1000
        funcs = list(function_type)

        data = pd.DataFrame(0.0, index=range(n), columns=list(graph.nodes()))

        for node in topo_order:
            parents = list(graph.predecessors(node))
            noise_vals = cls._sample_noise(noise, n, rng)

            if not parents:
                data[node] = noise_vals
            else:
                parent_data = data[parents].values
                signal = np.zeros(n)
                for i, parent_col in enumerate(parents):
                    w = rng.uniform(*weight_range)
                    fn = rng.choice(funcs)
                    signal += w * fn(parent_data[:, i])
                data[node] = signal + noise_vals

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
            Must match the seed used in ``load_dataframe`` to
            get the corresponding graph.
        dag : DAG, optional
            A user-provided DAG.  When given, ``n_nodes`` and
            ``edge_prob`` are ignored.
        n_nodes : int, optional
            Number of variables in the generated DAG.
        edge_prob : float, optional
            Edge probability for the random DAG.
        **kwargs
            Absorbed for call-signature compatibility with
            ``load_dataset()``.

        Returns
        -------
        pgmpy.base.DAG
        """
        return cls._get_dag(dag, n_nodes, edge_prob, seed)
