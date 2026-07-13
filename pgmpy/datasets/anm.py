from __future__ import annotations

import numbers
import warnings
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base import DAG
from pgmpy.datasets._base import BaseSimulatedDataset


class AdditiveNoiseModel(BaseSimulatedDataset):
    """Simulated dataset from a random Additive Noise Model.

    Generates data according to the model:

    .. math::

        X_j = \\sum_{i \\in \\text{Pa}(X_j)} w_i \\cdot f_i(X_i) + N_j

    where :math:`\\text{Pa}(X_j)` denotes the set of parent nodes of :math:`X_j` in the DAG, :math:`f_i` are
    functions randomly chosen from *function_type* (one per parent edge), :math:`w_i` are edge coefficients
    determined by *weight*, and :math:`N_j` are independent additive noise terms :cite:p:`hoyer_2008`.

    Parameters
    ----------
    dag : DAG, optional
        A user-provided DAG.  When given, ``n_nodes`` and ``edge_prob`` are ignored with a warning.
    n_nodes : int, default 5
        Number of variables in the random DAG.
    edge_prob : float, default 0.5
        Probability of an edge between any two topologically ordered nodes in the random DAG.
    noise : distribution object, optional
        Any object with a ``.sample(n_samples=...)`` or ``.rvs(size=...)`` method (e.g., ``scipy.stats`` or
        ``skpro`` distributions).  When ``None``, standard normal :math:`\\mathcal{N}(0, 1)` noise is used.
    function_type : tuple or list of callable, optional
        Functions to randomly apply to each parent column. Each callable must accept a 1-D numpy array and
        return a 1-D numpy array of the same shape.  Default is ``(np.sin, np.cos, np.tanh)``.
    weight : float or tuple of float, default ``(-1, 1)``
        Edge coefficients.  If a single float, every edge uses that fixed weight.  If a ``(low, high)`` tuple,
        each edge weight is sampled uniformly from the range.
    seed : int, optional
        Random seed for reproducible graph and data generation.

    References
    ----------
    - :cite:p:`hoyer_2008`
    """

    _tags = {
        "name": "anm",
        "has_ground_truth": True,
        "is_continuous": True,
    }

    def __init__(
        self,
        dag: DAG | None = None,
        n_nodes: int = 5,
        edge_prob: float = 0.5,
        noise: Any = None,
        function_type: tuple | list = (np.sin, np.cos, np.tanh),
        weight: float | tuple[float, float] = (-1, 1),
        seed: int | None = None,
    ):
        if dag is not None:
            if not isinstance(dag, DAG):
                raise TypeError(f"dag must be a pgmpy.base.DAG instance, got {type(dag).__name__}.")
            if n_nodes != 5 or edge_prob != 0.5:
                warnings.warn(
                    "dag was provided; n_nodes and edge_prob are ignored.",
                    UserWarning,
                    stacklevel=2,
                )
            self.dag = dag
        else:
            self.dag = DAG.get_random(n_nodes=n_nodes, edge_prob=edge_prob, seed=seed)

        if isinstance(weight, (tuple, list)):
            if len(weight) != 2:
                raise ValueError(f"weight range must be a (low, high) tuple of length 2, got length {len(weight)}.")
        elif not isinstance(weight, numbers.Real) or isinstance(weight, bool):
            raise TypeError(f"weight must be a float or a (low, high) tuple, got {type(weight).__name__}.")

        self.noise = noise
        self.function_type = function_type
        self.weight = weight
        self.seed = seed

    def load_dataframe(self, n_samples: int | None = None) -> pd.DataFrame:
        """Sample data from the generated Additive Noise Model.

        For each child node :math:`X_j`, every parent edge is independently assigned a random function from
        *function_type* and a weight from *weight*.  The weighted function outputs are summed and combined
        with additive noise.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate.  Defaults to 1000.

        Returns
        -------
        pd.DataFrame
        """
        rng = np.random.default_rng(self.seed)
        topo_order = list(nx.topological_sort(self.dag))
        n = n_samples if n_samples is not None else 1000
        funcs = list(self.function_type)

        data = pd.DataFrame(0.0, index=range(n), columns=list(self.dag.nodes()))

        for node in topo_order:
            parents = list(self.dag.predecessors(node))

            if self.noise is None:
                noise_vals = rng.normal(0, 1, size=n)
            elif hasattr(self.noise, "rvs"):
                noise_vals = np.asarray(self.noise.rvs(size=n, random_state=rng)).flatten()
            elif hasattr(self.noise, "sample"):
                # skpro's .sample() has no random_state arg and draws from numpy's legacy global RNG.
                skpro_seed = int(rng.integers(0, 2**32 - 1))
                global_state = np.random.get_state()
                try:
                    np.random.seed(skpro_seed)
                    noise_vals = np.asarray(self.noise.sample(n_samples=n)).flatten()
                finally:
                    np.random.set_state(global_state)
            else:
                raise TypeError(f"noise must have a .sample() or .rvs() method, got {type(self.noise).__name__}.")

            if not parents:
                data[node] = noise_vals
            else:
                parent_data = data[parents].values
                signal = np.zeros(n)
                for i, parent_col in enumerate(parents):
                    w = rng.uniform(*self.weight) if isinstance(self.weight, (tuple, list)) else self.weight
                    fn = rng.choice(funcs)
                    signal += w * fn(parent_data[:, i])
                data[node] = signal + noise_vals

        return data

    def load_ground_truth(self) -> DAG:
        """Return the ground-truth DAG of the generated ANM.

        Returns
        -------
        DAG
        """
        return self.dag
