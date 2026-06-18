import pandas as pd

from pgmpy.base import DAG
from pgmpy.datasets._base import BaseSimulatedDataset
from pgmpy.models import LinearGaussianBayesianNetwork


class LinearGaussianSCM(BaseSimulatedDataset):
    """
    Simulates a dataset from a random Linear Gaussian SCM.

    Delegates graph and parameter generation to
    ``LinearGaussianBayesianNetwork.get_random`` and data sampling
    to ``LinearGaussianBayesianNetwork.simulate``.

    Parameters
    ----------
    n_nodes : int, default 5
        Number of variables in the generated DAG.
    edge_prob : float, default 0.5
        Probability of an edge between any two topologically ordered
        nodes.
    scale : float, default 1
        Scale parameter passed to ``get_random``. Controls the standard
        deviation of the normal distribution used when sampling both
        linear coefficients and CPD noise terms.
    """

    _tags = {
        "name": "linear_gaussian",
        "has_ground_truth": True,
        "is_continuous": True,
    }

    @classmethod
    def _build_model(
        cls,
        seed: int | None = None,
        n_nodes: int = 5,
        edge_prob: float = 0.5,
        scale: float = 1.0,
    ) -> LinearGaussianBayesianNetwork:
        """Build a fitted LinearGaussianBayesianNetwork from random parameters.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducible graph generation.
        n_nodes : int, default 5
            Number of variables in the generated DAG.
        edge_prob : float, default 0.5
            Probability of an edge between any two topologically ordered
            nodes.
        scale : float, default 1.0
            Scale parameter for coefficient and noise sampling.

        Returns
        -------
        LinearGaussianBayesianNetwork
        """
        return LinearGaussianBayesianNetwork.get_random(
            n_nodes=n_nodes,
            edge_prob=edge_prob,
            scale=scale,
            seed=seed,
        )

    @classmethod
    def load_dataframe(
        cls,
        n_samples: int | None = None,
        seed: int | None = None,
        n_nodes: int = 5,
        edge_prob: float = 0.5,
        scale: float = 1.0,
    ) -> pd.DataFrame:
        """Generate data from a random Linear Gaussian SCM.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1000.
        seed : int, optional
            Random seed for reproducible graph and data generation.
        n_nodes : int, default 5
            Number of variables in the generated DAG.
        edge_prob : float, default 0.5
            Probability of an edge between any two topologically ordered
            nodes.
        scale : float, default 1.0
            Scale parameter for coefficient and noise sampling in
            ``LinearGaussianBayesianNetwork.get_random``.

        Returns
        -------
        pd.DataFrame
        """
        model = cls._build_model(seed=seed, n_nodes=n_nodes, edge_prob=edge_prob, scale=scale)
        return model.simulate(n_samples=n_samples if n_samples is not None else 1000, seed=seed)

    @classmethod
    def load_ground_truth(
        cls,
        seed: int | None = None,
        n_nodes: int = 5,
        edge_prob: float = 0.5,
        scale: float = 1.0,
    ) -> DAG:
        """Return the ground-truth DAG of the generated SCM.

        Parameters
        ----------
        seed : int, optional
            Must match the seed used in ``load_dataframe`` to get the
            corresponding graph.
        n_nodes : int, default 5
            Number of variables in the generated DAG.
        edge_prob : float, default 0.5
            Probability of an edge between any two topologically ordered
            nodes.
        scale : float, default 1.0
            Accepted for call-signature compatibility with
            ``load_dataset``. The graph structure is independent of
            this value.

        Returns
        -------
        DAG
        """
        # scale is intentionally not forwarded — graph structure is
        # determined only by n_nodes, edge_prob, and seed.
        model = cls._build_model(seed=seed, n_nodes=n_nodes, edge_prob=edge_prob)
        dag = DAG()
        dag.add_nodes_from(model.nodes())
        dag.add_edges_from(model.edges())
        return dag
