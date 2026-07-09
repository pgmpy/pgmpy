import pandas as pd

from pgmpy.base import DAG
from pgmpy.datasets._base import BaseSimulatedDataset
from pgmpy.models import LinearGaussianBayesianNetwork


class LinearGaussianSCM(BaseSimulatedDataset):
    """
    Simulates a dataset from a random Linear Gaussian SCM.

    The random graph and its Gaussian parameters are generated once in
    ``__init__`` via ``LinearGaussianBayesianNetwork.get_random`` and stored
    on the instance. ``load_dataframe`` samples from that model and
    ``load_ground_truth`` returns its DAG, so both share a single model.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducible graph, parameter, and data generation.
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

    def __init__(
        self,
        seed: int | None = None,
        n_nodes: int = 5,
        edge_prob: float = 0.5,
        scale: float = 1.0,
    ):
        self.model = LinearGaussianBayesianNetwork.get_random(
            n_nodes=n_nodes,
            edge_prob=edge_prob,
            scale=scale,
            seed=seed,
        )
        self.seed = seed

    def load_dataframe(self, n_samples: int | None = None) -> pd.DataFrame:
        """Sample data from the generated Linear Gaussian SCM.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1000.

        Returns
        -------
        pd.DataFrame
        """
        return self.model.simulate(
            n_samples=n_samples if n_samples is not None else 1000,
            seed=self.seed,
        )

    def load_ground_truth(self) -> DAG:
        """Return the ground-truth DAG of the generated SCM.

        Returns
        -------
        DAG
        """
        return DAG(self.model)
