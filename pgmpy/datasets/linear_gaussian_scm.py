from pgmpy.base import DAG
from pgmpy.datasets._base import _BaseDataset, _SimulationMixin


class LinearGaussianSCM(_SimulationMixin, _BaseDataset):
    """
    Simulated dataset from a randomly generated Linear Gaussian structural
    causal model.

    Wraps ``LinearGaussianBayesianNetwork.get_random()`` and ``.simulate()``
    in the ``_SimulationMixin`` pattern.
    """

    _tags = {
        "name": "linear_gaussian_scm",
        "n_variables": None,
        "n_samples": None,
        "has_ground_truth": True,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": True,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
        "sim_params": {
            "n_nodes": {"default": 5, "desc": "Number of variables in the DAG"},
            "edge_prob": {"default": 0.3, "desc": "Probability of edge between any two nodes"},
            "noise_scale": {"default": 1.0, "desc": "Standard deviation of additive Gaussian noise"},
        },
    }

    @classmethod
    def _build_model(cls, seed=None, n_nodes=5, edge_prob=0.3, noise_scale=1.0):
        from pgmpy.models import LinearGaussianBayesianNetwork as LGBN

        return LGBN.get_random(
            n_nodes=n_nodes,
            edge_prob=edge_prob,
            scale=noise_scale,
            seed=seed,
        )

    @classmethod
    def load_ground_truth(cls, seed=None, n_nodes=5, edge_prob=0.3, **kwargs):
        # noise_scale is intentionally absent — graph structure does not depend on CPD noise.
        model = cls._build_model(seed=seed, n_nodes=n_nodes, edge_prob=edge_prob)
        dag = DAG()
        dag.add_nodes_from(model.nodes())
        dag.add_edges_from(model.edges())
        return dag

    @classmethod
    def load_dataframe(
        cls,
        n_samples=1000,
        seed=None,
        n_nodes=5,
        edge_prob=0.3,
        noise_scale=1.0,
        **kwargs,
    ):
        model = cls._build_model(
            seed=seed,
            n_nodes=n_nodes,
            edge_prob=edge_prob,
            noise_scale=noise_scale,
        )
        actual_samples = 1000 if n_samples is None else n_samples
        return model.simulate(n_samples=actual_samples, seed=seed)
