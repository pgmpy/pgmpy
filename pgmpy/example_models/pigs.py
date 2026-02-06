from ._base import DiscreteMixin, _BaseExampleModel


class Pigs(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] Pigs (Genetic network). bnlearn Bayesian Network Repository.
    """

    _tags = {
        "name": "pigs",
        "n_nodes": 441,
        "n_edges": 592,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/pigs.bif.gz"
