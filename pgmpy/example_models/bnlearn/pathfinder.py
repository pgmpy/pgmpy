from .._base import BaseExampleModel, DiscreteMixin


class Pathfinder(DiscreteMixin, BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`heckerman_1992`
    """

    _tags = {
        "name": "bnlearn/pathfinder",
        "n_nodes": 109,
        "n_edges": 195,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/pathfinder.bif.gz"
