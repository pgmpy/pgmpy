from .._base import DiscreteMixin, _BaseExampleModel


class Water(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`jensen_water_1989`
    """

    _tags = {
        "name": "bnlearn/water",
        "n_nodes": 32,
        "n_edges": 66,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/water.bif.gz"
