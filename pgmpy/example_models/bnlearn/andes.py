from .._base import DiscreteMixin, _BaseExampleModel


class Andes(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`conati_1997`
    """

    _tags = {
        "name": "bnlearn/andes",
        "n_nodes": 223,
        "n_edges": 338,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/andes.bif.gz"
