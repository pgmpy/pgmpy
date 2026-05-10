from .._base import DiscreteMixin, _BaseExampleModel


class Diabetes(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`andreassen_1991`
    """

    _tags = {
        "name": "bnlearn/diabetes",
        "n_nodes": 413,
        "n_edges": 602,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/diabetes.bif.gz"
