from .._base import DiscreteMixin, _BaseExampleModel


class Diabetes(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] S. Andreassen, R. Hovorka, J. Benn, K. G. Olesen, and E. R. Carson. A Model-based Approach to Insulin
    Adjustment. In Proceedings of the 3rd Conference on Artificial Intelligence in Medicine, pages 239-248.
    Springer-Verlag, 1991.
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
