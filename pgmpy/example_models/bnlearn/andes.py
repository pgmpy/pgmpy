from .._base import DiscreteMixin, _BaseExampleModel


class Andes(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] C. Conati, A. S. Gertner, K. VanLehn, M. J. Druzdzel. On-line Student Modeling for Coached Problem
    Solving Using Bayesian Networks. In Proceedings of the 6th International Conference on User Modeling, pages
    231-242. Springer-Verlag, 1997.
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
