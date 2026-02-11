from ._base import DiscreteMixin, _BaseExampleModel


class Asia(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] S. Lauritzen, D. Spiegelhalter. Local Computation with Probabilities on Graphical Structures and their
    Application to Expert Systems (with discussion). Journal of the Royal Statistical Society: Series B (Statistical
    Methodology), 50(2):157-224, 1988.
    """

    _tags = {
        "name": "asia",
        "n_nodes": 8,
        "n_edges": 8,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/asia.bif.gz"
