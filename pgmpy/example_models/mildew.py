from ._base import DiscreteMixin, _BaseExampleModel


class Mildew(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] A. L. Jensen and F. V. Jensen. MIDAS - An Influence Diagram for Management of Mildew in Winter Wheat.
    Proceedings of the Twelfth Conference on Uncertainty in Artificial Intelligence (UAI1996), pages 349-356.
    """

    _tags = {
        "name": "mildew",
        "n_nodes": 35,
        "n_edges": 46,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/mildew.bif.gz"
