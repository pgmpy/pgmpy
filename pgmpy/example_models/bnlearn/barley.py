from .._base import DiscreteMixin, _BaseExampleModel


class Barley(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] Preliminary model for barley developed under the project: "Production of beer from Danish malting barley
    grown without the use of pesticides" by Kristian Kristensen, Ilse A. Rasmussen and others.
    """

    _tags = {
        "name": "bnlearn/barley",
        "n_nodes": 48,
        "n_edges": 84,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/barley.bif.gz"
