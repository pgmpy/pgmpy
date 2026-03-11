from .._base import DiscreteMixin, _BaseExampleModel


class Munin1(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] S. Andreassen, F. V. Jensen, S. K. Andersen, B. Falck, U. Kjærulff, M. Woldbye, A. R. Sørensen, A.
    Rosenfalck, and F. Jensen. MUNIN - an Expert EMG Assistant. In Computer-Aided Electromyography and Expert
    Systems, Chapter 21. Elsevier (North-Holland), 1989.
    """

    _tags = {
        "name": "bnlearn/munin1",
        "n_nodes": 186,
        "n_edges": 273,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/munin1.bif.gz"
