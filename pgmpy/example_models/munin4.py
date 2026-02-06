from ._base import DiscreteMixin, _BaseExampleModel


class Munin4(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] S. Andreassen, F. V. Jensen, S. K. Andersen, B. Falck, U. Kjærulff, M. Woldbye, A. R. Sørensen, A.
    Rosenfalck, and F. Jensen. MUNIN - an Expert EMG Assistant. In Computer-Aided Electromyography and Expert
    Systems, Chapter 21. Elsevier (North-Holland), 1989.
    """

    _tags = {
        "name": "munin4",
        "n_nodes": 1038,
        "n_edges": 1388,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/munin4.bif.gz"
