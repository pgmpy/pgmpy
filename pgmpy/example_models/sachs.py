from ._base import DiscreteMixin, _BaseExampleModel


class Sachs(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] K. Sachs, O. Perez, D. Pe'er, D. A. Lauffenburger and G. P. Nolan. Causal Protein-Signaling Networks
    Derived from Multiparameter Single-Cell Data. Science, 308:523-529, 2005.
    """

    _tags = {
        "name": "sachs",
        "n_nodes": 11,
        "n_edges": 17,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/sachs.bif.gz"
