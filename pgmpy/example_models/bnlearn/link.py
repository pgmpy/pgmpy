from .._base import DiscreteMixin, _BaseExampleModel


class Link(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] C. S. Jensen and A. Kong. Blocking Gibbs Sampling for Linkage Analysis in Large Pedigrees with Many
    Loops. The American Journal of Human Genetics, 65(3):885-901, 1999.
    """

    _tags = {
        "name": "bnlearn/link",
        "n_nodes": 724,
        "n_edges": 1125,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/link.bif.gz"
