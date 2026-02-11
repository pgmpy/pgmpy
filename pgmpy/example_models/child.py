from ._base import DiscreteMixin, _BaseExampleModel


class Child(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] D. J. Spiegelhalter, R. G. Cowell (1992). Learning in probabilistic expert systems. In Bayesian
    Statistics 4 (J. M. Bernardo, J. O. Berger, A. P. Dawid and A. F. M. Smith, eds.), 447-466. Clarendon Press,
    Oxford.
    """

    _tags = {
        "name": "child",
        "n_nodes": 20,
        "n_edges": 25,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/child.bif.gz"
