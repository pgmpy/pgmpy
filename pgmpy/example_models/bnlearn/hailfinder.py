from .._base import DiscreteMixin, _BaseExampleModel


class Hailfinder(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] B. Abramson, J. Brown, W. Edwards, A. Murphy, and R. L. Winkler. Hailfinder: A Bayesian system for
    forecasting severe weather. International Journal of Forecasting, 12(1):57-71, 1996.
    """

    _tags = {
        "name": "bnlearn/hailfinder",
        "n_nodes": 56,
        "n_edges": 66,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/hailfinder.bif.gz"
