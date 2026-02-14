from ._base import DiscreteMixin, _BaseExampleModel


class Earthquake(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] K. B. Korb, A. E. Nicholson. Bayesian Artificial Intelligence, 2nd edition, Section 2.2.2. CRC Press, 2010.
    """

    _tags = {
        "name": "earthquake",
        "n_nodes": 5,
        "n_edges": 4,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/earthquake.bif.gz"
