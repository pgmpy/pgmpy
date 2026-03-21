from .._base import BIFMixin, _BaseExampleModel


class Resilience(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/resilience",
        "n_nodes": 36,
        "n_edges": 52,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/resilience.bif"
