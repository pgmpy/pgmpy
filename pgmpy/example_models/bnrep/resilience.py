from .._base import PlainBIFMixin, _BaseExampleModel


class Resilience(PlainBIFMixin, _BaseExampleModel):
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
