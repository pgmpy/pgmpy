from .._base import PlainBIFMixin, _BaseExampleModel


class Twinframework(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/twinframework",
        "n_nodes": 7,
        "n_edges": 7,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/twinframework.bif"
