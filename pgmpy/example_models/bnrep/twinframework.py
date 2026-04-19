from .._base import BIFMixin, _BaseExampleModel


class Twinframework(BIFMixin, _BaseExampleModel):
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
