from .._base import BIFMixin, _BaseExampleModel


class Engines(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/engines",
        "n_nodes": 12,
        "n_edges": 18,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/engines.bif"
