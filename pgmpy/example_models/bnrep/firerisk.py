from .._base import BIFMixin, _BaseExampleModel


class Firerisk(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/firerisk",
        "n_nodes": 23,
        "n_edges": 26,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/firerisk.bif"
