from .._base import BIFMixin, _BaseExampleModel


class Covidrisk(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/covidrisk",
        "n_nodes": 9,
        "n_edges": 13,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/covidrisk.bif"
