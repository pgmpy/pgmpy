from .._base import BIFMixin, _BaseExampleModel


class Tbm(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/tbm",
        "n_nodes": 10,
        "n_edges": 15,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/tbm.bif"
