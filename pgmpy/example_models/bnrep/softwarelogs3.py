from .._base import BIFMixin, _BaseExampleModel


class Softwarelogs3(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/softwarelogs3",
        "n_nodes": 14,
        "n_edges": 13,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/softwarelogs3.bif"
