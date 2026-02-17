from .._base import BIFMixin, _BaseExampleModel


class Softwarelogs4(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/softwarelogs4",
        "n_nodes": 14,
        "n_edges": 21,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/softwarelogs4.bif"
