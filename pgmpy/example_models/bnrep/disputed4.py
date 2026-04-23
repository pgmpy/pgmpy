from .._base import BIFMixin, _BaseExampleModel


class Disputed4(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/disputed4",
        "n_nodes": 23,
        "n_edges": 29,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/disputed4.bif"
