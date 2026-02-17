from .._base import PlainBIFMixin, _BaseExampleModel


class Disputed3(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/disputed3",
        "n_nodes": 27,
        "n_edges": 34,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/disputed3.bif"
