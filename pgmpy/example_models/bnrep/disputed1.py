from .._base import PlainBIFMixin, _BaseExampleModel


class Disputed1(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/disputed1",
        "n_nodes": 11,
        "n_edges": 11,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/disputed1.bif"
