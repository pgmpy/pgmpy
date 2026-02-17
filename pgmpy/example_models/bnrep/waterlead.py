from .._base import PlainBIFMixin, _BaseExampleModel


class Waterlead(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/waterlead",
        "n_nodes": 17,
        "n_edges": 31,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/waterlead.bif"
