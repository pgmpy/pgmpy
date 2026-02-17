from .._base import PlainBIFMixin, _BaseExampleModel


class Vessel1(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/vessel1",
        "n_nodes": 16,
        "n_edges": 17,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/vessel1.bif"
