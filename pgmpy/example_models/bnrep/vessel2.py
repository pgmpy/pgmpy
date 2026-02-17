from .._base import PlainBIFMixin, _BaseExampleModel


class Vessel2(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/vessel2",
        "n_nodes": 13,
        "n_edges": 18,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/vessel2.bif"
