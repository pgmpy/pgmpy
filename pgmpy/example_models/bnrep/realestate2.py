from .._base import PlainBIFMixin, _BaseExampleModel


class Realestate2(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/realestate2",
        "n_nodes": 27,
        "n_edges": 81,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/realestate2.bif"
