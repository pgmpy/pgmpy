from .._base import PlainBIFMixin, _BaseExampleModel


class Realestate3(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/realestate3",
        "n_nodes": 27,
        "n_edges": 69,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/realestate3.bif"
