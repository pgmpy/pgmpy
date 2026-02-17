from .._base import PlainBIFMixin, _BaseExampleModel


class Realestate1(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/realestate1",
        "n_nodes": 27,
        "n_edges": 77,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/realestate1.bif"
