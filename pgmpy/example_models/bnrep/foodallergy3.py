from .._base import PlainBIFMixin, _BaseExampleModel


class Foodallergy3(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/foodallergy3",
        "n_nodes": 8,
        "n_edges": 8,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/foodallergy3.bif"
