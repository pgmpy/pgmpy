from .._base import PlainBIFMixin, _BaseExampleModel


class Poultry(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/poultry",
        "n_nodes": 47,
        "n_edges": 41,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/poultry.bif"
