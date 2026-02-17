from .._base import PlainBIFMixin, _BaseExampleModel


class Corrosion(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/corrosion",
        "n_nodes": 22,
        "n_edges": 24,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/corrosion.bif"
