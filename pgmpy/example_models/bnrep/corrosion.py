from .._base import BIFMixin, _BaseExampleModel


class Corrosion(BIFMixin, _BaseExampleModel):
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
