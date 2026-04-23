from .._base import BIFMixin, _BaseExampleModel


class Theft1(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/theft1",
        "n_nodes": 4,
        "n_edges": 3,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/theft1.bif"
