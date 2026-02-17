from .._base import BIFMixin, _BaseExampleModel


class Wheat(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/wheat",
        "n_nodes": 5,
        "n_edges": 4,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/wheat.bif"
