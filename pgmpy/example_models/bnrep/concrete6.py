from .._base import BIFMixin, _BaseExampleModel


class Concrete6(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/concrete6",
        "n_nodes": 3,
        "n_edges": 2,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/concrete6.bif"
