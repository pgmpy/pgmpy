from .._base import BIFMixin, _BaseExampleModel


class Concrete3(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/concrete3",
        "n_nodes": 3,
        "n_edges": 2,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/concrete3.bif"
