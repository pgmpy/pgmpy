from .._base import BIFMixin, _BaseExampleModel


class Concrete2(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/concrete2",
        "n_nodes": 4,
        "n_edges": 4,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/concrete2.bif"
