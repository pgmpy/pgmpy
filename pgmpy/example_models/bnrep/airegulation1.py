from .._base import BIFMixin, _BaseExampleModel


class Airegulation1(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/airegulation1",
        "n_nodes": 22,
        "n_edges": 42,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/airegulation1.bif"
