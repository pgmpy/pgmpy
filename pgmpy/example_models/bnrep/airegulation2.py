from .._base import BIFMixin, _BaseExampleModel


class Airegulation2(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/airegulation2",
        "n_nodes": 19,
        "n_edges": 36,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/airegulation2.bif"
