from .._base import BIFMixin, _BaseExampleModel


class Concrete7(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/concrete7",
        "n_nodes": 5,
        "n_edges": 4,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/concrete7.bif"
