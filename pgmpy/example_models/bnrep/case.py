from .._base import BIFMixin, _BaseExampleModel


class Case(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/case",
        "n_nodes": 23,
        "n_edges": 35,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/case.bif"
