from .._base import BIFMixin, _BaseExampleModel


class Titanic(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/titanic",
        "n_nodes": 4,
        "n_edges": 5,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/titanic.bif"
