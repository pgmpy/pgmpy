from .._base import BIFMixin, _BaseExampleModel


class Lawschool(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/lawschool",
        "n_nodes": 10,
        "n_edges": 21,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/lawschool.bif"
