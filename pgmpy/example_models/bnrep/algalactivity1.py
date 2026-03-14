from .._base import BIFMixin, _BaseExampleModel


class Algalactivity1(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/algalactivity1",
        "n_nodes": 8,
        "n_edges": 10,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/algalactivity1.bif"
