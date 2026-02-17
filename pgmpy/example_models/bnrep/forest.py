from .._base import BIFMixin, _BaseExampleModel


class Forest(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/forest",
        "n_nodes": 80,
        "n_edges": 121,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/forest.bif"
