from .._base import BIFMixin, _BaseExampleModel


class Agropastoral1(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/agropastoral1",
        "n_nodes": 15,
        "n_edges": 12,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/agropastoral1.bif"
