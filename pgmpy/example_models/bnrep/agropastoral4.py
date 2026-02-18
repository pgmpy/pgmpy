from .._base import BIFMixin, _BaseExampleModel


class Agropastoral4(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/agropastoral4",
        "n_nodes": 21,
        "n_edges": 14,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/agropastoral4.bif"
