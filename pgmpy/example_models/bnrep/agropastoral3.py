from .._base import PlainBIFMixin, _BaseExampleModel


class Agropastoral3(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/agropastoral3",
        "n_nodes": 11,
        "n_edges": 15,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/agropastoral3.bif"
