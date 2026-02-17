from .._base import PlainBIFMixin, _BaseExampleModel


class Cardiovascular(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/cardiovascular",
        "n_nodes": 13,
        "n_edges": 40,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/cardiovascular.bif"
