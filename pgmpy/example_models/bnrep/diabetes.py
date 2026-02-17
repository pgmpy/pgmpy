from .._base import PlainBIFMixin, _BaseExampleModel


class Diabetes(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/diabetes",
        "n_nodes": 9,
        "n_edges": 8,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/diabetes.bif"
