from .._base import BIFMixin, _BaseExampleModel


class Windturbine(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/windturbine",
        "n_nodes": 122,
        "n_edges": 123,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/windturbine.bif"
