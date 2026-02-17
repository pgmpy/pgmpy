from .._base import PlainBIFMixin, _BaseExampleModel


class Crimescene(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/crimescene",
        "n_nodes": 9,
        "n_edges": 13,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/crimescene.bif"
