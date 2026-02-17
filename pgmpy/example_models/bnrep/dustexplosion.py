from .._base import PlainBIFMixin, _BaseExampleModel


class Dustexplosion(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/dustexplosion",
        "n_nodes": 26,
        "n_edges": 32,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/dustexplosion.bif"
