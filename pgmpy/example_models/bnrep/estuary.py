from .._base import BIFMixin, _BaseExampleModel


class Estuary(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/estuary",
        "n_nodes": 30,
        "n_edges": 44,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/estuary.bif"
