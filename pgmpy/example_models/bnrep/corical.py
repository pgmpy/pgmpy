from .._base import BIFMixin, _BaseExampleModel


class Corical(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/corical",
        "n_nodes": 20,
        "n_edges": 26,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/corical.bif"
