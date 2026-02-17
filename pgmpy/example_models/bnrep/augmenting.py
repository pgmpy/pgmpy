from .._base import BIFMixin, _BaseExampleModel


class Augmenting(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/augmenting",
        "n_nodes": 6,
        "n_edges": 6,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/augmenting.bif"
