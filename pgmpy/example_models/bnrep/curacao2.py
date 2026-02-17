from .._base import BIFMixin, _BaseExampleModel


class Curacao2(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/curacao2",
        "n_nodes": 13,
        "n_edges": 12,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/curacao2.bif"
