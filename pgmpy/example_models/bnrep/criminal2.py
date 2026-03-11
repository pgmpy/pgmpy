from .._base import BIFMixin, _BaseExampleModel


class Criminal2(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/criminal2",
        "n_nodes": 12,
        "n_edges": 14,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/criminal2.bif"
