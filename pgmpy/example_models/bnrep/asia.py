from .._base import BIFMixin, _BaseExampleModel


class Asia(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/asia",
        "n_nodes": 8,
        "n_edges": 8,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/asia.bif"
