from .._base import BIFMixin, _BaseExampleModel


class Fire(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/fire",
        "n_nodes": 11,
        "n_edges": 14,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/fire.bif"
