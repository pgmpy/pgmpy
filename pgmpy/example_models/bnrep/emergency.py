from .._base import BIFMixin, _BaseExampleModel


class Emergency(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/emergency",
        "n_nodes": 17,
        "n_edges": 16,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/emergency.bif"
