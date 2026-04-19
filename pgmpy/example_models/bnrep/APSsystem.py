from .._base import BIFMixin, _BaseExampleModel


class APSsystem(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/APSsystem",
        "n_nodes": 10,
        "n_edges": 9,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/APSsystem.bif"
