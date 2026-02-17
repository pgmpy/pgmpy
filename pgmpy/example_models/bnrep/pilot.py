from .._base import BIFMixin, _BaseExampleModel


class Pilot(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/pilot",
        "n_nodes": 42,
        "n_edges": 41,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/pilot.bif"
