from .._base import PlainBIFMixin, _BaseExampleModel


class Firealarm(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/firealarm",
        "n_nodes": 6,
        "n_edges": 5,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/firealarm.bif"
