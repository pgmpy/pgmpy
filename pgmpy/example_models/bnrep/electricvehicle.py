from .._base import PlainBIFMixin, _BaseExampleModel


class Electricvehicle(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/electricvehicle",
        "n_nodes": 23,
        "n_edges": 22,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/electricvehicle.bif"
