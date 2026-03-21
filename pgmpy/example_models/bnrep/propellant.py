from .._base import BIFMixin, _BaseExampleModel


class Propellant(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/propellant",
        "n_nodes": 49,
        "n_edges": 48,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/propellant.bif"
