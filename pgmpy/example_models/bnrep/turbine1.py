from .._base import BaseExampleModel, ContinuousMixin


class Turbine1(ContinuousMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/turbine1",
        "n_nodes": 16,
        "n_edges": 20,
        "is_parameterized": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_hybrid": False,
    }

    data_url = "bnrep/turbine1.json"
