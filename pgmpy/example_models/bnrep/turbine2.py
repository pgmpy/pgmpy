from .._base import ContinuousMixin, _BaseExampleModel


class Turbine2(ContinuousMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/turbine2",
        "n_nodes": 16,
        "n_edges": 20,
        "is_parameterized": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_hybrid": False,
    }

    data_url = "bnrep/turbine2.json"
