from .._base import ContinuousMixin, _BaseExampleModel


class Building(ContinuousMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/building",
        "n_nodes": 24,
        "n_edges": 32,
        "is_parameterized": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_hybrid": False,
    }

    data_url = "bnrep/building.json"
