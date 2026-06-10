from .._base import BaseExampleModel, ContinuousMixin


class Foodsecurity(ContinuousMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/foodsecurity",
        "n_nodes": 4,
        "n_edges": 4,
        "is_parameterized": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_hybrid": False,
    }

    data_url = "bnrep/foodsecurity.json"
