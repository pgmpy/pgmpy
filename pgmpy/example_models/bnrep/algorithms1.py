from .._base import ContinuousMixin, _BaseExampleModel


class Algorithms1(ContinuousMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/algorithms1",
        "n_nodes": 4,
        "n_edges": 3,
        "is_parameterized": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_hybrid": False,
    }

    data_url = "bnrep/algorithms1.json"
