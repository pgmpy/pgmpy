from .._base import ContinuousMixin, _BaseExampleModel


class Cachexia2(ContinuousMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/cachexia2",
        "n_nodes": 6,
        "n_edges": 8,
        "is_parameterized": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_hybrid": False,
    }

    data_url = "bnrep/cachexia2.json"
