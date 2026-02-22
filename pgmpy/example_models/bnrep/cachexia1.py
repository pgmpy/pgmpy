from .._base import ContinuousMixin, _BaseExampleModel


class Cachexia1(ContinuousMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/cachexia1",
        "n_nodes": 6,
        "n_edges": 8,
        "is_parameterized": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_hybrid": False,
    }

    data_url = "bnrep/cachexia1.json"
