from .._base import ContinuousMixin, _BaseExampleModel


class Suffocation(ContinuousMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/suffocation",
        "n_nodes": 35,
        "n_edges": 34,
        "is_parameterized": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_hybrid": False,
    }

    data_url = "bnrep/suffocation.json"
