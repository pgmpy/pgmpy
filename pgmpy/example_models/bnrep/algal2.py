from .._base import ContinuousMixin, _BaseExampleModel


class Algal2(ContinuousMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/algal2",
        "n_nodes": 9,
        "n_edges": 9,
        "is_parameterized": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_hybrid": False,
    }

    data_url = "bnrep/algal2.json"
