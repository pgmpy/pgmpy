from .._base import BaseExampleModel, ContinuousMixin


class Lexical(ContinuousMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/lexical",
        "n_nodes": 8,
        "n_edges": 14,
        "is_parameterized": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_hybrid": False,
    }

    data_url = "bnrep/lexical.json"
