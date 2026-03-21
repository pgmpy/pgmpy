from .._base import ContinuousMixin, _BaseExampleModel


class Lexical(ContinuousMixin, _BaseExampleModel):
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
