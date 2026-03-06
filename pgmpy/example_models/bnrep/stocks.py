from .._base import ContinuousMixin, _BaseExampleModel


class Stocks(ContinuousMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/stocks",
        "n_nodes": 13,
        "n_edges": 23,
        "is_parameterized": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_hybrid": False,
    }

    data_url = "bnrep/stocks.json"
