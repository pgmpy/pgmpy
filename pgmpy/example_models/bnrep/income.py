from .._base import BaseExampleModel, BIFMixin


class Income(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/income",
        "n_nodes": 13,
        "n_edges": 20,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/income.bif"
