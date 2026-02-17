from .._base import PlainBIFMixin, _BaseExampleModel


class Income(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/income",
        "n_nodes": 13,
        "n_edges": 19,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/income.bif"
