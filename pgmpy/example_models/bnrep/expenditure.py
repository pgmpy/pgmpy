from .._base import ContinuousMixin, _BaseExampleModel


class Expenditure(ContinuousMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/expenditure",
        "n_nodes": 12,
        "n_edges": 20,
        "is_parameterized": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_hybrid": False,
    }

    data_url = "bnrep/expenditure.json"
