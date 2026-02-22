from .._base import ContinuousMixin, _BaseExampleModel


class Diagnosis(ContinuousMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/diagnosis",
        "n_nodes": 16,
        "n_edges": 15,
        "is_parameterized": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_hybrid": False,
    }

    data_url = "bnrep/diagnosis.json"
