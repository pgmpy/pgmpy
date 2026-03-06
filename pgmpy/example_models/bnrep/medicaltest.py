from .._base import BIFMixin, _BaseExampleModel


class Medicaltest(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/medicaltest",
        "n_nodes": 3,
        "n_edges": 2,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/medicaltest.bif"
