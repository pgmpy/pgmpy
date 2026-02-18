from .._base import BIFMixin, _BaseExampleModel


class Enrollment(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/enrollment",
        "n_nodes": 8,
        "n_edges": 7,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/enrollment.bif"
