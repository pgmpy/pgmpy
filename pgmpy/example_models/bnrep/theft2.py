from .._base import BIFMixin, _BaseExampleModel


class Theft2(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/theft2",
        "n_nodes": 5,
        "n_edges": 4,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/theft2.bif"
