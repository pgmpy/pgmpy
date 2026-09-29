from .._base import BaseExampleModel, BIFMixin


class Salmonella2(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/salmonella2",
        "n_nodes": 10,
        "n_edges": 10,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/salmonella2.bif"
