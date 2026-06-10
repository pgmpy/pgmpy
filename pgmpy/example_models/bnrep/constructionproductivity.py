from .._base import BaseExampleModel, BIFMixin


class Constructionproductivity(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/constructionproductivity",
        "n_nodes": 18,
        "n_edges": 19,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/constructionproductivity.bif"
