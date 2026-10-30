from .._base import BaseExampleModel, BIFMixin


class Project(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/project",
        "n_nodes": 21,
        "n_edges": 21,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/project.bif"
