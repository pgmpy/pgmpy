from .._base import BIFMixin, _BaseExampleModel


class Project(BIFMixin, _BaseExampleModel):
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
