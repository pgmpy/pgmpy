from .._base import PlainBIFMixin, _BaseExampleModel


class Student2(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/student2",
        "n_nodes": 26,
        "n_edges": 23,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/student2.bif"
