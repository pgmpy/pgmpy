from .._base import BIFMixin, _BaseExampleModel


class Student1(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/student1",
        "n_nodes": 26,
        "n_edges": 27,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/student1.bif"
