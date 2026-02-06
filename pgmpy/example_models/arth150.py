from ._base import ContinuousMixin, _BaseExampleModel


class arth150(ContinuousMixin, _BaseExampleModel):

    _tags = {
        "name": "arth150",
        "n_nodes": 107,
        "n_edges": 150,
        "is_parameterized": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_hybrid": False,
    }
    data_url = "continuous/arth150.json"
