from ._base import DiscreteMixin, _BaseExampleModel


class Alarm(DiscreteMixin, _BaseExampleModel):
    _tags = {
        "name": "alarm",
        "n_nodes": 37,
        "n_edges": 46,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/alarm.bif.gz"
