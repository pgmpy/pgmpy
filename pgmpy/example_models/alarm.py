from ._base import DiscreteMixin, _BaseExampleModel


class Alarm(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] I. A. Beinlich, H. J. Suermondt, R. M. Chavez, and G. F. Cooper. The ALARM Monitoring System: A Case Study
    with Two Probabilistic Inference Techniques for Belief Networks. In Proceedings of the 2nd European Conference on
    Artificial Intelligence in Medicine, pages 247-256. Springer-Verlag, 1989.
    """

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
