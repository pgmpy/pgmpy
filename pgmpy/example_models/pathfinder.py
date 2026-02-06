from ._base import DiscreteMixin, _BaseExampleModel


class Pathfinder(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] D. Heckerman, E. Horwitz, and B. Nathwani. Towards Normative Expert Systems: Part I. The Pathfinder
    Project. Methods of Information in Medicine, 31:90-105, 1992.
    """

    _tags = {
        "name": "pathfinder",
        "n_nodes": 109,
        "n_edges": 195,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/pathfinder.bif.gz"
