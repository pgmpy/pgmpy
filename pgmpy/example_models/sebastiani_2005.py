from ._base import DAGMixin, _BaseExampleModel


class Sebastiani2005(DAGMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] Sebastiani, P., Ramoni, M., Nolan, V., Baldwin, C. T., & Steinberg, M. H. (2005).
    Genetic dissection and prognostic modeling of overt stroke in sickle cell anemia.
    Nature Genetics, 37(4), 435–440.
    """

    _tags = {
        "name": "sebastiani_2005",
        "n_nodes": 36,
        "n_edges": 60,
        "is_parameterized": False,
    }
    data_url = "dags/Sebastiani_2005.txt"
