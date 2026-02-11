from ._base import DiscreteMixin, _BaseExampleModel


class Water(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] F. V. Jensen, U. Kjærulff, K. G. Olesen and J. Pedersen. Et Forprojekt Til et Ekspertsystem for Drift af
    Spildevandsrensning (An Expert System for Control of Waste Water Treatment - A Pilot Project). Technical Report,
    Judex Datasystemer A/S, Aalborg, 1989. In Danish.
    """

    _tags = {
        "name": "water",
        "n_nodes": 32,
        "n_edges": 66,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/water.bif.gz"
