from pgmpy.base._base import _CoreGraph


class ADMG(_CoreGraph):
    """
    A class representing an Acyclic Directed Mixed Graph (ADMG).

    An ADMG allows directed (``"->"``) and bidirected (``"<>"``) edges, where a bidirected edge
    ``X <> Y`` encodes a latent (unobserved) common cause of ``X`` and ``Y``. The directed part is
    acyclic.

    Built on :class:`~pgmpy.base._base._CoreGraph`, restricted to the directed/bidirected edge types
    via ``SUPPORTED_EDGE_TYPES`` (no undirected or circle endpoints).

    Parameters
    ----------
    edge_list : iterable of tuples, optional
        Edges of the form ``(u, v, edge_type)`` with ``edge_type`` one of ``"->"``, ``"<-"``, ``"<>"``.

    latents : set, default=set()
        Set of latent (unobserved) variables.

    exposures, outcomes : set, default=set()
        Treatment / response variables (causal-analysis roles).

    roles : dict, optional (default: None)
        A mapping of role name to node(s); equivalent to calling ``with_role`` for each entry.

    Examples
    --------
    >>> from pgmpy.base import ADMG
    >>> admg = ADMG(edge_list=[("X", "Y", "->"), ("Z", "Y", "->"), ("X", "Z", "<>")])
    >>> sorted(admg.nodes())
    ['X', 'Y', 'Z']
    >>> sorted(admg.get_edges(data=True))
    [('X', 'Y', '->'), ('X', 'Z', '<>'), ('Z', 'Y', '->')]
    """

    SUPPORTED_EDGE_TYPES = frozenset(["->", "<-", "<>"])
