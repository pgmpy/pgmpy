from typing import Hashable, Iterable, Optional

from pgmpy.base import AncestralBase


class PAG(AncestralBase):
    """
    Class for representing Partial Ancestral Graphs (PAGs).

    A PAG is a type of graph used in causal inference to represent a set of
    Ancestral Graphs (MAGs) that are Markov equivalent. PAGs allow for
    additional edge types including circle endpoints ('o'), which represent
    uncertainty about whether an edge is directed or bidirected.

    Parameters
    ----------
    ebunch : iterable of tuples, optional
        A list or iterable of edges to add at initialization.
        Each tuple should be of the form (u, v, u_mark, v_mark).
        Marks must be one of (">", "-", "o").

    latents : set, default=set()
        Set of latent (unobserved) variables.

    exposures : set, default=set()
        Set of exposure variables in the graph. Default is an empty set.

    outcomes : set, default=set()
        Set of outcome variables in the graph. Default is an empty set.

    roles : dict, optional (default: None)
        A dictionary mapping roles to node names.

    Examples
    --------
    >>> from pgmpy.base import PAG
    >>> pag = PAG(ebunch=[("X", "Y", "o", ">"), ("Z", "Y", "o", ">")])
    >>> pag.edges[("X", "Y")]["marks"]
    {'X': 'o', 'Y': '>'}
    """

    def __init__(
        self,
        ebunch: Optional[Iterable[tuple[Hashable, Hashable]]] = None,
        latents: set[Hashable] = set(),
        exposures: set[Hashable] = set(),
        outcomes: set[Hashable] = set(),
        roles=None,
    ):
        super().__init__(
            ebunch=ebunch,
            latents=latents,
            exposures=exposures,
            outcomes=outcomes,
            roles=roles,
        )
