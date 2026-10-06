from typing import TYPE_CHECKING

from pgmpy.base._base import _CoreGraph

if TYPE_CHECKING:
    from pgmpy.base import DAG


class ADMG(_CoreGraph):
    """
    A class representing an Acyclic Directed Mixed Graph (ADMG).

    An ADMG allows directed (``"->"``) and bidirected (``"<>"``) edges, where a bidirected edge ``X <> Y`` encodes a
    latent (unobserved) common cause of ``X`` and ``Y``. The directed part is acyclic. Every vertex of an ADMG is
    observed: latent variables are represented only through bidirected edges, so the ``latents`` role cannot be
    assigned.

    Built on :class:`~pgmpy.base._base._CoreGraph`, restricted to the directed/bidirected edge types via
    ``SUPPORTED_EDGE_TYPES`` (no undirected or circle endpoints) and ``SUPPORTS_LATENTS=False``.

    Parameters
    ----------
    edge_list : iterable of tuples, optional
        Edges of the form ``(u, v, edge_type)`` with ``edge_type`` one of ``"->"``, ``"<-"``, ``"<>"``.

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

    References
    ----------
    - :footcite:t:`richardson_2003`
    """

    SUPPORTED_EDGE_TYPES = frozenset(["->", "<-", "<>"])
    SUPPORTS_LATENTS = False

    def to_dag(self, latent_prefix: str = "U") -> "DAG":
        """
        Returns the canonical DAG of the ADMG, with an explicit latent variable for each bidirected edge.

        Each bidirected edge ``X <> Y`` is replaced by a new latent node ``{latent_prefix}_X_Y`` with the edges
        ``{latent_prefix}_X_Y -> X`` and ``{latent_prefix}_X_Y -> Y``. The endpoints in the name are ordered by
        their string representation, so the name does not depend on how the edge was added. Directed edges and
        roles are kept, and the latent projection of the result (:meth:`pgmpy.base.DAG.to_admg`) recovers the ADMG.

        Parameters
        ----------
        latent_prefix : str (default: "U")
            Prefix of the names of the new latent nodes.

        Returns
        -------
        pgmpy.base.DAG
            The canonical DAG, with the new nodes as its ``latents``.

        Raises
        ------
        ValueError
            If the name of a new latent node is already a node of the graph.

        See Also
        --------
        pgmpy.base.DAG.to_admg : The reverse conversion, projecting latent variables into bidirected edges.

        Examples
        --------
        >>> from pgmpy.base import ADMG
        >>> admg = ADMG(edge_list=[("X", "M", "->"), ("M", "Y", "->"), ("X", "Y", "<>")])
        >>> dag = admg.to_dag()
        >>> sorted(dag.edges())
        [('M', 'Y'), ('U_X_Y', 'X'), ('U_X_Y', 'Y'), ('X', 'M')]
        >>> dag.latents
        {'U_X_Y'}

        References
        ----------
        - :footcite:t:`richardson_2003`
        """
        from pgmpy.base import DAG

        dag = DAG()
        dag.add_nodes_from(self.nodes())
        dag.add_edges_from(self.get_directed_graph().edges())
        latents = set()
        for u, v in self.get_edges(data=False, edge_types={"<>"}):
            u, v = sorted((u, v), key=str)
            latent = f"{latent_prefix}_{u}_{v}"
            if latent in dag:
                raise ValueError(
                    f"Cannot add '{latent}' as the latent parent of {u} <> {v}: it is already a node of the graph. "
                    "Use a different `latent_prefix`."
                )
            dag.add_edges_from([(latent, u), (latent, v)])
            latents.add(latent)

        dag.latents = latents
        for role, variables in self.get_role_dict().items():
            dag.with_role(role=role, variables=variables, inplace=True)
        return dag
