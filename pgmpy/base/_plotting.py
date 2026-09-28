from collections.abc import Hashable

import networkx as nx


class _GraphPlotting:
    """Plotting / export methods (graphviz, daft) for ``_CoreGraph``-based classes."""

    def to_graphviz(self):
        """
        Returns a pygraphviz object for the graph for visualizing the network structure.

        Returns
        -------
        pygraphviz.AGraph
            A directed pygraphviz graph encoding the edge orientations as arrow styles.

        Examples
        --------
        >>> from pgmpy.base import ADMG
        >>> admg = ADMG(edge_list=[("A", "B", "->"), ("B", "C", "<>")])
        >>> admg.to_graphviz()  # doctest: +SKIP
        <AGraph ...>
        """
        marker_to_arrow = {"-": "none", ">": "normal", "o": "odot"}

        graph = nx.MultiDiGraph()
        graph.add_nodes_from(self.nodes())
        for u, v, markers in self.edges(data=True):
            graph.add_edge(
                u,
                v,
                dir="both",
                arrowtail=marker_to_arrow[markers[u]],
                arrowhead=marker_to_arrow[markers[v]],
            )
        return nx.nx_agraph.to_agraph(graph)

    def to_daft(
        self,
        node_pos: str | dict[Hashable, tuple[float, float]] = "circular",
        latex: bool = True,
        pgm_params: dict | None = None,
        node_params: dict | None = None,
        edge_params: dict | None = None,
    ):
        """
        Returns a daft (https://docs.daft-pgm.org/) object for publication-quality plots.

        daft only supports directed and undirected edges, so the graph may contain only ``"->"``,
        ``"<-"`` and ``"--"`` edges. A ``ValueError`` is raised for any bidirected (``"<>"``) or
        circle-endpoint edge, which daft cannot represent.

        Parameters
        ----------
        node_pos : str or dict (default: "circular")
            If a string, the name of a networkx layout used to position the nodes; one of
            ``"circular"``, ``"kamada_kawai"``, ``"planar"``, ``"random"``, ``"shell"``,
            ``"spring"``, ``"spectral"``, ``"spiral"``. If a dict, a mapping from each node to its
            ``(x, y)`` position.

        latex : bool (default: True)
            If True, node labels are wrapped in ``$...$`` so they are rendered as LaTeX.

        pgm_params : dict, optional
            Keyword arguments passed to ``daft.PGM``.

        node_params : dict, optional
            A mapping from a node to a dict of keyword arguments passed to ``daft.PGM.add_node`` for
            that node.

        edge_params : dict, optional
            A mapping from an ``(u, v)`` edge to a dict of keyword arguments passed to
            ``daft.PGM.add_edge`` for that edge.

        Returns
        -------
        daft.PGM
            The daft probabilistic graphical model object.

        Raises
        ------
        ValueError
            If ``node_pos`` is not a known layout name or a complete position dict, or if the graph
            contains an edge type that daft cannot represent.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> graph = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "--")])
        >>> graph.to_daft(node_pos="circular")  # doctest: +SKIP
        <daft.PGM ...>
        """
        # Step 0: Check the arguments
        try:
            from daft import PGM
        except ImportError as e:
            raise ImportError(
                f"{e}. Package `daft` is required for plotting probabilistic graphical models.\n"
                "Please install it using: pip install daft-pgm\n"
                "Documentation: https://docs.daft-pgm.org/en/latest/"
            ) from None

        _SUPPORTED_LAYOUTS = {
            "circular": nx.circular_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "planar": nx.planar_layout,
            "random": nx.random_layout,
            "shell": nx.shell_layout,
            "spring": nx.spring_layout,
            "spectral": nx.spectral_layout,
            "spiral": nx.spiral_layout,
        }

        if isinstance(node_pos, str):
            if node_pos not in _SUPPORTED_LAYOUTS:
                raise ValueError("Unknown node_pos argument. Please refer docstring for accepted values.")
            node_pos = _SUPPORTED_LAYOUTS[node_pos](self)
        elif isinstance(node_pos, dict):
            for node in self.nodes():
                if node not in node_pos:
                    raise ValueError(f"No position specified for {node}.")
        else:
            raise ValueError("Argument node_pos not valid. Please refer to the docstring.")

        # Step 1: Preprocess arguments
        pgm_params = {} if pgm_params is None else pgm_params
        node_params = {} if node_params is None else node_params
        edge_params = {} if edge_params is None else edge_params

        # Step 2: Construct the PGM object and add the nodes.
        daft_pgm = PGM(**pgm_params)
        for node in self.nodes():
            label = rf"${node}$" if latex else f"{node}"
            daft_pgm.add_node(
                node,
                label,
                node_pos[node][0],
                node_pos[node][1],
                observed=node in self.observed,
                **node_params.get(node, {}),
            )

        # Step 3: Add the edges to the graph. `get_edges` returns canonical orientations, so "->" (with
        #         its nodes already in source -> target order) covers every directed edge.
        for u, v, edge_type in self.get_edges(data=True):
            extra = edge_params.get((u, v), {})
            if edge_type == "->":
                daft_pgm.add_edge(u, v, directed=True, **extra)
            elif edge_type == "--":
                daft_pgm.add_edge(u, v, directed=False, **extra)
            else:
                raise ValueError(
                    f"daft cannot represent the edge type {edge_type!r} between {u} and {v}; "
                    "to_daft only supports directed ('->') and undirected ('--') edges."
                )

        # Step 4: Return the constructed daft graph.
        return daft_pgm
