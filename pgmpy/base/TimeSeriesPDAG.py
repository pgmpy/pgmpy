#!/usr/bin/env python

import networkx as nx
import itertools

from pgmpy.base import DAG, PDAG, TimeSeriesDAG


class TimeSeriesPDAG(PDAG):
    """
    Class for representing partially directed acyclic graphs for time series data.

    This class extends the PDAG class to include time series data with lagged relationships.
    Each node is represented as a tuple (variable, lag), where lag is an integer
    indicating the lag relative to the current timepoint.

    In TimeSeriesPDAG, both directed and undirected edges are supported, allowing
    for representation of causal relationships that may be uncertain in their direction.

    Parameters
    ----------
    directed_ebunch: list, array-like of 2-tuples
        List of directed edges in the PDAG. Each edge should be in the form
        ((var1, lag1), (var2, lag2)).

    undirected_ebunch: list, array-like of 2-tuples
        List of undirected edges in the PDAG. Each edge should be in the form
        ((var1, lag1), (var2, lag2)).

    latents: list, array-like
        List of nodes which are latent variables. Each node should be a tuple (var, lag).

    Examples
    --------
    >>> directed_edges = [(('A', 0), ('B', 1)), (('B', 1), ('C', 2))]
    >>> undirected_edges = [(('A', 0), ('C', 0)), (('B', 0), ('D', 1))]
    >>> tspdag = TimeSeriesPDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)
    """

    def __init__(self, directed_ebunch=None, undirected_ebunch=None, latents=None):
        # Initialize with empty lists if None is provided
        directed_ebunch = [] if directed_ebunch is None else directed_ebunch
        undirected_ebunch = [] if undirected_ebunch is None else undirected_ebunch
        latents = [] if latents is None else latents

        # Validate all edges before calling parent constructor
        for u, v in directed_ebunch:
            self._validate_node(u, "Source")
            self._validate_node(v, "Target")
            self._validate_temporal_relationship(u, v)

        for u, v in undirected_ebunch:
            self._validate_node(u, "Source")
            self._validate_node(v, "Target")
            # For undirected edges, we only validate temporality if same timepoint
            if u[1] == v[1]:
                pass  # This is fine, two variables at same timepoint can have undirected edge
            else:
                self._validate_temporal_relationship(u, v)
                # Also validate the other direction for undirected edges
                try:
                    self._validate_temporal_relationship(v, u)
                except ValueError:
                    # If validation fails in reverse, we should enforce directed edge
                    raise ValueError(
                        f"Edge between {u} and {v} cannot be undirected due to temporal constraints. "
                        f"Consider adding it as a directed edge."
                    )

        # Validate latent nodes
        for node in latents:
            self._validate_node(node, "Latent")

        # Call parent constructor with validated edges
        super(TimeSeriesPDAG, self).__init__(
            directed_ebunch=directed_ebunch,
            undirected_ebunch=undirected_ebunch,
            latents=latents,
        )

    def _validate_node(self, node, node_type=""):
        """Validate that a node has the correct format (variable, lag)."""
        if not (
            isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], int)
        ):
            raise ValueError(f"{node_type} node should be a tuple (variable, lag).")

    def _validate_temporal_relationship(self, u, v):
        """Validate the temporal relationship between two nodes."""
        if u[1] > v[1]:
            raise ValueError(
                f"The lag of the source node {u} should be less than or equal "
                f"to the lag of the target node {v}."
            )

    def add_edge(self, u, v, directed=True, **kwargs):
        """
        Adds an edge between the nodes u and v.

        These nodes will be automatically added if they are
        not already present in the graph.

        Parameters
        ----------
        u, v : nodes
            Nodes should be tuples (variable, lag).

        directed : bool, default=True
            If True, adds a directed edge. If False, adds an undirected edge.

        **kwargs : keyword arguments
            Additional attributes to add to the edge.

        Examples
        --------
        >>> tspdag = TimeSeriesPDAG()
        >>> tspdag.add_edge(('A', 0), ('B', 1))  # Add directed edge
        >>> tspdag.add_edge(('C', 0), ('D', 0), directed=False)  # Add undirected edge
        """
        # Validate nodes
        self._validate_node(u, "Source")
        self._validate_node(v, "Target")

        # Validate temporal relationship
        self._validate_temporal_relationship(u, v)

        # For undirected edges, additional checks
        if not directed:
            # If different timepoints, check reverse direction
            if u[1] != v[1]:
                try:
                    self._validate_temporal_relationship(v, u)
                except ValueError:
                    raise ValueError(
                        f"Edge between {u} and {v} cannot be undirected due to temporal constraints. "
                        f"Consider adding it as a directed edge."
                    )

            # Add edge to appropriate collection
            if u not in self.undirected_edges:
                self.undirected_edges.add((u, v))

            # Add both directions to the graph
            super(PDAG, self).add_edge(u, v, **kwargs)
            super(PDAG, self).add_edge(v, u, **kwargs)
        else:
            # Add to directed edges collection
            if (u, v) not in self.directed_edges:
                self.directed_edges.add((u, v))

            # Add directed edge to graph
            super(PDAG, self).add_edge(u, v, **kwargs)

    def add_edges_from(self, ebunch, directed=True, **kwargs):
        """
        Add all the edges in ebunch.

        Parameters
        ----------
        ebunch : container of edges
            Each edge given in the container will be added to the graph.
            The edges must be given as 2-tuples (u, v).

        directed : bool, default=True
            If True, adds directed edges. If False, adds undirected edges.

        **kwargs : keyword arguments
            Additional attributes to add to the edges.

        Examples
        --------
        >>> tspdag = TimeSeriesPDAG()
        >>> directed_edges = [(('A', 0), ('B', 1)), (('B', 1), ('C', 2))]
        >>> tspdag.add_edges_from(directed_edges)
        >>> undirected_edges = [(('A', 0), ('C', 0)), (('B', 0), ('D', 0))]
        >>> tspdag.add_edges_from(undirected_edges, directed=False)
        """
        for u, v in ebunch:
            self.add_edge(u, v, directed=directed, **kwargs)

    def to_dag(self):
        """
        Returns one possible TimeSeriesDAG which is represented using the TimeSeriesPDAG.

        Returns
        -------
        TimeSeriesDAG
            Returns an instance of TimeSeriesDAG with all edges directed.

        Examples
        --------
        >>> tspdag = TimeSeriesPDAG(directed_ebunch=[(('A', 0), ('B', 1))],
        ...                         undirected_ebunch=[(('A', 0), ('C', 0))])
        >>> ts_dag = tspdag.to_dag()
        """
        # First get a DAG from the PDAG using the parent class method
        dag = super(TimeSeriesPDAG, self).to_dag()

        # Create a new TimeSeriesDAG with the edges from the DAG
        ts_dag = TimeSeriesDAG(edges=dag.edges())

        # Copy latent nodes
        ts_dag.latents = dag.latents.copy()

        return ts_dag

    def to_summary_graph(self, include_undirected=True):
        """
        Converts the time series PDAG to a summary graph where each variable
        appears only once, and the edges represent the existence of a causal
        link at any lag.

        Parameters
        ----------
        include_undirected : bool, default=True
            If True, includes undirected edges in the summary graph.

        Returns
        -------
        summary_graph : nx.DiGraph or nx.Graph
            A graph where each node is a variable and edges represent
            the existence of a causal link at any lag. If include_undirected is True,
            the result will be a mixed graph with both directed and undirected edges.

        Examples
        --------
        >>> tspdag = TimeSeriesPDAG(directed_ebunch=[(('A', 0), ('B', 1))],
        ...                         undirected_ebunch=[(('A', 0), ('C', 0))])
        >>> summary = tspdag.to_summary_graph()
        """
        # Use a mixed graph to represent both directed and undirected edges
        summary_graph = nx.DiGraph()

        # Add all the variables as nodes
        variables = set(var for var, _ in self.nodes())
        summary_graph.add_nodes_from(variables)

        # Add directed edges with lag information
        for (var1, lag1), (var2, lag2) in self.directed_edges:
            if not summary_graph.has_edge(var1, var2):
                summary_graph.add_edge(
                    var1, var2, directed_lags=set(), undirected_lags=set()
                )
            summary_graph[var1][var2]["directed_lags"].add((lag1, lag2))

        # Add undirected edges if requested
        if include_undirected:
            for (var1, lag1), (var2, lag2) in self.undirected_edges:
                # For undirected edges, ensure we're not duplicating
                if var1 == var2:
                    continue

                # Determine edge direction in summary graph
                if var1 < var2:  # Use lexicographical ordering for consistency
                    v1, v2 = var1, var2
                    l1, l2 = lag1, lag2
                else:
                    v1, v2 = var2, var1
                    l1, l2 = lag2, lag1

                # Add edge if it doesn't exist
                if not summary_graph.has_edge(v1, v2):
                    summary_graph.add_edge(
                        v1, v2, directed_lags=set(), undirected_lags=set()
                    )

                # Add the lag information
                summary_graph[v1][v2]["undirected_lags"].add((l1, l2))

        return summary_graph

    def plot(self, **kwargs):
        """
        Plots the time series PDAG using networkx.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to be passed to the networkx drawing function.

        Returns
        -------
        fig, ax : matplotlib figure and axis
            The matplotlib figure and axis

        Examples
        --------
        >>> tspdag = TimeSeriesPDAG(directed_ebunch=[(('A', 0), ('B', 1))],
        ...                         undirected_ebunch=[(('A', 0), ('C', 0))])
        >>> fig, ax = tspdag.plot()
        """
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines

        # Create a layout for the nodes
        pos = {}
        for node in self.nodes():
            var, lag = node
            # Use a hash value for y to separate their positions
            pos[node] = (lag, hash(var) % 100)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw nodes
        nx.draw_networkx_nodes(self, pos, ax=ax, **kwargs.get("node_kwargs", {}))
        nx.draw_networkx_labels(self, pos, ax=ax, **kwargs.get("label_kwargs", {}))

        # Draw directed edges
        directed_edges = list(self.directed_edges)
        if directed_edges:
            nx.draw_networkx_edges(
                self,
                pos,
                edgelist=directed_edges,
                ax=ax,
                arrows=True,
                **kwargs.get("directed_edge_kwargs", {}),
            )

        # Draw undirected edges with custom style to make them distinguishable
        undirected_edges = []
        for u, v in self.undirected_edges:
            # Only include one direction for plotting
            if (v, u) not in undirected_edges:
                undirected_edges.append((u, v))

        if undirected_edges:
            nx.draw_networkx_edges(
                self,
                pos,
                edgelist=undirected_edges,
                ax=ax,
                arrows=False,
                style="dashed",
                **kwargs.get("undirected_edge_kwargs", {}),
            )

        # Label the lags on x-axis
        lags = sorted(set(lag for _, lag in self.nodes()))
        ax.set_xticks(lags)
        ax.set_xticklabels([f"t{l}" if l == 0 else f"t{l}" for l in lags])
        ax.set_title("Time Series Partially Directed Acyclic Graph")

        # Add a legend
        directed_line = mlines.Line2D(
            [], [], color="black", marker=">", markersize=10, label="Directed Edge"
        )
        undirected_line = mlines.Line2D(
            [], [], color="black", linestyle="dashed", label="Undirected Edge"
        )
        ax.legend(handles=[directed_line, undirected_line])

        return fig, ax

    def get_ancestral_graph(self, nodes):
        """
        Returns the ancestral graph of the given `nodes` in the time series PDAG.

        The ancestral graph only contains the nodes which are ancestors of at least
        one of the variables in nodes.

        Parameters
        ----------
        nodes: iterable
            List of nodes whose ancestral graph needs to be computed.
            Each node should be a tuple (variable, lag).

        Returns
        -------
        TimeSeriesPDAG
            A new TimeSeriesPDAG containing only the ancestors of the specified nodes.

        Examples
        --------
        >>> tspdag = TimeSeriesPDAG(
        ...     directed_ebunch=[(('A', 0), ('B', 1)), (('B', 1), ('C', 2))],
        ...     undirected_ebunch=[(('A', 0), ('D', 0))]
        ... )
        >>> ancestral = tspdag.get_ancestral_graph([('C', 2)])
        """
        # Find all ancestors
        ancestors = set()
        for node in nodes:
            # For each node, trace back through both directed and undirected edges
            visited = set()
            stack = [node]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue

                visited.add(current)
                ancestors.add(current)

                # Add predecessors (both from directed and undirected edges)
                stack.extend(self.predecessors(current))

        # Create a new TimeSeriesPDAG with only the ancestors
        directed_ancestors = [
            (u, v) for u, v in self.directed_edges if u in ancestors and v in ancestors
        ]
        undirected_ancestors = [
            (u, v)
            for u, v in self.undirected_edges
            if u in ancestors and v in ancestors
        ]
        latent_ancestors = [node for node in self.latents if node in ancestors]

        return TimeSeriesPDAG(
            directed_ebunch=directed_ancestors,
            undirected_ebunch=undirected_ancestors,
            latents=latent_ancestors,
        )

    def get_markov_blanket(self, node):
        """
        Returns the Markov blanket for a node in the TimeSeriesPDAG.

        In the context of a PDAG, the Markov blanket of a node includes:
        - Parents (directed edges coming in)
        - Children (directed edges going out)
        - Neighbors (connected by undirected edges)
        - Other parents of children

        Parameters
        ----------
        node: tuple
            The node whose Markov blanket is to be determined.
            Should be in the form (variable, lag).

        Returns
        -------
        list
            List of nodes in the Markov blanket.

        Examples
        --------
        >>> tspdag = TimeSeriesPDAG(
        ...     directed_ebunch=[(('A', 0), ('C', 1)), (('B', 0), ('C', 1))],
        ...     undirected_ebunch=[(('A', 0), ('D', 0))]
        ... )
        >>> mb = tspdag.get_markov_blanket(('A', 0))
        """
        if node not in self.nodes():
            raise ValueError(f"Node {node} not found in the graph")

        markov_blanket = set()

        # Get parents (direct predecessors that are not neighbors)
        parents = set()
        for predecessor in self.predecessors(node):
            if node not in self.predecessors(predecessor):  # Not an undirected edge
                parents.add(predecessor)

        # Get children (direct successors that are not neighbors)
        children = set()
        for successor in self.successors(node):
            if node not in self.successors(successor):  # Not an undirected edge
                children.add(successor)

        # Get neighbors (connected by undirected edges)
        neighbors = set()
        for possible_neighbor in self.predecessors(node):
            if node in self.predecessors(possible_neighbor):  # Undirected edge
                neighbors.add(possible_neighbor)

        # Add parents, children, and neighbors to the Markov blanket
        markov_blanket.update(parents)
        markov_blanket.update(children)
        markov_blanket.update(neighbors)

        # Add other parents of children
        for child in children:
            for parent in self.predecessors(child):
                if parent != node:
                    markov_blanket.add(parent)

        return list(markov_blanket)

    def is_dconnected(self, start, end, observed=None):
        """
        Determines if there is a d-connecting path between start and end nodes,
        given that observed nodes are observed.

        Parameters
        ----------
        start: tuple
            The starting node, in the form (variable, lag).

        end: tuple
            The ending node, in the form (variable, lag).

        observed: list, set, or None
            A list or set of observed nodes. If None, no nodes are observed.

        Returns
        -------
        bool
            True if there is a d-connecting path, False otherwise.

        Examples
        --------
        >>> tspdag = TimeSeriesPDAG(
        ...     directed_ebunch=[(('A', 0), ('B', 1)), (('C', 0), ('B', 1))],
        ...     undirected_ebunch=[(('A', 0), ('C', 0))]
        ... )
        >>> tspdag.is_dconnected(('A', 0), ('C', 0))
        True
        >>> tspdag.is_dconnected(('A', 0), ('B', 1), observed=[('C', 0)])
        True
        """
        # Convert to DAG to use the d-separation algorithm
        dag = self.to_dag()
        return dag.is_dconnected(start, end, observed=observed)

    def get_independencies(self):
        """
        Returns the independencies implied by the d-separation in the TimeSeriesPDAG.

        Returns
        -------
        Independencies
            An Independencies object containing the conditional independencies
            implied by the graph structure.

        Examples
        --------
        >>> tspdag = TimeSeriesPDAG(
        ...     directed_ebunch=[(('A', 0), ('B', 1)), (('C', 0), ('B', 1))],
        ...     undirected_ebunch=[]
        ... )
        >>> independencies = tspdag.get_independencies()
        """
        # Convert to DAG to use the get_independencies method
        dag = self.to_dag()
        return dag.get_independencies()

    def copy(self):
        """
        Returns a copy of the TimeSeriesPDAG.

        Returns
        -------
        TimeSeriesPDAG
            A new TimeSeriesPDAG with the same nodes and edges.

        Examples
        --------
        >>> tspdag = TimeSeriesPDAG(
        ...     directed_ebunch=[(('A', 0), ('B', 1))],
        ...     undirected_ebunch=[(('A', 0), ('C', 0))]
        ... )
        >>> tspdag_copy = tspdag.copy()
        """
        return TimeSeriesPDAG(
            directed_ebunch=list(self.directed_edges),
            undirected_ebunch=list(self.undirected_edges),
            latents=list(self.latents),
        )
