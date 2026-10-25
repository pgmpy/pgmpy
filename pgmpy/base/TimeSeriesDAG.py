#!/usr/bin/env python

import networkx as nx

from pgmpy.base.DAG import DAG


class TimeSeriesDAG(DAG):
    """
    Class for representing directed acyclic graphs for time series data.

    This class extends the DAG class to include time series data with lagged relationships.
    Each node is represented as a tuple (variable, lag), where lag is an integer
    indicating the lag relative to the current timepoint.

    Parameters
    ----------
    ebunch: list, array-like
        A list of edges to be added to the graph. Each edge is represented as a tuple
        (source, target), where source and target are tuples representing the variable
        and its lag.

    Examples
    --------
    >>> from pgmpy.base.TimeSeriesDAG import TimeSeriesDAG
    >>> edges = [(('A', 0), ('B', 1)), (('B', 1), ('C', 2))]
    >>> dag = TimeSeriesDAG(edges)
    """

    def __init__(
        self,
        ebunch=None,
        num_time_slices=1,
        latents=None,
    ):
        self.num_time_slices = num_time_slices
        if latents is None:
            latents = set()
        self.latents = set(latents)
        super().__init__(
            ebunch=ebunch,
            latents=self.latents,
        )

    def is_dconnected(self, start, end, observed=None, include_latents=False):
        """
        Returns True if there is an active trail (i.e. d-connection) between
        `start` and `end` node given that `observed` is observed.

        Parameters
        ----------
        start, end : tuple or str
            The nodes in the TimeSeriesDAG between which to check the d-connection/active trail.
            Can be tuples (variable, lag) or strings (variable names).
            If strings are provided, they will be converted to tuples with lag 0.

        observed : list, array-like (optional)
            If given the active trail would be computed assuming these nodes to
            be observed. Can contain tuples (variable, lag) or strings (variable names).

        include_latents: boolean (default: False)
            If true, latent variables are returned as part of the active trail.

        Examples
        --------
        >>> from pgmpy.base.TimeSeriesDAG import TimeSeriesDAG
        >>> edges = [(('A', 0), ('B', 1)), (('B', 1), ('C', 2))]
        >>> dag = TimeSeriesDAG(edges)
        >>> dag.is_dconnected(('A', 0), ('C', 2))
        True
        >>> dag.is_dconnected('A', 'C')  # Equivalent to ('A', 0) and ('C', 0)
        False
        """

        # Use the active_trail_nodes method to find reachable nodes
        active_trails = self.active_trail_nodes(
            variables=start, observed=observed, include_latents=include_latents
        )

        # Check if end_node is reachable from start_node
        return end in active_trails[start]

    def active_trail_nodes(self, variables, observed=None, include_latents=False):
        """
        Returns a dictionary with the given variables as keys and all the nodes reachable
        from that respective variable as values.

        Parameters
        ----------
        variables: tuple, str, or array-like
            Variables whose active trails are to be found.
            Can be tuples (variable, lag) or strings (variable names).

        observed : List of nodes (optional)
            If given the active trails would be computed assuming these nodes to be
            observed. Can contain tuples (variable, lag) or strings (variable names).

        include_latents: boolean (default: False)
            Whether to include the latent variables in the returned active trail nodes.

        Examples
        --------
        >>> from pgmpy.base.TimeSeriesDAG import TimeSeriesDAG
        >>> edges = [(('A', 0), ('B', 1)), (('B', 1), ('C', 2))]
        >>> dag = TimeSeriesDAG(edges)
        >>> dag.active_trail_nodes(('A', 0))
        {('A', 0): {('A', 0), ('B', 1), ('C', 2)}}
        >>> dag.active_trail_nodes('A')  # Equivalent to ('A', 0)
        {('A', 0): {('A', 0), ('B', 1), ('C', 2)}}
        """

        # Get ancestors of observed nodes
        ancestors_list = self._get_ancestors_of(observed)

        # Direction of flow of information
        # up ->  from parent to child
        # down -> from child to parent

        active_trails = {}
        for start in variables:
            visit_list = set()
            visit_list.add((start, "up"))
            traversed_list = set()
            active_nodes = set()

            while visit_list:
                node, direction = visit_list.pop()

                if (node, direction) not in traversed_list:
                    if node not in observed:
                        active_nodes.add(node)
                    traversed_list.add((node, direction))

                    if direction == "up" and node not in observed:
                        for parent in self.predecessors(node):
                            visit_list.add((parent, "up"))
                        for child in self.successors(node):
                            visit_list.add((child, "down"))

                    elif direction == "down":
                        if node not in observed:
                            for child in self.successors(node):
                                visit_list.add((child, "down"))
                        if node in ancestors_list:
                            for parent in self.predecessors(node):
                                visit_list.add((parent, "up"))

            if include_latents:
                active_trails[start] = active_nodes
            else:
                active_trails[start] = active_nodes - self.latents

        return active_trails

    def get_ancestral_graph(self, nodes):
        """
        Returns the ancestral graph of the given `nodes` in the time series DAG.

        The ancestral graph only contains the nodes which are ancestors of at least
        one of the variables in nodes.

        Parameters
        ----------
        nodes: iterable
            List of nodes whose ancestral graph needs to be computed.
            Each node should be a tuple (variable, lag).

        Returns
        -------
        TimeSeriesDAG
            A new TimeSeriesDAG containing only the ancestors of the specified nodes.

        Examples
        --------
        >>> from pgmpy.base.TimeSeriesDAG import TimeSeriesDAG
        >>> edges = [(('A', 0), ('B', 1)), (('B', 1), ('C', 2))]
        >>> dag = TimeSeriesDAG(edges)
        >>> ancestral_graph = dag.get_ancestral_graph([('C', 2)])
        >>> ancestral_graph.nodes()
        {('A', 0), ('B', 1), ('C', 2)}
        """
        # Find all ancestors
        ancestors = set()
        for node in nodes:
            # For each node, trace back its ancestors
            visited = set()
            stack = [node]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue

                visited.add(current)
                ancestors.add(current)

                # Add predecessors
                stack.extend(self.predecessors(current))

        # Create a new TimeSeriesDAG with the ancestors
        ancestral_edges = []
        for ancestor in ancestors:
            for successor in self.successors(ancestor):
                ancestral_edges.append((ancestor, successor))
        return TimeSeriesDAG(
            ebunch=ancestral_edges,
            num_time_slices=self.num_time_slices,
            latents=self.latents,
        )

    def plot_summary_graph(self, **kwargs):
        """
        Generates and plots the summary graph of the time series DAG.
        The summary graph represents each variable only once, and edges
        indicate causal influence at any lag.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to be passed to the networkx drawing function.

        Returns
        -------
        summary_graph : nx.DiGraph
            The summary graph object.
        fig, ax : matplotlib figure and axis
            The matplotlib figure and axis of the plot.
        """

        import matplotlib.pyplot as plt
        import networkx as nx

        # Create the summary graph
        summary_graph = nx.DiGraph()
        variables = set(var for var, _ in self.nodes())
        summary_graph.add_nodes_from(variables)

        for (var1, lag1), (var2, lag2) in self.edges():
            if not summary_graph.has_edge(var1, var2):
                summary_graph.add_edge(var1, var2, lags=set())
            summary_graph[var1][var2]["lags"].add((lag1, lag2))

        # Generate layout
        pos = {node: (i, 0) for i, node in enumerate(summary_graph.nodes())}

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        nx.draw(summary_graph, pos, with_labels=True, ax=ax, **kwargs)

        # Add edge labels for lags
        edge_labels = {
            (u, v): f"lags: {sorted(list(data['lags']))}"
            for u, v, data in summary_graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(summary_graph, pos, edge_labels=edge_labels, ax=ax)

        ax.set_title("Summary Graph (Causal Links at Any Lag)")
        return fig, ax
