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
    >>> from pgmpy.estimators import TimeSeriesDAG
    >>> edges = [(('A', 0), ('B', 1)), (('B', 1), ('C', 2))]
    >>> dag = TimeSeriesDAG()
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
