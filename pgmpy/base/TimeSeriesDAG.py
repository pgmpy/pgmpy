#!/usr/bin/env python

from itertools import chain, combinations, permutations, product
import numpy as np
import pandas  as pd
import networkx as nx
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import PDAG, DAG
from pgmpy.global_vars import logger

class TimeSeriesDAG(DAG):
    """
    Class for representing directed acyclic graphs for time series data.
    
    This class extends the DAG class to include time series data with lagged relationships.
    Each node is represented as a tuple (variable, lag), where lag is an integer
    indicating the lag relative to the current timepoint.

    Parameters
    ----------
    edges: list, array-like
        A list of edges to be added to the graph. Each edge is represented as a tuple
        (source, target), where source and target are tuples representing the variable
        and its lag.

    Examples
    --------
    >>> from pgmpy.estimators import TimeSeriesDAG
    >>> edges = [(('A', 0), ('B', 1)), (('B', 1), ('C', 2))]
    >>> dag = TimeSeriesDAG(edges)
    """

    def __init__(self, edges=None):
        super(TimeSeriesDAG, self).__init__(edges=edges)

    def add_edge(self, u, v, **kwargs):
        """
        Adds an edge between the nodes u and v.
        
        These nodes will be automatically added if they are
        not already present in the graph.

        Parameters
        ----------
        u, v : nodes
            Nodes can be anu hashable python object.
            Each node should be a tuple (variable, lag).
        """

        # check if u anv are valid nodes
        if not (isinstance(u, tuple) and len(u) == 2 and isinstance(u[1], int)):
            raise ValueError("Node u should be a tuple (variable, lag).")
        if not (isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], int)):
            raise ValueError("Node v should be a tuple (variable, lag).")
        
        # validate the temporal relationship
        if(u[1] > v[1]):
            raise ValueError("The lag of the source node should be less than or equal to the target node.")
        
        super(TimeSeriesDAG, self).add_edge(u, v, **kwargs)

    def to_summary_graph(self):
        """
        Converts the time series DAG to a summary graph where each variable
        appears only once, and the edges represent the existance of a causal
        link at any lag.
        
        Returns
        -------
        summary_graph : nx.DiGraph
            A directed graph where each node is a variable and edges represent
            the existence of a causal link at any lag.
        """

        summary_graph = nx.DiGraph()
        
        # Add all the variables as nodes
        variables = set(var for var, _ in self.nodes())
        summary_graph.add_nodes_from(variables)

        # add nodes for each causal link
        for (var1 , lag1) , (var2, lag2) in self.edges():
            if not summary_graph.has_edge(var1, var2):
                summary_graph.add_edge(var1, var2, lags = set())
            summary_graph[var1][var2]['lags'].add((lag1, lag2))
        
        return summary_graph

    def plot(self, **kwargs):
        """
        Plots the time series DAG using networkx.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to be passed to the networkx drawing function.
        
        Returns
        -------
        fig, ax : matplotlib figure and axis
            The matplotlib figure and axis
        """

        import matplotlib.pyplot as plt
        import networkx as nx

        # create a layout for the nodes
        pos = {}
        for node in self.nodes():
            var, lag = node
            # we use a hash value for y to separate
            # their positions
            pos[node] = (lag, hash(var) % 100)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        nx.draw(self, pos, with_labels=True, ax=ax, **kwargs)
        

        # label the edges with the lags
        lags = sorted(set(lag for _, lag in self.edges()))
        ax.set_xticks(lags)
        ax.set_xticklabels([f't{l}' if l == 0 else f't{l}' for l in lags])
        ax.set_title("Time Series Causal Graph")

        return fig, ax

        
        
