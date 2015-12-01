import itertools
from collections import defaultdict
import logging

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base import DirectedGraph
from pgmpy.factors import TabularCPD
from pgmpy.independencies import Independencies
from pgmpy.extern.six.moves import range
from pgmpy.models import BayesianModel

class NaiveBayesModel(BayesianModel):
    """
    Class to represent Naive Bayes.
    Subclass of Bayesian Model.
    Model holds directed edges from one parent node to multiple
    children nodes only.

    Parameters
    ----------
    data : input graph
        Data to initialize graph.  If data=None (default) an empty
        graph is created.  The data can be an edge list, or any
        NetworkX graph object.

    Examples
    --------
    Create an empty Naive Bayes Model with no nodes and no edges.

    >>> from pgmpy.models import NaiveBayesModel
    >>> G = NaiveBayesModel()

    G can be grown in several ways.

    **Nodes:**

    Add one node at a time:

    >>> G.add_node('a')

    Add the nodes from any container (a list, set or tuple or the nodes
    from another graph).

    >>> G.add_nodes_from(['a', 'b', 'c'])

    **Edges:**

    G can also be grown by adding edges.

    Add one edge,

    >>> G.add_edge('a', 'b')

    a list of edges,

    >>> G.add_edges_from([('a', 'b'), ('a', 'c')])

    If some edges connect nodes not yet in the model, the nodes
    are added automatically.  There are no errors when adding
    nodes or edges that already exist.

    **Shortcuts:**

    Many common graph features allow python syntax for speed reporting.

    >>> 'a' in G     # check if node in graph
    True
    >>> len(G)  # number of nodes in graph
    3
    """

    def __init__(self, ebunch=None):
        super(NaiveBayesModel, self).__init__(ebunch)
        self.parent_node = None
        self.children_nodes = []
        if ebunch:
            for parent,child in self.edges():
                if self.parent_node and parent != self.parent_node:
                    raise ValueError("Model can have only one parent node.")
                self.parent_node = parent

            self.children_nodes = list(set(self.nodes()) - set(self.parent_node))

    
    def add_edge(self, u, v, *kwargs):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph

        Parameters
        ----------
        u,v : nodes
              Nodes can be any hashable python object.

        EXAMPLE
        -------
        >>> from pgmpy.models import NaiveBayesModel
        >>> G = NaiveBayesModel()
        >>> G.add_nodes_from(['a', 'b', 'c'])
        >>> G.add_edge('a', 'b')
        >>> G.add_edge('a', 'c')
        """

        if self.parent_node and u != self.parent_node:
            raise ValueError("Model can have only one parent node.")
        self.parent_node = u
        self.children_nodes.append(v)
        super(NaiveBayesModel, self).add_edge(u, v, *kwargs)
