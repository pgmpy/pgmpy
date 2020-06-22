#!/usr/bin/env python

import networkx as nx

from pgmpy.estimators import StructureEstimator
from sklearn.metrics import mutual_info_score
from pgmpy.base import DAG


class TreeSearch(StructureEstimator):

    def __init__(self, data, root_node, **kwargs):
        """
        Search class for learning tree graph structure from a given set of variables.

        Parameters
        ----------
        data: pandas DataFrame object
            dataframe object where each column represents one variable.

        root_node: node
            Root node of the learned tree structure.

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
            This sets the behavior of the `state_count`-method.

        References
        ----------
        Chow, C. K.; Liu, C.N. (1968), "Approximating discrete probability distributions with dependence trees",
        IEEE Transactions on Information Theory, IT-14 (3): 462–467

        """
        self.root_node = root_node

        super(TreeSearch, self).__init__(data, **kwargs)

    def estimate(self):
        """
        Estimate the tree `DAG` structure using Chow-Liu algorithm that fits best to the given data set without
        parametrization.

        Returns
        -------
        model: `DAG` instance
            A tree `DAG`.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import networkx as nx
        >>> import matplotlib.pyplot as plt
        >>> from pgmpy.estimators import TreeSearch
        >>> data = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)), columns=['A', 'B', 'C', 'D', 'E'])
        >>> est = TreeSearch(data, root_node='A')
        >>> model = est.estimate()
        >>> nx.draw_circular(model, with_labels=True, arrowsize=20, arrowstyle='fancy', alpha=0.3)
        >>> plt.show()

        """
        if self.root_node not in self.data.columns:
            raise ValueError("Root node must exist in data")

        # construct maximum spanning tree
        G = nx.Graph()
        total_cols = len(self.data.columns)
        for i in range(total_cols):
            from_node = self.data.columns[i]
            G.add_node(from_node)
            for j in range(i + 1, total_cols):
                to_node = self.data.columns[j]
                G.add_node(to_node)
                # edge weight is the MI between a pair of independent variables
                mi = mutual_info_score(self.data.iloc[:, i], self.data.iloc[:, j])
                G.add_edge(from_node, to_node, weight=mi)
        T = nx.maximum_spanning_tree(G)

        # create DAG by directing edges away from root node
        return DAG(nx.bfs_tree(T, self.root_node))
