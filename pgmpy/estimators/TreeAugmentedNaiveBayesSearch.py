#!/usr/bin/env python

import networkx as nx

from pgmpy.estimators import StructureEstimator
from sklearn.metrics import mutual_info_score
from pgmpy.base import DAG


class TreeAugmentedNaiveBayesSearch(StructureEstimator):
    def __init__(self, data, class_node, root_node=None, **kwargs):
        """
        Search class for learning tree-augmented naive bayes (TAN) graph structure with a given set of variables.

        TAN is an extension of Naive Bayes classifer and allows a tree structure over the independent variables
        to account for interaction.

        Parameters
        ----------
        data: pandas DataFrame object
            dataframe object where each column represents one variable.

        class_node: node
            Dependent variable of the model (i.e. the class label to predict)

        root_node: node (optional)
            The root node of the tree structure over the independent variables.  If not specified, then
            an arbitrary independent variable is selected as the root.

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
        Friedman N, Geiger D and Goldszmidt M (1997). Bayesian network classifiers. Machine Learning 29: 131–163
        """
        self.class_node = class_node
        self.root_node = root_node

        super(TreeAugmentedNaiveBayesSearch, self).__init__(data, **kwargs)

    def estimate(self):
        """
        Estimate the `DAG` structure that fits best to the given data set without parametrization.

        Returns
        -------
        model: `DAG` instance
            A tree augmented naive bayes `DAG`.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import networkx as nx
        >>> import matplotlib.pyplot as plt
        >>> from pgmpy.estimators import ExhaustiveSearch
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)), columns=['A', 'B', 'C', 'D', 'E'])
        >>> class_node = 'A'
        >>> est = TreeAugmentedNaiveBayesSearch()
        >>> model = est.estimate()
        >>> nx.draw_circular(model, with_labels=True, arrowsize=20, arrowstyle='fancy', alpha=0.3)
        >>> plt.show()
        """
        if self.class_node not in self.data.columns:
            raise ValueError("class node must exist in data")

        if self.root_node is not None and self.root_node not in self.data.columns:
            raise ValueError("root node must exist in data")

        # construct maximum spanning tree
        G = nx.Graph()
        df_features = self.data.loc[:, self.data.columns != self.class_node]
        total_cols = len(df_features.columns)
        for i in range(total_cols):
            from_node = df_features.columns[i]
            G.add_node(from_node)
            for j in range(i+1, total_cols):
                to_node = df_features.columns[j]
                G.add_node(to_node)
                # edge weight is the MI between a pair of independent variables
                mi = mutual_info_score(df_features.iloc[:,i], df_features.iloc[:,j])
                G.add_edge(from_node, to_node, weight=mi)
        T = nx.maximum_spanning_tree(G)

        # create DAG by directing edges away from root node
        if self.root_node:
            D = nx.bfs_tree(T, self.root_node)
        else:
            D = nx.bfs_tree(T, df_features.columns[0])

        # add edge from class node to other nodes
        for node in df_features.columns:
            D.add_edge(self.class_node, node)

        return DAG(D)

