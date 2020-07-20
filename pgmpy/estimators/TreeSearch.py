#!/usr/bin/env python

from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import mutual_info_score

from pgmpy.estimators import StructureEstimator
from pgmpy.base import DAG


class TreeSearch(StructureEstimator):
    def __init__(self, data, root_node, return_type='chow-liu', class_node=None, n_jobs=-1, **kwargs):
        """
        Search class for learning tree related graph structure.  The algorithms supported are Chow-Liu and 
        Tree-augmented naive bayes (TAN).

        Chow-Liu constructs the maximum-weight spanning tree with mutual information score as edge weights.

        TAN is an extension of Naive Bayes classifier to allow a tree structure over the independent variables to
        account for interaction.

        Parameters
        ----------
        data: pandas DataFrame object
            dataframe object where each column represents one variable.

        root_node: string, int or any hashable python object.
            The root node of the tree structure.

        return_type: string (chow-liu | tan)
            Return type for graph structure, either of 'chow-liu' or 'tan'.
            Defaults to 'chow-liu'

        class_node: string, int or any hashable python object. (optional)
            Specifies the class node when return_type is 'tan'.

        n_jobs: int (default: -1)
            Number of jobs to run in parallel. `-1` means use all processors.

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

        Friedman N, Geiger D and Goldszmidt M (1997). Bayesian network classifiers. Machine Learning 29: 131–163

        """
        self.data = data
        self.root_node = root_node
        self.return_type = return_type
        self.class_node = class_node
        self.n_jobs = n_jobs

        super(TreeSearch, self).__init__(data, **kwargs)

    def estimate(self):
        """
        Estimate the `DAG` structure that fits best to the given data set without parametrization.

        Step 1: Argument checks
        Step 2: Create a fully connected graph over the independent variables with edge weights as the mutual info
                score between them
        Step 3: Construct the maximum spanning tree
        Step 4: Create DAG from tree by directing edges away from given root node
        Step 5: If return_type is 'tan', add edges from the class variable to independent variables

        Returns
        -------
        model: `DAG` instance
            A tree or TAN `DAG`.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import networkx as nx
        >>> import matplotlib.pyplot as plt
        >>> from pgmpy.estimators import TreeSearch
        >>>
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)), columns=['A', 'B', 'C', 'D', 'E'])
        >>>
        >>> est = TreeSearch(values, root_node='B', return_type='chow-liu')
        >>> model = est.estimate()
        >>> nx.draw_circular(model, with_labels=True, arrowsize=20, arrowstyle='fancy', alpha=0.3)
        >>> plt.show()
        >>>
        >>> est = TreeSearch(values, root_node='B', return_type='tan', class_node='A')
        >>> model = est.estimate()
        >>> nx.draw_circular(model, with_labels=True, arrowsize=20, arrowstyle='fancy', alpha=0.3)
        >>> plt.show()

        """
        if self.root_node not in self.data.columns:
            raise ValueError("Root node %s not found in data columns" % self.root_node)

        if self.return_type not in ['chow-liu', 'tan']:
            raise ValueError("Invalid return_type %s" % self.return_type)

        data = self.data
        n_jobs = self.n_jobs
        root_node = self.root_node
        return_type = self.return_type
        class_node = self.class_node

        if return_type == 'chow-liu':
            # construct tree with chow-liu algorithm
            return DAG(self.chow_liu(data, root_node, n_jobs))

        else:
            if class_node not in data.columns:
                raise ValueError("Class node %s not found in data columns" % class_node)

            if root_node == class_node:
                raise ValueError("Root node %s and class node %s are identical" % (root_node, class_node))

            # construct tree with chow-liu algorithm
            df_features = data.loc[:, data.columns != class_node]
            D = self.chow_liu(df_features, root_node, n_jobs)

            # add edge from class node to other nodes
            for node in df_features.columns:
                D.add_edge(class_node, node)

            return DAG(D)

    def chow_liu(self, data, root_node, n_jobs):
        # create a fully connected graph over the independent variables
        def calculate_mi(u_index, v_index):
            return (u_index, v_index, mutual_info_score(data.iloc[:, u_index], data.iloc[:, v_index]))

        total_cols = len(data.columns)
        pbar = tqdm(combinations(enumerate(data.columns), 2), total=(total_cols*(total_cols-1)/2))
        pbar.set_description("Building tree")

        vals = Parallel(n_jobs=n_jobs)(
            delayed(calculate_mi)(
                u_index = u_index,
                v_index = v_index
            )
            for (u_index, u), (v_index, v) in pbar
        )

        weights = pd.DataFrame(0.0, index=data.columns, columns=data.columns)
        for u_index, v_index, mi in vals:
            weights.iat[u_index, v_index] = mi 

        # construct maximum spanning tree
        T = nx.maximum_spanning_tree(nx.from_pandas_adjacency(weights, create_using=nx.Graph))

        # create DAG by directing edges away from root node
        D = nx.bfs_tree(T, root_node)

        return DAG(D)
