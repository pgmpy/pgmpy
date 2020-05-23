from pgmpy.models import BayesianModel
from sklearn.metrics import mutual_info_score
import networkx as nx



class TreeAugmentedNaiveBayes(BayesianModel):
    """
    Class to represent Tree Augmented Naive Bayes (TAN). TAN is an extension of Naive Bayes classifer
    and allows a tree structure over the independent variables to account for interaction.
    """

    def __init__(self):
        """
        Instantiate the `TreeAugmentedNaiveBayes` class with an uninitialized model structure.
        """
        super(TreeAugmentedNaiveBayes, self).__init__()

    def learn_structure(self, data, class_node, root_node=None):
        """
        Learn the model structure using the chow-liu algorithm on a given data in the form of a pandas dataframe.

        Parameters
        ----------
        data : pandas DataFrame object
            A DataFrame object with column names same as the variable names of network

        class_node: node
            Dependent variable of the model.

        root_node: node (optional)
            The root node of the tree structure over the independent variables.  If not specified, then
            an arbitrary independent variable is selected as the root.

        Returns
        -------
        pgmpy.models.BayesianModel instance: An instance of a Bayesian Model with learned model structure.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import networkx as nx
        >>> import matplotlib.pyplot as plt
        >>> from pgmpy.models import TreeAugmentedNaiveBayes
        >>> model = TreeAugmentedNaiveBayes()
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> class_node = 'A'
        >>> D = model.learn_structure(values, class_node)
        >>> nx.draw_circular(D, with_labels=True, arrowsize=20, arrowstyle='fancy', alpha=0.3)
        >>> plt.show()
        >>> model.fit(data)
        >>> model.predict(data.loc[:10, data.columns != class_node])
        """
        if not class_node:
            raise ValueError("class node must be specified for the model")

        if class_node not in data.columns:
            raise ValueError("class node must exist in data")

        if root_node is not None and root_node not in data.columns:
            raise ValueError("root node must exist in data")

        # construct maximum spanning tree
        G = nx.Graph()
        df_features = data.loc[:, data.columns != class_node]
        total_cols = len(df_features.columns)
        for i in range(total_cols):
            for j in range(i+1, total_cols):
                # edge weight is the MI between a pair of independent variables
                mi = mutual_info_score(df_features.iloc[:,i], df_features.iloc[:,j])
                G.add_edge(df_features.columns[i], df_features.columns[j], weight=mi)
        T = nx.maximum_spanning_tree(G)

        # create DAG by directing edges away from root node
        if root_node:
            D = nx.bfs_tree(T, root_node)
        else:
            D = nx.bfs_tree(T, df_features.columns[0])

        # add edge from class node to other nodes
        for node in df_features.columns:
            D.add_edge(class_node, node)

		# initialize BayesianModel
        for parent_node, child_node in D.edges():
            super(TreeAugmentedNaiveBayes, self).add_edge(parent_node, child_node)

        return self

