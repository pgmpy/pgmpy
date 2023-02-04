from pgmpy.independencies import Independencies
from pgmpy.models import BayesianNetwork


class NaiveBayes(BayesianNetwork):
    """
    Class to represent Naive Bayes. Naive Bayes is a special case of Bayesian Model
    where the only edges in the model are from the feature variables to the dependent variable.
    """

    def __init__(self, feature_vars=None, dependent_var=None):
        """
        Method to initialize the `NaiveBayes` class.

        Parameters
        ----------
        feature_vars: list (array-like)
            A list of variable predictor variables (i.e. the features) in the model.

        dependent_var: hashable object
            The dependent variable (i.e. the variable to be predicted) in the model.

        Returns
        -------
        pgmpy.models.BayesianNetwork instance: An instance of a Bayesian Model with the
            initialized model structure.
        """
        self.dependent = dependent_var
        self.features = set(feature_vars) if feature_vars is not None else set()
        if (feature_vars is not None) and (dependent_var is not None):
            ebunch = [(self.dependent, feature) for feature in self.features]
        else:
            ebunch = []

        super(NaiveBayes, self).__init__(ebunch=ebunch)

    def add_edge(self, u, v, *kwargs):
        """
        Add an edge between `u` and `v`.

        The nodes `u` and `v` will be automatically added if they are
        not already in the graph. `u` will be the dependent variable (i.e. variable to be predicted)
        and `v` will be one of the features (i.e. predictors) in the model.

        Parameters
        ----------
        u, v : nodes
               Nodes can be any hashable python object.

        Returns
        -------
        None

        Examples
        --------
        >>> from pgmpy.models import NaiveBayes
        >>> G = NaiveBayes()
        >>> G.add_nodes_from(['a', 'b', 'c'])
        >>> G.add_edge('a', 'b')
        >>> G.add_edge('a', 'c')
        >>> G.edges()
        OutEdgeView([('a', 'b'), ('a', 'c')])
        """
        if self.dependent and u != self.dependent:
            raise ValueError(
                f"Model can only have edges outgoing from: {self.dependent}"
            )
        self.dependent = u
        self.features.add(v)
        super(NaiveBayes, self).add_edge(u, v, *kwargs)

    def add_edges_from(self, ebunch):
        """
        Adds edges to the model.

        Each tuple of the form (u, v) in ebunch adds a new edge in the model.
        Since there can only be one dependent variable in a Naive Bayes model, `u` should
        be the same for each tuple in `ebunch`.

        Parameters
        ----------
        ebunch: list (array-like)
            A list of tuples of the form (u, v) representing an edge from u to v.

        Returns
        -------
        None

        Examples
        --------
        >>> from pgmpy.models import NaiveBayes
        >>> G = NaiveBayes()
        >>> G.add_nodes_from(['a', 'b', 'c'])
        >>> G.add_edges_from([('a', 'b'), ('a', 'c')])
        >>> G.edges()
        OutEdgeView([('a', 'b'), ('a', 'c')])
        """
        for u, v in ebunch:
            self.add_edge(u, v)

    def _get_ancestors_of(self, obs_nodes_list):
        """
        Returns a list of all ancestors of all the observed nodes.

        Parameters
        ----------
        obs_nodes_list: string, list-type
            name of all the observed nodes
        """
        if not obs_nodes_list:
            return set()
        return set(obs_nodes_list) | set(self.dependent)

    def active_trail_nodes(self, start, observed=None):
        """
        Returns all the nodes reachable from start via an active trail.

        Parameters
        ----------
        start: Graph node

        observed : List of nodes (optional)
            If given the active trail would be computed assuming these nodes to be observed.

        Examples
        --------
        >>> from pgmpy.models import NaiveBayes
        >>> model = NaiveBayes()
        >>> model.add_nodes_from(['a', 'b', 'c', 'd'])
        >>> model.add_edges_from([('a', 'b'), ('a', 'c'), ('a', 'd')])
        >>> model.active_trail_nodes('a')
        {'a', 'd', 'c', 'b'}
        >>> model.active_trail_nodes('a', ['b', 'c'])
        {'a', 'd'}
        >>> model.active_trail_nodes('b', ['a'])
        {'b'}
        """

        if observed and self.dependent in observed:
            return set(start)
        else:
            return set(self.nodes()) - set(observed if observed else [])

    def local_independencies(self, variables):
        """
        Returns an instance of Independencies containing the local independencies
        of each of the variables.


        Parameters
        ----------
        variables: str or array like
            variables whose local independencies are to found.

        Examples
        --------
        >>> from pgmpy.models import NaiveBayes
        >>> model = NaiveBayes()
        >>> model.add_edges_from([('a', 'b'), ('a', 'c'), ('a', 'd')])
        >>> ind = model.local_independencies('b')
        >>> ind
        (b \u27C2 d, c | a)
        """
        independencies = Independencies()
        for variable in [variables] if isinstance(variables, str) else variables:
            if variable != self.dependent:
                independencies.add_assertions(
                    [variable, list(set(self.features) - set(variable)), self.dependent]
                )
        return independencies

    def fit(self, data, parent_node=None, estimator=None):
        """
        Computes the CPD for each node from a given data in the form of a pandas dataframe.
        If a variable from the data is not present in the model, it adds that node into the model.

        Parameters
        ----------
        data : pandas DataFrame object
            A DataFrame object with column names same as the variable names of network

        parent_node: any hashable python object (optional)
            Parent node of the model, if not specified it looks for a previously specified
            parent node.

        estimator: Estimator class
            Any pgmpy estimator. If nothing is specified, the default ``MaximumLikelihoodEstimator``
            would be used.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import NaiveBayes
        >>> model = NaiveBayes()
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model.fit(values, 'A')
        >>> model.get_cpds()
        [<TabularCPD representing P(D:2 | A:2) at 0x4b72870>,
         <TabularCPD representing P(E:2 | A:2) at 0x4bb2150>,
         <TabularCPD representing P(A:2) at 0x4bb23d0>,
         <TabularCPD representing P(B:2 | A:2) at 0x4bb24b0>,
         <TabularCPD representing P(C:2 | A:2) at 0x4bb2750>]
        >>> model.edges()
        [('A', 'D'), ('A', 'E'), ('A', 'B'), ('A', 'C')]
        """
        if not parent_node:
            if not self.dependent:
                raise ValueError("parent node must be specified for the model")
            else:
                parent_node = self.dependent
        if parent_node not in data.columns:
            raise ValueError(
                f"Dependent variable: {parent_node} is not present in the data"
            )
        for child_node in data.columns:
            if child_node != parent_node:
                self.add_edge(parent_node, child_node)
        super(NaiveBayes, self).fit(data, estimator)
