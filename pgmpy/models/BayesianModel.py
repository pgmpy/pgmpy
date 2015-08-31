#!/usr/bin/env python3

import itertools
from collections import defaultdict
import logging

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base import DirectedGraph
from pgmpy.factors import TabularCPD
from pgmpy.independencies import Independencies


class BayesianModel(DirectedGraph):
    """
    Base class for bayesian model.

    A models stores nodes and edges with conditional probability
    distribution (cpd) and other attributes.

    models hold directed edges.  Self loops are not allowed neither
    multiple (parallel) edges.

    Nodes should be strings.

    Edges are represented as links between nodes.

    Parameters
    ----------
    data : input graph
        Data to initialize graph.  If data=None (default) an empty
        graph is created.  The data can be an edge list, or any
        NetworkX graph object.

    Examples
    --------
    Create an empty bayesian model with no nodes and no edges.

    >>> from pgmpy.models import BayesianModel
    >>> G = BayesianModel()

    G can be grown in several ways.

    **Nodes:**

    Add one node at a time:

    >>> G.add_node('a')

    Add the nodes from any container (a list, set or tuple or the nodes
    from another graph).

    >>> G.add_nodes_from(['a', 'b'])

    **Edges:**

    G can also be grown by adding edges.

    Add one edge,

    >>> G.add_edge('a', 'b')

    a list of edges,

    >>> G.add_edges_from([('a', 'b'), ('b', 'c')])

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
        super().__init__()
        if ebunch:
            self.add_edges_from(ebunch)
        self.cpds = []
        self.cardinalities = defaultdict(int)

    def add_edge(self, u, v, **kwargs):
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
        >>> from pgmpy.models import BayesianModel/home/abinash/software_packages/numpy-1.7.1
        >>> G = BayesianModel()
        >>> G.add_nodes_from(['grade', 'intel'])
        >>> G.add_edge('grade', 'intel')
        """
        if u == v:
            raise ValueError('Self loops are not allowed.')
        if u in self.nodes() and v in self.nodes() and nx.has_path(self, v, u):
            raise ValueError(
                 'Loops are not allowed. Adding the edge from (%s->%s) forms a loop.' % (u, v))
        else:
            super(BayesianModel, self).add_edge(u, v, **kwargs)
        
    def add_cpds(self, *cpds):
        """
        Add CPD (Conditional Probability Distribution) to the Bayesian Model.

        Parameters
        ----------
        cpds  :  list, set, tuple (array-like)
            List of CPDs which will be associated with the model

        EXAMPLE
        -------
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors.CPD import TabularCPD
        >>> student = BayesianModel([('diff', 'grades'), ('intel', 'grades')])
        >>> grades_cpd = TabularCPD('grades', 3, [[0.1,0.1,0.1,0.1,0.1,0.1],
        ...                                       [0.1,0.1,0.1,0.1,0.1,0.1],
        ...                                       [0.8,0.8,0.8,0.8,0.8,0.8]],
        ...                         evidence=['diff', 'intel'], evidence_card=[2, 3])
        >>> student.add_cpds(grades_cpd)

        +------+-----------------------+---------------------+
        |diff: |          easy         |         hard        |
        +------+------+------+---------+------+------+-------+
        |intel:| dumb |  avg |  smart  | dumb | avg  | smart |
        +------+------+------+---------+------+------+-------+
        |gradeA| 0.1  | 0.1  |   0.1   |  0.1 |  0.1 |   0.1 |
        +------+------+------+---------+------+------+-------+
        |gradeB| 0.1  | 0.1  |   0.1   |  0.1 |  0.1 |   0.1 |
        +------+------+------+---------+------+------+-------+
        |gradeC| 0.8  | 0.8  |   0.8   |  0.8 |  0.8 |   0.8 |
        +------+------+------+---------+------+------+-------+
        """
        for cpd in cpds:
            if not isinstance(cpd, TabularCPD):
                raise ValueError('Only TabularCPD can be added.')

            if set(cpd.variables) - set(cpd.variables).intersection(
                    set(self.nodes())):
                raise ValueError('CPD defined on variable not in the model', cpd)

            for prev_cpd_index in range(len(self.cpds)):
                if self.cpds[prev_cpd_index].variable == cpd.variable:
                    logging.warning("Replacing existing CPD for {var}".format(var=cpd.variable))
                    self.cpds[prev_cpd_index] = cpd
                    break
            else:
                self.cpds.append(cpd)

    def get_cpds(self, node=None):
        """
        Returns the cpds that have been added till now to the graph

        Parameter
        ---------
        node: any hashable python object (optional)
            The node whose CPD we want. If node not specified returns all the
            CPDs added to the model.

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors import TabularCPD
        >>> student = BayesianModel([('diff', 'grade'), ('intel', 'grade')])
        >>> cpd = TabularCPD('grade', 2, [[0.1, 0.9, 0.2, 0.7],
        ...                               [0.9, 0.1, 0.8, 0.3]],
        ...                  ['intel', 'diff'], [2, 2])
        >>> student.add_cpds(cpd)
        >>> student.get_cpds()
        """
        if node:
            if node not in self.nodes():
                raise ValueError('Node not present in the Directed Graph')
            for cpd in self.cpds:
                if cpd.variable == node:
                    return cpd
        else:
            return self.cpds

    def remove_cpds(self, *cpds):
        """
        Removes the cpds that are provided in the argument.

        Parameters
        ----------
        *cpds: TabularCPD, TreeCPD, RuleCPD object
            A CPD object on any subset of the variables of the model which
            is to be associated with the model.

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors import TabularCPD
        >>> student = BayesianModel([('diff', 'grade'), ('intel', 'grade')])
        >>> cpd = TabularCPD('grade', 2, [[0.1, 0.9, 0.2, 0.7],
        ...                               [0.9, 0.1, 0.8, 0.3]],
        ...                  ['intel', 'diff'], [2, 2])
        >>> student.add_cpds(cpd)
        >>> student.remove_cpds(cpd)
        """
        for cpd in cpds:
            if isinstance(cpd, str):
                cpd = self.get_cpds(cpd)
            self.cpds.remove(cpd)

    def check_model(self):
        """
        Check the model for various errors. This method checks for the following
        errors.

        * Checks if the sum of the probabilities for each state is equal to 1 (tol=0.01).
        * Checks if the CPDs associated with nodes are consistent with their parents.

        Returns
        -------
        check: boolean
            True if all the checks are passed
        """
        for node in self.nodes():
            cpd = self.get_cpds(node=node)
            if isinstance(cpd, TabularCPD):
                evidence = cpd.evidence
                parents = self.get_parents(node)
                if set(evidence if evidence else []) != set(parents if parents else []):
                    raise ValueError("CPD associated with %s doesn't have "
                                     "proper parents associated with it." % node)
                if not np.allclose(cpd.marginalize(node, inplace=False).values,
                                   np.ones(np.product(cpd.evidence_card)),
                                   atol=0.01):
                    raise ValueError('Sum of probabilites of states for node %s'
                                     ' is not equal to 1.' % node)
        return True

    def _get_ancestors_of(self, obs_nodes_list):
        """
        Returns a list of all ancestors of all the observed nodes.

        Parameters
        ----------
        obs_nodes_list: string, list-type
            name of all the observed nodes
        """
        if not isinstance(obs_nodes_list, (list, tuple)):
            obs_nodes_list = [obs_nodes_list]

        ancestors_list = set()
        nodes_list = set(obs_nodes_list)
        while nodes_list:
            node = nodes_list.pop()
            if node not in ancestors_list:
                nodes_list.update(self.predecessors(node))
            ancestors_list.add(node)
        return ancestors_list

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

        >>> from pgmpy.models import BayesianModel
        >>> student = BayesianModel()
        >>> student.add_nodes_from(['diff', 'intel', 'grades'])
        >>> student.add_edges_from([('diff', 'grades'), ('intel', 'grades')])
        >>> student.active_trail_nodes('diff')
        {'diff', 'grade'}
        >>> student.active_trail_nodes('diff', observed='grades')
        {'diff', 'intel'}

        References
        ----------
        Details of the algorithm can be found in 'Probabilistic Graphical Model
        Principles and Techniques' - Koller and Friedman
        Page 75 Algorithm 3.1
        """
        if observed:
            observed_list = [observed] if isinstance(observed, str) else observed
        else:
            observed_list = []
        ancestors_list = self._get_ancestors_of(observed_list)

        # Direction of flow of information
        # up ->  from parent to child
        # down -> from child to parent

        visit_list = set()
        visit_list.add((start, 'up'))
        traversed_list = set()
        active_nodes = set()
        while visit_list:
            node, direction = visit_list.pop()
            if (node, direction) not in traversed_list:
                if node not in observed_list:
                    active_nodes.add(node)
                traversed_list.add((node, direction))
                if direction == 'up' and node not in observed_list:
                    for parent in self.predecessors(node):
                        visit_list.add((parent, 'up'))
                    for child in self.successors(node):
                        visit_list.add((child, 'down'))
                elif direction == 'down':
                    if node not in observed_list:
                        for child in self.successors(node):
                            visit_list.add((child, 'down'))
                    if node in ancestors_list:
                        for parent in self.predecessors(node):
                            visit_list.add((parent, 'up'))
        return active_nodes

    def local_independencies(self, variables):
        """
        Returns a independencies object containing the local independencies
        of each of the variables.

        Parameters
        ----------
        variables: str or array like
            variables whose local independencies are to found.

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> student = BayesianModel()
        >>> student.add_edges_from([('diff', 'grade'), ('intel', 'grade'),
        >>>                         ('grade', 'letter'), ('intel', 'SAT')])
        >>> ind = student.local_independencies('grade')
        >>> ind.event1
        {'grade'}
        >>> ind.event2
        {'SAT'}
        >>> ind.event3
        {'diff', 'intel'}
        """
        def dfs(node):
            """
            Returns the descendents of node.

            Since there can't be any cycles in the Bayesian Network. This is a
            very simple dfs which doen't remember which nodes it has visited.
            """
            descendents = []
            visit = [node]
            while visit:
                n = visit.pop()
                neighbors = self.neighbors(n)
                visit.extend(neighbors)
                descendents.extend(neighbors)
            return descendents

        from pgmpy.independencies import Independencies
        independencies = Independencies()
        for variable in [variables] if isinstance(variables, str) else variables:
            independencies.add_assertions([variable, set(self.nodes()) - set(dfs(variable)) -
                                           set(self.get_parents(variable)) - {variable},
                                           set(self.get_parents(variable))])
        return independencies

    def is_active_trail(self, start, end, observed=None):
        """
        Returns True if there is any active trail between start and end node

        Parameters
        ----------
        start : Graph Node

        end : Graph Node

        observed : List of nodes (optional)
            If given the active trail would be computed assuming these nodes to be observed.

        additional_observed : List of nodes (optional)
            If given the active trail would be computed assuming these nodes to be observed along with
            the nodes marked as observed in the model.

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> student = BayesianModel()
        >>> student.add_nodes_from(['diff', 'intel', 'grades', 'letter', 'sat'])
        >>> student.add_edges_from([('diff', 'grades'), ('intel', 'grades'), ('grade', 'letter'),
        ...                         ('intel', 'sat')])
        >>> student.is_active_trail('diff', 'intel')
        False
        >>> student.is_active_trail('grade', 'sat')
        True
        """
        if end in self.active_trail_nodes(start, observed):
            return True
        else:
            return False

    def get_independencies(self, latex=False):
        """
        Compute independencies in Bayesian Network.

        Parameters
        ----------
        latex: boolean
            If latex=True then latex string of the independence assertion
            would be created.

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> student = BayesianModel()
        >>> student.add_nodes_from(['diff', 'intel', 'grades', 'letter', 'sat'])
        >>> student.add_edges_from([('diff', 'grades'), ('intel', 'grades'), ('grade', 'letter'),
        ...                         ('intel', 'sat')])
        >>> student.get_independencies()
        """
        independencies = Independencies()
        for start in (self.nodes()):
            for r in (1, len(self.nodes())):
                for observed in itertools.combinations(self.nodes(), r):
                    independent_variables = self.active_trail_nodes(start, observed=observed)
                    independent_variables = set(independent_variables) - {start}
                    if independent_variables:
                        independencies.add_assertions([start, independent_variables,
                                                       observed])

        independencies.reduce()

        if not latex:
            return independencies
        else:
            return independencies.latex_string()

    def to_markov_model(self):
        """
        Converts bayesian model to markov model. The markov model created would
        be the moral graph of the bayesian model.

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> G = BayesianModel([('diff', 'grade'), ('intel', 'grade'),
        ...                    ('intel', 'SAT'), ('grade', 'letter')])
        >>> mm = G.to_markov_model()
        >>> mm.nodes()
        ['diff', 'grade', 'intel', 'SAT', 'letter']
        >>> mm.edges()
        [('diff', 'intel'), ('diff', 'grade'), ('intel', 'grade'),
        ('intel', 'SAT'), ('grade', 'letter')]
        """
        from pgmpy.models import MarkovModel
        moral_graph = self.moralize()
        mm = MarkovModel(moral_graph.edges())
        mm.add_factors(*[cpd.to_factor() for cpd in self.cpds])

        return mm

    def to_junction_tree(self):
        """
        Creates a junction tree (or clique tree) for a given bayesian model.

        For converting a Bayesian Model into a Clique tree, first it is converted
        into a Markov one.

        For a given markov model (H) a junction tree (G) is a graph
        1. where each node in G corresponds to a maximal clique in H
        2. each sepset in G separates the variables strictly on one side of the
        edge to other.

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors import TabularCPD
        >>> G = BayesianModel([('diff', 'grade'), ('intel', 'grade'),
        ...                    ('intel', 'SAT'), ('grade', 'letter')])
        >>> diff_cpd = TabularCPD('diff', 2, [[0.2], [0.8]])
        >>> intel_cpd = TabularCPD('intel', 3, [[0.5], [0.3], [0.2]])
        >>> grade_cpd = TabularCPD('grade', 3,
        ...                        [[0.1,0.1,0.1,0.1,0.1,0.1],
        ...                         [0.1,0.1,0.1,0.1,0.1,0.1],
        ...                         [0.8,0.8,0.8,0.8,0.8,0.8]],
        ...                        evidence=['diff', 'intel'],
        ...                        evidence_card=[2, 3])
        >>> sat_cpd = TabularCPD('SAT', 2,
        ...                      [[0.1, 0.2, 0.7],
        ...                       [0.9, 0.8, 0.3]],
        ...                      evidence=['intel'], evidence_card=[3])
        >>> letter_cpd = TabularCPD('letter', 2,
        ...                         [[0.1, 0.4, 0.8],
        ...                          [0.9, 0.6, 0.2]],
        ...                         evidence=['grade'], evidence_card=[3])
        >>> G.add_cpds(diff_cpd, intel_cpd, grade_cpd, sat_cpd, letter_cpd)
        >>> jt = G.to_junction_tree()
        """
        mm = self.to_markov_model()
        return mm.to_junction_tree()

    def fit(self, data, estimator_type=None):
        """
        Computes the CPD for each node from a given data in the form of a pandas dataframe.

        Parameters
        ----------
        data : pandas DataFrame object
            A DataFrame object with column names same as the variable names of network

        estimator: Estimator class
            Any pgmpy estimator. If nothing is specified, the default Maximum Likelihood
            estimator would be used

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> model.get_cpds()
        [<pgmpy.factors.CPD.TabularCPD at 0x7fd173b2e588>,
         <pgmpy.factors.CPD.TabularCPD at 0x7fd173cb5e10>,
         <pgmpy.factors.CPD.TabularCPD at 0x7fd173b2e470>,
         <pgmpy.factors.CPD.TabularCPD at 0x7fd173b2e198>,
         <pgmpy.factors.CPD.TabularCPD at 0x7fd173b2e2e8>]
        """

        from pgmpy.estimators import MaximumLikelihoodEstimator, BaseEstimator

        if estimator_type is None:
            estimator_type = MaximumLikelihoodEstimator
        else:
            if not isinstance(estimator_type, BaseEstimator):
                raise TypeError("Estimator object should be a valid pgmpy estimator.")

        estimator = estimator_type(self, data)

        cpds_list = estimator.get_parameters()
        self.add_cpds(*cpds_list)

    def predict(self, data):
        """
        Predicts states of all the missing variables.

        Parameters
        ----------
        data : pandas DataFrame object
            A DataFrame object with column names same as the variables in the model.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> train_data = values[:800]
        >>> predict_data = values[800:]
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> predict_data = predict_data.copy()
        >>> predict_data.drop('E', axis=1, inplace=True)
        >>> y_pred = model.predict(predict_data)
        >>> y_pred
        array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1,
               1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1,
               1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1,
               1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1,
               1, 1, 1, 0, 0, 0, 1, 0])
        """
        from pgmpy.inference import VariableElimination

        if set(data.columns) == set(self.nodes()):
            raise ValueError("No variable missing in data. Nothing to predict")

        elif set(data.columns) - set(self.nodes()):
            raise ValueError("data has variables which are not in the model")

        missing_variables = set(self.nodes()) - set(data.columns)
        pred_values = defaultdict(list)

        model_inference = VariableElimination(self)
        for index, data_point in data.iterrows():
            states_dict = model_inference.map_query(variables=missing_variables, evidence=data_point.to_dict())
            for k, v in states_dict.items():
                pred_values[k].append(v)
        return pd.DataFrame(pred_values, index=data.index)

    def get_factorized_product(self, latex=False):
        # TODO: refer to IMap class for explanation why this is not implemented.
        pass

    def is_iequivalent(self, model):
        pass

    def is_imap(self, independence):
        pass
