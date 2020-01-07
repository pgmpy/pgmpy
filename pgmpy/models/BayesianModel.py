#!/usr/bin/env python3

import itertools
from collections import defaultdict
import logging
from operator import mul
from functools import reduce

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from pgmpy.base import DAG
from pgmpy.factors.discrete import (
    TabularCPD,
    JointProbabilityDistribution,
    DiscreteFactor,
)
from pgmpy.factors.continuous import ContinuousFactor
from pgmpy.independencies import Independencies
from pgmpy.models.MarkovModel import MarkovModel


class BayesianModel(DAG):
    """
    Base class for bayesian model.

    A models stores nodes and edges with conditional probability
    distribution (cpd) and other attributes.

    models hold directed edges.  Self loops are not allowed neither
    multiple (parallel) edges.

    Nodes can be any hashable python object.

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
        super(BayesianModel, self).__init__()
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

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> G = BayesianModel()
        >>> G.add_nodes_from(['grade', 'intel'])
        >>> G.add_edge('grade', 'intel')
        """
        if u == v:
            raise ValueError("Self loops are not allowed.")
        if u in self.nodes() and v in self.nodes() and nx.has_path(self, v, u):
            raise ValueError(
                "Loops are not allowed. Adding the edge from (%s->%s) forms a loop."
                % (u, v)
            )
        else:
            super(BayesianModel, self).add_edge(u, v, **kwargs)

    def remove_node(self, node):
        """
        Remove node from the model.

        Removing a node also removes all the associated edges, removes the CPD
        of the node and marginalizes the CPDs of it's children.

        Parameters
        ----------
        node : node
            Node which is to be removed from the model.

        Returns
        -------
        None

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.models import BayesianModel
        >>> model = BayesianModel([('A', 'B'), ('B', 'C'),
        ...                        ('A', 'D'), ('D', 'C')])
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 4)),
        ...                       columns=['A', 'B', 'C', 'D'])
        >>> model.fit(values)
        >>> model.get_cpds()
        [<TabularCPD representing P(A:2) at 0x7f28248e2438>,
         <TabularCPD representing P(B:2 | A:2) at 0x7f28248e23c8>,
         <TabularCPD representing P(C:2 | B:2, D:2) at 0x7f28248e2748>,
         <TabularCPD representing P(D:2 | A:2) at 0x7f28248e26a0>]
        >>> model.remove_node('A')
        >>> model.get_cpds()
        [<TabularCPD representing P(B:2) at 0x7f28248e23c8>,
         <TabularCPD representing P(C:2 | B:2, D:2) at 0x7f28248e2748>,
         <TabularCPD representing P(D:2) at 0x7f28248e26a0>]
        """
        affected_nodes = [v for u, v in self.edges() if u == node]

        for affected_node in affected_nodes:
            node_cpd = self.get_cpds(node=affected_node)
            if node_cpd:
                node_cpd.marginalize([node], inplace=True)

        if self.get_cpds(node=node):
            self.remove_cpds(node)
        super(BayesianModel, self).remove_node(node)

    def remove_nodes_from(self, nodes):
        """
        Remove multiple nodes from the model.

        Removing a node also removes all the associated edges, removes the CPD
        of the node and marginalizes the CPDs of it's children.

        Parameters
        ----------
        nodes : list, set (iterable)
            Nodes which are to be removed from the model.

        Returns
        -------
        None

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.models import BayesianModel
        >>> model = BayesianModel([('A', 'B'), ('B', 'C'),
        ...                        ('A', 'D'), ('D', 'C')])
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 4)),
        ...                       columns=['A', 'B', 'C', 'D'])
        >>> model.fit(values)
        >>> model.get_cpds()
        [<TabularCPD representing P(A:2) at 0x7f28248e2438>,
         <TabularCPD representing P(B:2 | A:2) at 0x7f28248e23c8>,
         <TabularCPD representing P(C:2 | B:2, D:2) at 0x7f28248e2748>,
         <TabularCPD representing P(D:2 | A:2) at 0x7f28248e26a0>]
        >>> model.remove_nodes_from(['A', 'B'])
        >>> model.get_cpds()
        [<TabularCPD representing P(C:2 | D:2) at 0x7f28248e2a58>,
         <TabularCPD representing P(D:2) at 0x7f28248e26d8>]
        """
        for node in nodes:
            self.remove_node(node)

    def add_cpds(self, *cpds):
        """
        Add CPD (Conditional Probability Distribution) to the Bayesian Model.

        Parameters
        ----------
        cpds  :  list, set, tuple (array-like)
            List of CPDs which will be associated with the model

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors.discrete.CPD import TabularCPD
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
            if not isinstance(cpd, (TabularCPD, ContinuousFactor)):
                raise ValueError("Only TabularCPD or ContinuousFactor can be added.")

            if set(cpd.scope()) - set(cpd.scope()).intersection(set(self.nodes())):
                raise ValueError("CPD defined on variable not in the model", cpd)

            for prev_cpd_index in range(len(self.cpds)):
                if self.cpds[prev_cpd_index].variable == cpd.variable:
                    logging.warning(
                        "Replacing existing CPD for {var}".format(var=cpd.variable)
                    )
                    self.cpds[prev_cpd_index] = cpd
                    break
            else:
                self.cpds.append(cpd)

    def get_cpds(self, node=None):
        """
        Returns the cpd of the node. If node is not specified returns all the CPDs
        that have been added till now to the graph

        Parameters
        ----------
        node: any hashable python object (optional)
            The node whose CPD we want. If node not specified returns all the
            CPDs added to the model.

        Returns
        -------
        A list of TabularCPDs.

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> student = BayesianModel([('diff', 'grade'), ('intel', 'grade')])
        >>> cpd = TabularCPD('grade', 2, [[0.1, 0.9, 0.2, 0.7],
        ...                               [0.9, 0.1, 0.8, 0.3]],
        ...                  ['intel', 'diff'], [2, 2])
        >>> student.add_cpds(cpd)
        >>> student.get_cpds()
        """
        if node is not None:
            if node not in self.nodes():
                raise ValueError("Node not present in the Directed Graph")
            for cpd in self.cpds:
                if cpd.variable == node:
                    return cpd
            else:
                return None
        else:
            return self.cpds

    def remove_cpds(self, *cpds):
        """
        Removes the cpds that are provided in the argument.

        Parameters
        ----------
        *cpds: TabularCPD object
            A CPD object on any subset of the variables of the model which
            is to be associated with the model.

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors.discrete import TabularCPD
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

    def get_cardinality(self, node=None):
        """
        Returns the cardinality of the node. Throws an error if the CPD for the
        queried node hasn't been added to the network.

        Parameters
        ----------
        node: Any hashable python object(optional).
              The node whose cardinality we want. If node is not specified returns a
              dictionary with the given variable as keys and their respective cardinality
              as values.

        Returns
        -------
        int or dict : If node is specified returns the cardinality of the node.
                      If node is not specified returns a dictionary with the given
                      variable as keys and their respective cardinality as values.

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> student = BayesianModel([('diff', 'grade'), ('intel', 'grade')])
        >>> cpd_diff = TabularCPD('diff',2,[[0.6,0.4]]);
        >>> cpd_intel = TabularCPD('intel',2,[[0.7,0.3]]);
        >>> cpd_grade = TabularCPD('grade', 2, [[0.1, 0.9, 0.2, 0.7],
        ...                                     [0.9, 0.1, 0.8, 0.3]],
        ...                                 ['intel', 'diff'], [2, 2])
        >>> student.add_cpds(cpd_diff,cpd_intel,cpd_grade)
        >>> student.get_cardinality()
        defaultdict(int, {'diff': 2, 'grade': 2, 'intel': 2})

        >>> student.get_cardinality('intel')
        2
        """

        if node:
            return self.get_cpds(node).cardinality[0]
        else:
            cardinalities = defaultdict(int)
            for cpd in self.cpds:
                cardinalities[cpd.variable] = cpd.cardinality[0]
            return cardinalities

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

            if cpd is None:
                raise ValueError("No CPD associated with {}".format(node))
            elif isinstance(cpd, (TabularCPD, ContinuousFactor)):
                evidence = cpd.get_evidence()
                parents = self.get_parents(node)
                if set(evidence if evidence else []) != set(parents if parents else []):
                    raise ValueError(
                        "CPD associated with {node} doesn't have "
                        "proper parents associated with it.".format(node=node)
                    )
                if not cpd.is_valid_cpd():
                    raise ValueError(
                        "Sum or integral of conditional probabilites for node {node}"
                        " is not equal to 1.".format(node=node)
                    )
        return True

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
        >>> from pgmpy.factors.discrete import TabularCPD
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

    def fit(
        self, data, estimator=None, state_names=[], complete_samples_only=True, **kwargs
    ):
        """
        Estimates the CPD for each variable based on a given data set.

        Parameters
        ----------
        data: pandas DataFrame object
            DataFrame object with column names identical to the variable names of the network.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        estimator: Estimator class
            One of:
            - MaximumLikelihoodEstimator (default)
            - BayesianEstimator: In this case, pass 'prior_type' and either 'pseudo_counts'
            or 'equivalent_sample_size' as additional keyword arguments.
            See `BayesianEstimator.get_parameters()` for usage.

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states
            that the variable can take. If unspecified, the observed values
            in the data set are taken to be the only possible states.

        complete_samples_only: bool (default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.

        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.estimators import MaximumLikelihoodEstimator
        >>> data = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})
        >>> model = BayesianModel([('A', 'C'), ('B', 'C')])
        >>> model.fit(data)
        >>> model.get_cpds()
        [<TabularCPD representing P(A:2) at 0x7fb98a7d50f0>,
        <TabularCPD representing P(B:2) at 0x7fb98a7d5588>,
        <TabularCPD representing P(C:2 | A:2, B:2) at 0x7fb98a7b1f98>]
        """

        from pgmpy.estimators import (
            MaximumLikelihoodEstimator,
            BayesianEstimator,
            BaseEstimator,
        )

        if estimator is None:
            estimator = MaximumLikelihoodEstimator
        else:
            if not issubclass(estimator, BaseEstimator):
                raise TypeError("Estimator object should be a valid pgmpy estimator.")

        _estimator = estimator(
            self,
            data,
            state_names=state_names,
            complete_samples_only=complete_samples_only,
        )
        cpds_list = _estimator.get_parameters(**kwargs)
        self.add_cpds(*cpds_list)

    def predict(self, data, n_jobs=-1):
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
            E
        800 0
        801 1
        802 1
        803 1
        804 0
        ... ...
        993 0
        994 0
        995 1
        996 1
        997 0
        998 0
        999 0
        """
        from pgmpy.inference import VariableElimination

        if set(data.columns) == set(self.nodes()):
            raise ValueError("No variable missing in data. Nothing to predict")

        elif set(data.columns) - set(self.nodes()):
            raise ValueError("Data has variables which are not in the model")

        data_unique = data.drop_duplicates()
        missing_variables = set(self.nodes()) - set(data_unique.columns)
        #         pred_values = defaultdict(list)
        pred_values = []

        # Send state_names dict from one of the estimated CPDs to the inference class.
        model_inference = VariableElimination(self)
        pred_values = Parallel(n_jobs=n_jobs)(
            delayed(model_inference.map_query)(
                variables=missing_variables,
                evidence=data_point.to_dict(),
                show_progress=False,
            )
            for index, data_point in tqdm(
                data_unique.iterrows(), total=data_unique.shape[0]
            )
        )

        df_results = pd.DataFrame(pred_values, index=data_unique.index)
        data_with_results = pd.concat([data_unique, df_results], axis=1)
        return data.merge(data_with_results, how="left").loc[:, missing_variables]

    def predict_probability(self, data):
        """
        Predicts probabilities of all states of the missing variables.

        Parameters
        ----------
        data : pandas DataFrame object
            A DataFrame object with column names same as the variables in the model.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(100, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> train_data = values[:80]
        >>> predict_data = values[80:]
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> predict_data = predict_data.copy()
        >>> predict_data.drop('B', axis=1, inplace=True)
        >>> y_prob = model.predict_probability(predict_data)
        >>> y_prob
            B_0         B_1
        80  0.439178    0.560822
        81  0.581970    0.418030
        82  0.488275    0.511725
        83  0.581970    0.418030
        84  0.510794    0.489206
        85  0.439178    0.560822
        86  0.439178    0.560822
        87  0.417124    0.582876
        88  0.407978    0.592022
        89  0.429905    0.570095
        90  0.581970    0.418030
        91  0.407978    0.592022
        92  0.429905    0.570095
        93  0.429905    0.570095
        94  0.439178    0.560822
        95  0.407978    0.592022
        96  0.559904    0.440096
        97  0.417124    0.582876
        98  0.488275    0.511725
        99  0.407978    0.592022
        """
        from pgmpy.inference import VariableElimination

        if set(data.columns) == set(self.nodes()):
            raise ValueError("No variable missing in data. Nothing to predict")

        elif set(data.columns) - set(self.nodes()):
            raise ValueError("Data has variables which are not in the model")

        missing_variables = set(self.nodes()) - set(data.columns)
        pred_values = defaultdict(list)

        model_inference = VariableElimination(self)
        for index, data_point in data.iterrows():
            full_distribution = model_inference.query(
                variables=missing_variables,
                evidence=data_point.to_dict(),
                show_progress=False,
            )
            states_dict = {}
            for var in missing_variables:
                states_dict[var] = full_distribution.marginalize(
                    missing_variables - {var}, inplace=False
                )
            for k, v in states_dict.items():
                for l in range(len(v.values)):
                    state = self.get_cpds(k).state_names[k][l]
                    pred_values[k + "_" + str(state)].append(v.values[l])
        return pd.DataFrame(pred_values, index=data.index)

    def get_factorized_product(self, latex=False):
        # TODO: refer to IMap class for explanation why this is not implemented.
        pass

    def is_imap(self, JPD):
        """
        Checks whether the bayesian model is Imap of given JointProbabilityDistribution

        Parameters
        ----------
        JPD : An instance of JointProbabilityDistribution Class, for which you want to
            check the Imap

        Returns
        -------
        boolean : True if bayesian model is Imap for given Joint Probability Distribution
                False otherwise

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.factors.discrete import JointProbabilityDistribution
        >>> G = BayesianModel([('diff', 'grade'), ('intel', 'grade')])
        >>> diff_cpd = TabularCPD('diff', 2, [[0.2], [0.8]])
        >>> intel_cpd = TabularCPD('intel', 3, [[0.5], [0.3], [0.2]])
        >>> grade_cpd = TabularCPD('grade', 3,
        ...                        [[0.1,0.1,0.1,0.1,0.1,0.1],
        ...                         [0.1,0.1,0.1,0.1,0.1,0.1],
        ...                         [0.8,0.8,0.8,0.8,0.8,0.8]],
        ...                        evidence=['diff', 'intel'],
        ...                        evidence_card=[2, 3])
        >>> G.add_cpds(diff_cpd, intel_cpd, grade_cpd)
        >>> val = [0.01, 0.01, 0.08, 0.006, 0.006, 0.048, 0.004, 0.004, 0.032,
                   0.04, 0.04, 0.32, 0.024, 0.024, 0.192, 0.016, 0.016, 0.128]
        >>> JPD = JointProbabilityDistribution(['diff', 'intel', 'grade'], [2, 3, 3], val)
        >>> G.is_imap(JPD)
        True
        """
        if not isinstance(JPD, JointProbabilityDistribution):
            raise TypeError("JPD must be an instance of JointProbabilityDistribution")
        factors = [cpd.to_factor() for cpd in self.get_cpds()]
        factor_prod = reduce(mul, factors)
        JPD_fact = DiscreteFactor(JPD.variables, JPD.cardinality, JPD.values)
        if JPD_fact == factor_prod:
            return True
        else:
            return False

    def copy(self):
        """
        Returns a copy of the model.

        Returns
        -------
        BayesianModel: Copy of the model on which the method was called.

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> model = BayesianModel([('A', 'B'), ('B', 'C')])
        >>> cpd_a = TabularCPD('A', 2, [[0.2], [0.8]])
        >>> cpd_b = TabularCPD('B', 2, [[0.3, 0.7], [0.7, 0.3]],
                               evidence=['A'],
                               evidence_card=[2])
        >>> cpd_c = TabularCPD('C', 2, [[0.1, 0.9], [0.9, 0.1]],
                               evidence=['B'],
                               evidence_card=[2])
        >>> model.add_cpds(cpd_a, cpd_b, cpd_c)
        >>> copy_model = model.copy()
        >>> copy_model.nodes()
        ['C', 'A', 'B']
        >>> copy_model.edges()
        [('A', 'B'), ('B', 'C')]
        >>> copy_model.get_cpds()
        [<TabularCPD representing P(A:2) at 0x7f2824930a58>,
         <TabularCPD representing P(B:2 | A:2) at 0x7f2824930a90>,
         <TabularCPD representing P(C:2 | B:2) at 0x7f2824944240>]
        """
        model_copy = BayesianModel()
        model_copy.add_nodes_from(self.nodes())
        model_copy.add_edges_from(self.edges())
        if self.cpds:
            model_copy.add_cpds(*[cpd.copy() for cpd in self.cpds])
        return model_copy

    def get_markov_blanket(self, node):
        """
        Returns a markov blanket for a random variable. In the case
        of Bayesian Networks, the markov blanket is the set of
        node's parents, its children and its children's other parents.

        Returns
        -------
        list(blanket_nodes): List of nodes contained in Markov Blanket

        Parameters
        ----------
        node: string, int or any hashable python object.
              The node whose markov blanket would be returned.

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> G = BayesianModel([('x', 'y'), ('z', 'y'), ('y', 'w'), ('y', 'v'), ('u', 'w'),
                               ('s', 'v'), ('w', 't'), ('w', 'm'), ('v', 'n'), ('v', 'q')])
        >>> bayes_model.get_markov_blanket('y')
        ['s', 'w', 'x', 'u', 'z', 'v']
        """
        children = self.get_children(node)
        parents = self.get_parents(node)
        blanket_nodes = children + parents
        for child_node in children:
            blanket_nodes.extend(self.get_parents(child_node))
        blanket_nodes = set(blanket_nodes)
        blanket_nodes.remove(node)
        return list(blanket_nodes)
