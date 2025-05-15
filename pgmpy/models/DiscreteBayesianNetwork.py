#!/usr/bin/env python3

import itertools
from collections import defaultdict
from functools import reduce
from operator import mul

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from pgmpy.base import DAG
from pgmpy.factors.discrete import (
    DiscreteFactor,
    JointProbabilityDistribution,
    TabularCPD,
)
from pgmpy.global_vars import logger
from pgmpy.models.MarkovNetwork import MarkovNetwork
from pgmpy.utils import compat_fns


class DiscreteBayesianNetwork(DAG):
    """
    Initializes a Bayesian Network.
    A models stores nodes and edges with conditional probability
    distribution (cpd) and other attributes.

    Models hold directed edges.  Self loops are not allowed neither
    multiple (parallel) edges.

    Nodes can be any hashable python object.

    Edges are represented as links between nodes.

    Parameters
    ----------
    ebunch: input graph
        Data to initialize graph.  If ebunch=None (default) an empty
        graph is created.  The ebunch can be an edge list, or any
        NetworkX graph object.

    latents: list, array-like
        List of variables which are latent (i.e. unobserved) in the model.

    Examples
    --------
    Create an empty Bayesian Network with no nodes and no edges.

    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> G = DiscreteBayesianNetwork()

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

    def __init__(self, ebunch=None, latents=set(), lavaan_str=None, dagitty_str=None):
        super(DiscreteBayesianNetwork, self).__init__(
            ebunch=ebunch,
            latents=latents,
            lavaan_str=lavaan_str,
            dagitty_str=dagitty_str,
        )
        self.cpds = []
        self.cardinalities = defaultdict(int)

    def add_edge(self, u, v, w=None, **kwargs):
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
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> G = DiscreteBayesianNetwork()
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
            if w:
                super(DiscreteBayesianNetwork, self).add_edge(u, v, w, **kwargs)
            else:
                super(DiscreteBayesianNetwork, self).add_edge(u, v, **kwargs)

    def remove_node(self, node):
        """
        Remove node from the model.

        Removing a node also removes all the associated edges, removes the CPD
        of the node and marginalizes the CPDs of its children.

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
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> model = DiscreteBayesianNetwork([('A', 'B'), ('B', 'C'),
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

        self.latents = self.latents - set([node])

        super(DiscreteBayesianNetwork, self).remove_node(node)

    def remove_nodes_from(self, nodes):
        """
        Remove multiple nodes from the model.

        Removing a node also removes all the associated edges, removes the CPD
        of the node and marginalizes the CPDs of its children.

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
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> model = DiscreteBayesianNetwork([('A', 'B'), ('B', 'C'),
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
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete.CPD import TabularCPD
        >>> student = DiscreteBayesianNetwork([('diff', 'grades'), ('aptitude', 'grades')])
        >>> grades_cpd = TabularCPD('grades', 3, [[0.1,0.1,0.1,0.1,0.1,0.1],
        ...                                       [0.1,0.1,0.1,0.1,0.1,0.1],
        ...                                       [0.8,0.8,0.8,0.8,0.8,0.8]],
        ...                         evidence=['diff', 'aptitude'], evidence_card=[2, 3],
        ...                         state_names={'grades': ['gradeA', 'gradeB', 'gradeC'],
        ...                                      'diff': ['easy', 'hard'],
        ...                                      'aptitude': ['low', 'medium', 'high']})
        >>> student.add_cpds(grades_cpd)

        +---------+-------------------------+------------------------+
        |diff:    |          easy           |         hard           |
        +---------+------+--------+---------+------+--------+--------+
        |aptitude:| low  | medium |  high   | low  | medium |  high  |
        +---------+------+--------+---------+------+--------+--------+
        |gradeA   | 0.1  | 0.1    |   0.1   |  0.1 |  0.1   |   0.1  |
        +---------+------+--------+---------+------+--------+--------+
        |gradeB   | 0.1  | 0.1    |   0.1   |  0.1 |  0.1   |   0.1  |
        +---------+------+--------+---------+------+--------+--------+
        |gradeC   | 0.8  | 0.8    |   0.8   |  0.8 |  0.8   |   0.8  |
        +---------+------+--------+---------+------+--------+--------+
        """
        for cpd in cpds:
            if not isinstance(cpd, TabularCPD):
                raise ValueError("Only TabularCPD can be added.")

            if set(cpd.scope()) - set(cpd.scope()).intersection(set(self.nodes())):
                raise ValueError("CPD defined on variable not in the model", cpd)

            for prev_cpd_index in range(len(self.cpds)):
                if self.cpds[prev_cpd_index].variable == cpd.variable:
                    logger.warning(f"Replacing existing CPD for {cpd.variable}")
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
        cpd : TabularCPD object or list of TabularCPD objects
            If 'node' is specified, returns the 'TabularCPD' object corresponding to the node.
            If 'node' is not specified, returns a list of all 'TabularCPD' objects added to the model.

        Raises
        ------
        ValueError
            If the specified node is not present in the model.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model
        >>> model = get_example_model('asia')
        >>> cpds = model.get_cpds()
        >>> cpds
        [<TabularCPD representing P(asia:2) at 0x7dbbd9bdbb80>,
        <TabularCPD representing P(bronc:2 | smoke:2) at 0x7dbbd9bda3e0>,
        <TabularCPD representing P(dysp:2 | bronc:2, either:2) at 0x7dbbd9bd8550>,
        <TabularCPD representing P(either:2 | lung:2, tub:2) at 0x7dbbd9bda800>,
        <TabularCPD representing P(lung:2 | smoke:2) at 0x7dbbd9bd89d0>,
        <TabularCPD representing P(smoke:2) at 0x7dbbd9bd8f70>,
        <TabularCPD representing P(tub:2 | asia:2) at 0x7dbbd9bda860>,
        <TabularCPD representing P(xray:2 | either:2) at 0x7dbbd9bd9a80>]
        >>> cpd = model.get_cpds('bronc')
        >>> cpd
        <TabularCPD representing P(bronc:2 | smoke:2) at 0x7dbbd9bda3e0>
        """
        if node is not None:
            if node not in self.nodes():
                raise ValueError("Node not present in the Directed Graph")
            else:
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
        *cpds: TabularCPD object
            A CPD object on any subset of the variables of the model which
            is to be associated with the model.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> student = DiscreteBayesianNetwork([('diff', 'grade'), ('intel', 'grade')])
        >>> cpd = TabularCPD('grade', 2, [[0.1, 0.9, 0.2, 0.7],
        ...                               [0.9, 0.1, 0.8, 0.3]],
        ...                  ['intel', 'diff'], [2, 2])
        >>> student.add_cpds(cpd)
        >>> student.remove_cpds(cpd)
        """
        for cpd in cpds:
            if isinstance(cpd, (str, int)):
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
        variable cardinalities: dict or int
            If node is specified returns the cardinality of the node else returns a dictionary
            with the cardinality of each variable in the network

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> student = DiscreteBayesianNetwork([('diff', 'grade'), ('intel', 'grade')])
        >>> cpd_diff = TabularCPD('diff', 2, [[0.6], [0.4]]);
        >>> cpd_intel = TabularCPD('intel', 2, [[0.7], [0.3]]);
        >>> cpd_grade = TabularCPD('grade', 2, [[0.1, 0.9, 0.2, 0.7],
        ...                                     [0.9, 0.1, 0.8, 0.3]],
        ...                                 ['intel', 'diff'], [2, 2])
        >>> student.add_cpds(cpd_diff,cpd_intel,cpd_grade)
        >>> student.get_cardinality()
        defaultdict(<class 'int'>, {'diff': 2, 'intel': 2, 'grade': 2})

        >>> student.get_cardinality('intel')
        2
        """

        if node is not None:
            return self.get_cpds(node).cardinality[0]
        else:
            cardinalities = defaultdict(int)
            for cpd in self.cpds:
                cardinalities[cpd.variable] = cpd.cardinality[0]
            return cardinalities

    @property
    def states(self):
        """
        Returns a dictionary mapping each node to its list of possible states.

        Returns
        -------
        state_dict: dict
            Dictionary of nodes to possible states
        """
        state_names_list = [cpd.state_names for cpd in self.cpds]
        state_dict = {
            node: states for d in state_names_list for node, states in d.items()
        }
        return state_dict

    def check_model(self):
        """
        Check the model for various errors. This method checks for the following
        errors.

        * Checks if the sum of the probabilities for each state is equal to 1 (tol=0.01).
        * Checks if the CPDs associated with nodes are consistent with their parents.

        Returns
        -------
        check: boolean
            True if all the checks pass otherwise should throw an error.
        """
        for node in self.nodes():
            cpd = self.get_cpds(node=node)

            # Check if a CPD is associated with every node.
            if cpd is None:
                raise ValueError(f"No CPD associated with {node}")

            # Check if the CPD is an instance of TabularCPD.
            elif isinstance(cpd, TabularCPD):
                evidence = cpd.get_evidence()
                parents = self.get_parents(node)

                # Check if the evidence set of the CPD is same as its parents.
                if set(evidence) != set(parents):
                    raise ValueError(
                        f"CPD associated with {node} doesn't have proper parents associated with it."
                    )

                if len(set(cpd.variables) - set(cpd.state_names.keys())) > 0:
                    raise ValueError(
                        f"CPD for {node} doesn't have state names defined for all the variables."
                    )

                # Check if the values of the CPD sum to 1.
                if not cpd.is_valid_cpd():
                    raise ValueError(
                        f"Sum or integral of conditional probabilities for node {node} is not equal to 1."
                    )

        for node in self.nodes():
            cpd = self.get_cpds(node=node)
            for index, node in enumerate(cpd.variables[1:]):
                parent_cpd = self.get_cpds(node)
                # Check if the evidence cardinality specified is same as parent's cardinality
                if parent_cpd.cardinality[0] != cpd.cardinality[1 + index]:
                    raise ValueError(
                        f"The cardinality of {node} doesn't match in it's child nodes."
                    )
                # Check if the state_names are the same in parent and child CPDs.
                if parent_cpd.state_names[node] != cpd.state_names[node]:
                    raise ValueError(
                        f"The state names of {node} doesn't match in it's child nodes."
                    )

        return True

    def to_markov_model(self):
        """
        Converts Bayesian Network to Markov Model. The Markov Model created would
        be the moral graph of the Bayesian Network.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> G = DiscreteBayesianNetwork([('diff', 'grade'), ('intel', 'grade'),
        ...                    ('intel', 'SAT'), ('grade', 'letter')])
        >>> mm = G.to_markov_model()
        >>> mm.nodes()
        NodeView(('diff', 'grade', 'intel', 'letter', 'SAT'))
        >>> mm.edges()
        EdgeView([('diff', 'grade'), ('diff', 'intel'), ('grade', 'letter'), ('grade', 'intel'), ('intel', 'SAT')])
        """
        moral_graph = self.moralize()
        mm = MarkovNetwork(moral_graph.edges())
        mm.add_nodes_from(moral_graph.nodes())
        mm.add_factors(*[cpd.to_factor() for cpd in self.cpds])

        return mm

    def to_junction_tree(self):
        """
        Creates a junction tree (or clique tree) for a given Bayesian Network.

        For converting a Bayesian Model into a Clique tree, first it is converted
        into a Markov one.

        For a given markov model (H) a junction tree (G) is a graph
        1. where each node in G corresponds to a maximal clique in H
        2. each sepset in G separates the variables strictly on one side of the
        edge to other.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> G = DiscreteBayesianNetwork([('diff', 'grade'), ('intel', 'grade'),
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

    def fit_update(self, data, n_prev_samples=None, n_jobs=1):
        """
        Method to update the parameters of the DiscreteBayesianNetwork with more data.
        Internally, uses BayesianEstimator with dirichlet prior, and uses
        the current CPDs (along with `n_prev_samples`) to compute the pseudo_counts.

        Parameters
        ----------
        data: pandas.DataFrame
            The new dataset which to use for updating the model.

        n_prev_samples: int
            The number of samples/datapoints on which the model was trained before.
            This parameter determines how much weight should the new data be given.
            If None, n_prev_samples = nrow(data).

        n_jobs: int (default: 1)
            Number of threads/processes to use for estimation. Using n_jobs > 1
            for small models or datasets might be slower.

        Returns
        -------
        Updated model: None
            Modifies the network inplace.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model
        >>> from pgmpy.sampling import BayesianModelSampling
        >>> model = get_example_model('alarm')
        >>> # Generate some new data.
        >>> data = BayesianModelSampling(model).forward_sample(int(1e3))
        >>> model.fit_update(data)
        """
        from pgmpy.estimators import BayesianEstimator

        if n_prev_samples is None:
            n_prev_samples = data.shape[0]

        # Step 1: Compute the pseudo_counts for the dirichlet prior.
        pseudo_counts = {
            var: compat_fns.to_numpy(self.get_cpds(var).get_values()) * n_prev_samples
            for var in data.columns
        }

        # Step 2: Get the current order of state names for aligning pseudo counts.
        state_names = {}
        for var in data.columns:
            state_names.update(self.get_cpds(var).state_names)

        # Step 3: Estimate the new CPDs.
        _est = BayesianEstimator(self, data, state_names=state_names)
        cpds = _est.get_parameters(
            prior_type="dirichlet", pseudo_counts=pseudo_counts, n_jobs=n_jobs
        )

        # Temporarily disable logger to stop giving warning about replacing CPDs.
        logger.disabled = True
        self.add_cpds(*cpds)
        logger.disabled = False

    def predict(
        self, data, algo=None, stochastic=False, n_jobs=-1, seed=None, **kwargs
    ):
        """
        Predicts states of all the missing variables.

        Parameters
        ----------
        data: pandas DataFrame object
            A DataFrame object with column names same as the variables in the model.

        algo: a subclass of pgmpy.inference.Inference or pgmpy.inference.ApproxInference
            An algorithm class from pgmpy Inference algorithms. Default is Variable Elimination.

        stochastic: boolean
            If True, does prediction by sampling from the distribution of predicted variable(s).
            If False, returns the states with the highest probability value (i.e. MAP) for the
                predicted variable(s).

        n_jobs: int (default: -1)
            The number of CPU cores to use. If -1, uses all available cores.

        seed: int (default: None)
            When `stochastic=True`, the seed value to use for random number generators.

        **kwargs
            Optional keyword arguments specific to the selected algorithm.
            - Variable Elimination:
            - elimination_order: str or list (default='greedy')
                Order in which to eliminate the variables in the algorithm. If list is provided,
                should contain all variables in the model except the ones in `variables`. str options
                are: `greedy`, `WeightedMinFill`, `MinNeighbors`, `MinWeight`, `MinFill`. Please
                refer https://pgmpy.org/exact_infer/ve.html#module-pgmpy.inference.EliminationOrder
                for details.

            - joint: boolean (should only be used with stochastic=True i.e. when not calculating MAP)
                If True, returns a Joint Distribution over `variables`.
                If False, returns a dict of distributions over each of the `variables`.

            - Belief Propagation:
                - joint: boolean (should only be used with stochastic=True i.e. when not calculating MAP)
                If True, returns a Joint Distribution over `variables`.
                If False, returns a dict of distributions over each of the `variables`.

            - Approx Inference:
                - n_samples: int
                    The number of samples to generate for computing the distributions. Higher `n_samples`
                    results in more accurate results at the cost of more computation time.

                - samples: pd.DataFrame (default: None)
                    If provided, uses these samples to compute the distribution instead
                    of generating samples. `samples` **must** conform with the
                    `evidence` and `virtual_evidence`.

                - state_names: dict (default: None)
                    A dict of state names for each variable in `variables` in the form {variable_name: list of states}.
                    If None, inferred from the data but is possible that the final distribution misses some states.

                - seed: int (default: None)
                    Sets the seed for the random generators.

                - joint: boolean (should only be used with stochastic=True i.e. when not calculating MAP)
                    If True, returns a Joint Distribution over `variables`.
                    If False, returns a dict of distributions over each of the `variables`.

        Returns
        -------
        Inference results: Pandas DataFrame
            If `stochastic` is True, returns state(s) by sampling from the distribution of predicted variables.
            If `stochastic` is False, returns state(s) with the highest probability value.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.inference import ApproxInference
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> train_data = values[:800]
        >>> predict_data = values[800:]
        >>> model = DiscreteBayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(train_data)
        >>> predict_data = predict_data.copy()
        >>> predict_data.drop('E', axis=1, inplace=True)
        >>> approx_inf_parameters = {'n_samples':int(1e3),'seed':42}
        >>> y_pred = model.predict(predict_data,algo=ApproxInference,**approx_inf_parameters)
        >>> y_pred['E']
            E
        800 1
        801 0
        802 1
        803 1
        804 1
        ... ...
        995 1
        996 1
        997 1
        998 1
        999 0
        """
        from pgmpy.inference import (
            ApproxInference,
            BeliefPropagation,
            Inference,
            VariableElimination,
        )

        if set(data.columns) == set(self.nodes()):
            raise ValueError("No variable missing in data. Nothing to predict")

        elif set(data.columns) - set(self.nodes()):
            raise ValueError("Data has variables which are not in the model")

        missing_variables = set(self.nodes()) - set(data.columns)

        if algo is None:
            algo = VariableElimination
        else:
            if not issubclass(algo, Inference) and algo is not ApproxInference:
                raise TypeError(
                    f"Algorithm should be a valid pgmpy inference method. Got {type(algo)} instead."
                )

        model_inference = algo(self)
        data_unique_indexes = data.groupby(list(data.columns), dropna=False).apply(
            lambda t: t.index.tolist()
        )
        data_unique = data_unique_indexes.index.to_frame()
        pred_values = Parallel(n_jobs=n_jobs)(
            delayed(model_inference.query if stochastic else model_inference.map_query)(
                variables=missing_variables.union(
                    set(data_point.index[data_point.isna()])
                ),
                evidence=data_point[~data_point.isna()].to_dict(),
                show_progress=False,
                **kwargs,
            )
            for index, data_point in tqdm(
                data_unique.iterrows(), total=data_unique.shape[0]
            )
        )

        all_columns = data.columns.tolist() + [col for col in missing_variables]
        predictions = pd.DataFrame()

        for i, row in enumerate(data_unique_indexes):
            if stochastic:
                predicted_df = (
                    pred_values[i].sample(n=len(row), seed=seed).reset_index(drop=True)
                )
            else:
                predicted = pd.DataFrame(pred_values[i], index=[0])
                predicted_df = predicted.loc[
                    predicted.index.repeat(len(row))
                ].reset_index(drop=True)

            initial_variables = data_unique.iloc[[i]].reset_index(drop=True)
            known_variables = initial_variables.dropna(axis=1)
            known_df = known_variables.loc[
                known_variables.index.repeat(len(row))
            ].reset_index(drop=True)

            complete_data = pd.concat([predicted_df, known_df], axis="columns")
            complete_data.index = row
            complete_data = complete_data.reindex(columns=all_columns)
            predictions = pd.concat([predictions, complete_data], copy=False)

        return predictions.sort_index()

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
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(100, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> train_data = values[:80]
        >>> predict_data = values[80:]
        >>> model = DiscreteBayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
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
        for _, data_point in data.iterrows():
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

    def get_state_probability(self, states):
        """
        Given a fully specified Bayesian Network, returns the probability of the given set
        of states.

        Parameters
        ----------
        state: dict
            dict of the form {variable: state}

        Returns
        -------
        float: The probability value

        Examples
        --------
        >>> from pgmpy.utils import get_example_model
        >>> model = get_example_model('asia')
        >>> model.get_state_probability({'either': 'no', 'tub': 'no', 'xray': 'yes', 'bronc': 'no'})
        0.02605122
        """
        # Step 1: Check that all variables and states are in the model.
        self.check_model()
        for var, state in states.items():
            if var not in self.nodes():
                raise ValueError(f"{var} not in the model.")
            if state not in self.states[var]:
                raise ValueError(f"State: {state} not define for {var}")

        # Step 2: Missing variables in states.
        missing_vars = list(set(self.nodes()) - set(states.keys()))
        missing_var_states = {var: self.states[var] for var in missing_vars}

        # Step 2: Compute the probability
        final_prob = 0
        for state_comb in itertools.product(*missing_var_states.values()):
            temp_states = {
                **{var: state_comb[i] for i, var in enumerate(missing_vars)},
                **states,
            }
            prob = 1
            for cpd in self.cpds:
                index = []
                for var in cpd.variables:
                    index.append(cpd.name_to_no[var][temp_states[var]])
                prob *= cpd.values[tuple(index)]
            final_prob += prob

        return final_prob

    def get_factorized_product(self, latex=False):
        # TODO: refer to IMap class for explanation why this is not implemented.
        pass

    def is_imap(self, JPD):
        """
        Checks whether the Bayesian Network is Imap of given JointProbabilityDistribution

        Parameters
        ----------
        JPD: An instance of JointProbabilityDistribution Class, for which you want to check the Imap

        Returns
        -------
        is IMAP: True or False
            True if Bayesian Network is Imap for given Joint Probability Distribution False otherwise

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.factors.discrete import JointProbabilityDistribution
        >>> G = DiscreteBayesianNetwork([('diff', 'grade'), ('intel', 'grade')])
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
        Model's copy: pgmpy.models.DiscreteBayesianNetwork
            Copy of the model on which the method was called.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> model = DiscreteBayesianNetwork([('A', 'B'), ('B', 'C')])
        >>> cpd_a = TabularCPD('A', 2, [[0.2], [0.8]])
        >>> cpd_b = TabularCPD('B', 2, [[0.3, 0.7], [0.7, 0.3]],
        ...                    evidence=['A'],
        ...                    evidence_card=[2])
        >>> cpd_c = TabularCPD('C', 2, [[0.1, 0.9], [0.9, 0.1]],
        ...                    evidence=['B'],
        ...                    evidence_card=[2])
        >>> model.add_cpds(cpd_a, cpd_b, cpd_c)
        >>> copy_model = model.copy()
        >>> copy_model.nodes()
        NodeView(('A', 'B', 'C'))
        >>> copy_model.edges()
        OutEdgeView([('A', 'B'), ('B', 'C')])
        >>> len(copy_model.get_cpds())
        3
        """
        model_copy = DiscreteBayesianNetwork()
        model_copy.add_nodes_from(self.nodes())
        model_copy.add_edges_from(self.edges())
        if self.cpds:
            model_copy.add_cpds(*[cpd.copy() for cpd in self.cpds])
        model_copy.latents = self.latents
        return model_copy

    def get_markov_blanket(self, node):
        """
        Returns a markov blanket for a random variable. In the case
        of Bayesian Networks, the markov blanket is the set of
        node's parents, its children and its children's other parents.

        Returns
        -------
        Markov Blanket: list
            List of nodes contained in Markov Blanket of `node`

        Parameters
        ----------
        node: string, int or any hashable python object.
              The node whose markov blanket would be returned.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> G = DiscreteBayesianNetwork([('x', 'y'), ('z', 'y'), ('y', 'w'), ('y', 'v'), ('u', 'w'),
        ...                    ('s', 'v'), ('w', 't'), ('w', 'm'), ('v', 'n'), ('v', 'q')])
        >>> G.get_markov_blanket('y')
        ['s', 'u', 'w', 'v', 'z', 'x']
        """
        children = self.get_children(node)
        parents = self.get_parents(node)
        blanket_nodes = children + parents
        for child_node in children:
            blanket_nodes.extend(self.get_parents(child_node))
        blanket_nodes = set(blanket_nodes)
        blanket_nodes.discard(node)
        return list(blanket_nodes)

    @staticmethod
    def get_random(
        n_nodes=5,
        edge_prob=0.5,
        node_names=None,
        n_states=None,
        latents=False,
        seed=None,
    ):
        """
        Returns a randomly generated Bayesian Network on `n_nodes` variables
        with edge probabiliy of `edge_prob` between variables.

        Parameters
        ----------
        n_nodes: int
            The number of nodes in the randomly generated DAG.

        edge_prob: float
            The probability of edge between any two nodes in the topologically
            sorted DAG.

        node_names: list (default: None)
            A list of variables names to use in the random graph.
            If None, the node names are integer values starting from 0.

        n_states: int or dict (default: None)
            The number of states of each variable in the form
            {variable: no_of_states}. If a single value is provided,
            all nodes will have the same number of states. When None
            randomly generates the number of states.

        latents: bool (default: False)
            If True, also creates latent variables.

        seed: int (default: None)
            The seed value for random number generators.

        Returns
        -------
        Random DAG: pgmpy.base.DAG
            The randomly generated DAG.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> model = DiscreteBayesianNetwork.get_random(n_nodes=5)
        >>> model.nodes()
        NodeView((0, 1, 3, 4, 2))
        >>> model.edges()
        OutEdgeView([(0, 1), (0, 3), (1, 3), (1, 4), (3, 4), (2, 3)])
        >>> model.cpds
        [<TabularCPD representing P(0:0) at 0x7f97e16eabe0>,
         <TabularCPD representing P(1:1 | 0:0) at 0x7f97e16ea670>,
         <TabularCPD representing P(3:3 | 0:0, 1:1, 2:2) at 0x7f97e16820d0>,
         <TabularCPD representing P(4:4 | 1:1, 3:3) at 0x7f97e16eae80>,
         <TabularCPD representing P(2:2) at 0x7f97e1682c40>]
        """
        if node_names is None:
            node_names = list([f"X_{i}" for i in range(n_nodes)])

        if n_states is None:
            gen = np.random.default_rng(seed=seed)
            n_states = gen.integers(low=1, high=5, size=n_nodes)
            n_states_dict = {node_names[i]: n_states[i] for i in range(n_nodes)}

        elif isinstance(n_states, int):
            n_states = np.array([n_states] * n_nodes)
            n_states_dict = {node_names[i]: n_states[i] for i in range(n_nodes)}

        elif isinstance(n_states, dict):
            n_states_dict = n_states

        dag = DAG.get_random(
            n_nodes=n_nodes,
            edge_prob=edge_prob,
            node_names=node_names,
            latents=latents,
            seed=seed,
        )
        bn_model = DiscreteBayesianNetwork(dag.edges(), latents=dag.latents)
        bn_model.add_nodes_from(dag.nodes())

        cpds = []
        for node in bn_model.nodes():
            parents = list(bn_model.predecessors(node))
            cpds.append(
                TabularCPD.get_random(
                    variable=node,
                    evidence=parents,
                    cardinality=n_states_dict,
                    seed=seed,
                )
            )

        bn_model.add_cpds(*cpds)
        return bn_model

    def get_random_cpds(self, n_states=None, inplace=False, seed=None):
        """
        Given a `model`, generates and adds random `TabularCPD` for each node resulting in a fully parameterized network.

        Parameters
        ----------
        n_states: int or dict (default: None)
            The number of states of each variable in the `model`. If None, randomly
            generates the number of states.

        inplace: bool (default: False)
            If inplace=True, adds the generated TabularCPDs to `model` itself, else creates
            a copy of the model.

        seed: int (default: None)
            The seed value for random number generators.

        """
        if isinstance(n_states, int):
            n_states = {var: n_states for var in self.nodes()}
        elif isinstance(n_states, dict):
            if set(n_states.keys()) != set(self.nodes()):
                raise ValueError("Number of states not specified for each variable")
        elif n_states is None:
            gen = np.random.default_rng(seed=seed)
            n_states = {
                var: gen.integers(low=1, high=5, size=1)[0] for var in self.nodes()
            }

        cpds = []
        for node in self.nodes():
            parents = list(self.predecessors(node))
            cpds.append(
                TabularCPD.get_random(
                    variable=node, evidence=parents, cardinality=n_states, seed=seed
                )
            )

        if inplace:
            self.add_cpds(*cpds)
        else:
            return cpds

    def do(self, nodes, inplace=False):
        """
        Applies the do operation. The do operation removes all incoming edges
        to variables in `nodes` and marginalizes their CPDs to only contain the
        variable itself.

        Parameters
        ----------
        nodes : list, array-like
            The names of the nodes to apply the do-operator for.

        inplace: boolean (default: False)
            If inplace=True, makes the changes to the current object,
            otherwise returns a new instance.

        Returns
        -------
        Modified network: pgmpy.models.DiscreteBayesianNetwork or None
            If inplace=True, modifies the object itself else returns an instance of
            DiscreteBayesianNetwork modified by the do operation.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model
        >>> asia = get_example_model('asia')
        >>> asia.edges()
        OutEdgeView([('asia', 'tub'), ('tub', 'either'), ('smoke', 'lung'), ('smoke', 'bronc'),
                     ('lung', 'either'), ('bronc', 'dysp'), ('either', 'xray'), ('either', 'dysp')])
        >>> do_bronc = asia.do(['bronc'])
        OutEdgeView([('asia', 'tub'), ('tub', 'either'), ('smoke', 'lung'), ('lung', 'either'),
                     ('bronc', 'dysp'), ('either', 'xray'), ('either', 'dysp')])
        """
        if isinstance(nodes, (str, int)):
            nodes = [nodes]
        else:
            nodes = list(nodes)

        if not set(nodes).issubset(set(self.nodes())):
            raise ValueError(
                f"Nodes not found in the model: {set(nodes) - set(self.nodes)}"
            )

        model = self if inplace else self.copy()
        adj_model = DAG.do(model, nodes, inplace=inplace)

        if adj_model.cpds:
            for node in nodes:
                cpd = adj_model.get_cpds(node=node)
                cpd.marginalize(cpd.variables[1:], inplace=True)
        return adj_model

    def simulate(
        self,
        n_samples=10,
        do=None,
        evidence=None,
        virtual_evidence=None,
        virtual_intervention=None,
        include_latents=False,
        partial_samples=None,
        seed=None,
        show_progress=True,
        missing_prob=None,
        return_full=False,
    ):
        """
        Simulates data from the given model. Internally uses methods from
        pgmpy.sampling.BayesianModelSampling to generate the data.

        Parameters
        ----------
        n_samples: int
            The number of data samples to simulate from the model.

        do: dict
            The interventions to apply to the model. dict should be of the form
            {variable_name: state}

        evidence: dict
            Observed evidence to apply to the model. dict should be of the form
            {variable_name: state}

        virtual_evidence: list
            Probabilistically apply evidence to the model. `virtual_evidence` should
            be a list of `pgmpy.factors.discrete.TabularCPD` objects specifying the
            virtual probabilities.

        virtual_intervention: list
            Also known as soft intervention. `virtual_intervention` should be a list
            of `pgmpy.factors.discrete.TabularCPD` objects specifying the virtual/soft
            intervention probabilities.

        include_latents: boolean
            Whether to include the latent variable values in the generated samples.

        partial_samples: pandas.DataFrame
            A pandas dataframe specifying samples on some of the variables in the model. If
            specified, the sampling procedure uses these sample values, instead of generating them.
            partial_samples.shape[0] must be equal to `n_samples`.

        seed: int (default: None)
            If a value is provided, sets the seed for numpy.random.

        show_progress: bool
            If True, shows a progress bar when generating samples.

        missing_prob: TabularCPD, list  (default: None)
            The probability of missing value for the variable of TabularCPD.
            In case of missing value for more than one variable, provide list of TabularCPD.
            The variable name of each TabularCPD should end with the name of node in DiscreteBayesianNetwork with * at the end of the name.
            The state names of each TabularCPD should be the same as the state names of the corresponding node in DiscreteBayesianNetwork.

        return_full: bool (default: False)
            If True, return both full samples and samples with missing values (if performed).

        Returns
        -------
        A dataframe with the simulated data: pd.DataFrame

        Examples
        --------
        >>> from pgmpy.utils import get_example_model

        Simulation without any evidence or intervention:

        >>> model = get_example_model('alarm')
        >>> model.simulate(n_samples=10)

        Simulation with the hard evidence: MINVOLSET = HIGH:

        >>> model.simulate(n_samples=10, evidence={"MINVOLSET": "HIGH"})

        Simulation with hard intervention: CVP = LOW:

        >>> model.simulate(n_samples=10, do={"CVP": "LOW"})

        Simulation with virtual/soft evidence: p(MINVOLSET=LOW) = 0.8, p(MINVOLSET=HIGH) = 0.2,
        p(MINVOLSET=NORMAL) = 0:

        >>> virt_evidence = [TabularCPD("MINVOLSET", 3, [[0.8], [0.0], [0.2]], state_names={"MINVOLSET": ["LOW", "NORMAL", "HIGH"]})]
        >>> model.simulate(n_samples, virtual_evidence=virt_evidence)

        Simulation with virtual/soft intervention: p(CVP=LOW) = 0.2, p(CVP=NORMAL)=0.5, p(CVP=HIGH)=0.3:

        >>> virt_intervention = [TabularCPD("CVP", 3, [[0.2], [0.5], [0.3]], state_names={"CVP": ["LOW", "NORMAL", "HIGH"]})]
        >>> model.simulate(n_samples, virtual_intervention=virt_intervention)

        Simulation with missing values:
        >>> from pgmpy.factors.discrete.CPD import TabularCPD
        >>> cpd = TabularCPD("HISTORY*", 2, [[0.5], [0.5]])
        >>> model.simulate(n_samples, missing_prob=cpd)

        >>> cpd = TabularCPD("HISTORY*", 2, [[0.5, 0.5], [0.5, 0.5]],["HISTORY"], [2], state_names={"HISTORY*" : [0,1],
                        "HISTORY" : ['TRUE', 'FALSE']})
        >>> model.simulate(n_samples, missing_prob=cpd)

        >>> cpd = TabularCPD("HISTORY*", 2, [[0.2, 0.1, 0.6, 0.4, 0.7, 0.2], [0.8, 0.9, 0.4, 0.6, 0.3, 0.8]],
                            ["HYPOVOLEMIA", "LVEDVOLUME"], [2, 3], state_names={"HISTORY*" : [0,1],
                        "HYPOVOLEMIA" : ['TRUE', 'FALSE'], 'LVEDVOLUME': ['LOW', 'NORMAL', 'HIGH']})
        >>> model.simulate(n_samples=10, missing_prob=cpd)
        """
        from pgmpy.sampling import BayesianModelSampling

        self.check_model()
        model = self.copy()
        state_names = self.states

        evidence = {} if evidence is None else evidence
        for var, state in evidence.items():
            if state not in state_names[var]:
                raise ValueError(f"Evidence state: {state} for {var} doesn't exist")

        do = {} if do is None else do
        for var, state in do.items():
            if state not in state_names[var]:
                raise ValueError(f"Do state: {state} for {var} doesn't exist")

        virtual_intervention = (
            [] if virtual_intervention is None else virtual_intervention
        )
        virtual_evidence = [] if virtual_evidence is None else virtual_evidence

        if set(do.keys()).intersection(set(evidence.keys())):
            raise ValueError("Variable can't be in both do and evidence")

        # Step 1: If do or virtual_intervention is specified, modify the network structure.
        if (do != {}) or (virtual_intervention != []):
            virt_nodes = [cpd.variables[0] for cpd in virtual_intervention]
            model = model.do(list(do.keys()) + virt_nodes)
            evidence = {**evidence, **do}
            virtual_evidence = [*virtual_evidence, *virtual_intervention]

        # Step 2: If virtual_evidence; modify the network structure
        if virtual_evidence != []:
            for cpd in virtual_evidence:
                var = cpd.variables[0]
                if var not in model.nodes():
                    raise ValueError(
                        "Evidence provided for variable which is not in the model"
                    )
                elif len(cpd.variables) > 1:
                    raise (
                        "Virtual evidence should be defined on individual variables. Maybe you are looking for soft evidence."
                    )
                elif self.get_cardinality(var) != cpd.get_cardinality([var])[var]:
                    raise ValueError(
                        "The number of states/cardinality for the evidence should be same as the number of states/cardinality of the variable in the model"
                    )

            for cpd in virtual_evidence:
                var = cpd.variables[0]
                new_var = "__" + var
                model.add_edge(var, new_var)
                values = compat_fns.get_compute_backend().vstack(
                    (cpd.values, 1 - cpd.values)
                )
                new_cpd = TabularCPD(
                    variable=new_var,
                    variable_card=2,
                    values=values,
                    evidence=[var],
                    evidence_card=[model.get_cardinality(var)],
                    state_names={new_var: [0, 1], var: cpd.state_names[var]},
                )
                model.add_cpds(new_cpd)
                evidence[new_var] = 0

        # Step 3: If missing_prob; include missing values in samples.
        if missing_prob is not None:
            if isinstance(missing_prob, list):
                for cpd in missing_prob:
                    if not isinstance(cpd, TabularCPD):
                        raise ValueError(
                            f"missing_prob must be a list of TabularCPD objects. Got {type(cpd)}"
                        )
            else:
                if isinstance(missing_prob, TabularCPD):
                    missing_prob = [missing_prob]
                else:
                    raise ValueError(
                        f"missing_prob should be TabularCPD. Got {type(missing_prob)}"
                    )

            for cpd in missing_prob:
                variable = cpd.variables[0]

                if not variable.endswith("*"):
                    raise ValueError(
                        f"Got {variable}. TabularCPD variable should end with * symbol to represent missingnness variable."
                    )

                if variable.split("*")[0] not in model.nodes:
                    raise ValueError(
                        f"Got {variable}. TabularCPD variable not in model nodes."
                    )

                if cpd.cardinality[0] != 2:
                    raise ValueError(
                        f"Got cardinality of variable = {cpd.cardinality[0]}. Tabular CPD variable should have 2 possible states : Missing (1) and Not Missing (0)"
                    )

                model.add_node(variable)

                if len(cpd.variables) > 1:
                    evidences = cpd.variables[1:]
                    for node in evidences:
                        if node not in model.nodes():
                            raise ValueError(
                                f"TabularCPD evidence {node} not in model nodes."
                            )
                        else:
                            model.add_edge(node, variable)

                model.add_cpds(cpd)

        # Step 4: If no evidence do a forward sampling
        if len(evidence) == 0:
            samples = BayesianModelSampling(model).forward_sample(
                size=n_samples,
                include_latents=include_latents,
                seed=seed,
                show_progress=show_progress,
                partial_samples=partial_samples,
            )

        # Step 5: If evidence; do a rejection sampling
        else:
            samples = BayesianModelSampling(model).rejection_sample(
                size=n_samples,
                evidence=[(k, v) for k, v in evidence.items()],
                include_latents=include_latents,
                seed=seed,
                show_progress=show_progress,
                partial_samples=partial_samples,
            )

        # Step 6: If missing_prob; perform masking
        if missing_prob:
            for cpd in missing_prob:
                variable = cpd.variables[0]
                if return_full:
                    samples[variable.split("*")[0] + "_full"] = samples.loc[
                        :, variable.split("*")[0]
                    ]

                samples.loc[samples[variable] == 1, variable.split("*")[0]] = np.nan
                samples.drop(columns=[variable], inplace=True)

        # Step 7: Postprocess and return
        if include_latents:
            return samples.astype("category")
        else:
            return (samples.loc[:, list(set(samples.columns) - self.latents)]).astype(
                "category"
            )

    def save(self, filename, filetype="bif"):
        """
        Writes the model to a file. Plese avoid using any special characters or
        spaces in variable or state names.

        Parameters
        ----------
        filename: str
            The path along with the filename where to write the file.

        filetype: str (default: bif)
            The format in which to write the model to file. Can be one of
            the following: bif, uai, xmlbif, xdsl.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model
        >>> alarm = get_example_model('alarm')
        >>> alarm.save('alarm.bif', filetype='bif')
        """
        supported_formats = {"bif", "uai", "xmlbif", "xdsl"}
        if filename.split(".")[-1].lower() in supported_formats:
            filetype = filename.split(".")[-1].lower()

        if filetype == "bif":
            from pgmpy.readwrite import BIFWriter

            writer = BIFWriter(self)
            writer.write_bif(filename=filename)

        elif filetype == "uai":
            from pgmpy.readwrite import UAIWriter

            writer = UAIWriter(self)
            writer.write_uai(filename=filename)

        elif filetype == "xmlbif":
            from pgmpy.readwrite import XMLBIFWriter

            writer = XMLBIFWriter(self)
            writer.write_xmlbif(filename=filename)

        elif filetype == "xdsl":
            from pgmpy.readwrite import XDSLWriter

            writer = XDSLWriter(self)
            writer.write_xdsl(filename=filename)

    @staticmethod
    def load(filename, filetype="bif", **kwargs):
        """
        Read the model from a file.

        Parameters
        ----------
        filename: str
            The path along with the filename where to read the file.

        filetype: str (default: bif)
            The format of the model file. Can be one of
            the following: bif, uai, xmlbif, xdsl.

        kwargs: kwargs
            Any additional arguments for the reader class or get_model method.
            Please refer the file format class for details.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model
        >>> alarm = get_example_model('alarm')
        >>> alarm.save('alarm.bif', filetype='bif')
        >>> alarm_model = DiscreteBayesianNetwork.load('alarm.bif', filetype='bif')
        """
        supported_formats = {"bif", "uai", "xmlbif", "xdsl"}
        if filename.split(".")[-1].lower() in supported_formats:
            filetype = filename.split(".")[-1].lower()

        if filetype == "bif":
            from pgmpy.readwrite import BIFReader

            if "n_jobs" in kwargs:
                n_jobs = kwargs["n_jobs"]
            else:
                n_jobs = -1

            if "state_name_type" in kwargs:
                state_name_type = kwargs["state_name_type"]
            else:
                state_name_type = str

            reader = BIFReader(path=filename, n_jobs=n_jobs)
            return reader.get_model(state_name_type=state_name_type)

        elif filetype == "uai":
            from pgmpy.readwrite import UAIReader

            reader = UAIReader(path=filename)
            return reader.get_model()

        elif filetype == "xmlbif":
            from pgmpy.readwrite import XMLBIFReader

            reader = XMLBIFReader(path=filename)
            return reader.get_model()

        elif filetype == "xdsl":
            from pgmpy.readwrite import XDSLReader

            reader = XDSLReader(path=filename)
            return reader.get_model()
