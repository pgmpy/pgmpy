#!/usr/bin/env python3
import copy
import itertools

import networkx as nx
import numpy as np
from pgmpy.extern.six.moves import filter, range

from pgmpy.extern.six import string_types
from pgmpy.factors import factor_product
from pgmpy.inference import Inference
from pgmpy.models import JunctionTree
from pgmpy.utils import StateNameDecorator


class VariableElimination(Inference):

    @StateNameDecorator(argument='evidence', return_val=None)
    def _variable_elimination(self, variables, operation, evidence=None, elimination_order=None):
        """
        Implementation of a generalized variable elimination.

        Parameters
        ----------
        variables: list, array-like
            variables that are not to be eliminated.
        operation: str ('marginalize' | 'maximize')
            The operation to do for eliminating the variable.
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence
        elimination_order: list, array-like
            list of variables representing the order in which they
            are to be eliminated. If None order is computed automatically.
        """
        if isinstance(variables, string_types):
            raise TypeError("variables must be a list of strings")
        if isinstance(evidence, string_types):
            raise TypeError("evidence must be a list of strings")

        # Dealing with the case when variables is not provided.
        if not variables:
            all_factors = []
            for factor_li in self.factors.values():
                all_factors.extend(factor_li)
            return set(all_factors)

        eliminated_variables = set()
        working_factors = {node: {factor for factor in self.factors[node]}
                           for node in self.factors}

        # Dealing with evidence. Reducing factors over it before VE is run.
        if evidence:
            for evidence_var in evidence:
                for factor in working_factors[evidence_var]:
                    factor_reduced = factor.reduce([(evidence_var, evidence[evidence_var])], inplace=False)
                    for var in factor_reduced.scope():
                        working_factors[var].remove(factor)
                        working_factors[var].add(factor_reduced)
                del working_factors[evidence_var]

        # TODO: Modify it to find the optimal elimination order
        if not elimination_order:
            elimination_order = list(set(self.variables) -
                                     set(variables) -
                                     set(evidence.keys() if evidence else []))

        elif any(var in elimination_order for var in
                 set(variables).union(set(evidence.keys() if evidence else []))):
            raise ValueError("Elimination order contains variables which are in"
                             " variables or evidence args")

        for var in elimination_order:
            # Removing all the factors containing the variables which are
            # eliminated (as all the factors should be considered only once)
            factors = [factor for factor in working_factors[var]
                       if not set(factor.variables).intersection(eliminated_variables)]
            phi = factor_product(*factors)
            phi = getattr(phi, operation)([var], inplace=False)
            del working_factors[var]
            for variable in phi.variables:
                working_factors[variable].add(phi)
            eliminated_variables.add(var)

        final_distribution = set()
        for node in working_factors:
            factors = working_factors[node]
            for factor in factors:
                if not set(factor.variables).intersection(eliminated_variables):
                    final_distribution.add(factor)

        query_var_factor = {}
        for query_var in variables:
            phi = factor_product(*final_distribution)
            query_var_factor[query_var] = phi.marginalize(list(set(variables) -
                                                               set([query_var])),
                                                          inplace=False).normalize(inplace=False)
        return query_var_factor

    def query(self, variables, evidence=None, elimination_order=None):
        """
        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence
        elimination_order: list
            order of variable eliminations (if nothing is provided) order is
            computed automatically

        Examples
        --------
        >>> from pgmpy.inference import VariableElimination
        >>> from pgmpy.models import BayesianModel
        >>> import numpy as np
        >>> import pandas as pd
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> inference = VariableElimination(model)
        >>> phi_query = inference.query(['A', 'B'])
        """
        return self._variable_elimination(variables, 'marginalize',
                                          evidence=evidence, elimination_order=elimination_order)

    def max_marginal(self, variables=None, evidence=None, elimination_order=None):
        """
        Computes the max-marginal over the variables given the evidence.

        Parameters
        ----------
        variables: list
            list of variables over which we want to compute the max-marginal.
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence
        elimination_order: list
            order of variable eliminations (if nothing is provided) order is
            computed automatically

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.inference import VariableElimination
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> inference = VariableElimination(model)
        >>> phi_query = inference.max_marginal(['A', 'B'])
        """
        if not variables:
            variables = []
        final_distribution = self._variable_elimination(variables, 'maximize',
                                                        evidence=evidence,
                                                        elimination_order=elimination_order)

        # To handle the case when no argument is passed then
        # _variable_elimination returns a dict.
        if isinstance(final_distribution, dict):
            final_distribution = final_distribution.values()
        return np.max(factor_product(*final_distribution).values)

    @StateNameDecorator(argument=None, return_val=True)
    def map_query(self, variables=None, evidence=None, elimination_order=None):
        """
        Computes the MAP Query over the variables given the evidence.

        Note: When multiple variables are passed, it returns the map_query for each
        of them individually.

        Parameters
        ----------
        variables: list
            list of variables over which we want to compute the max-marginal.
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence
        elimination_order: list
            order of variable eliminations (if nothing is provided) order is
            computed automatically

        Examples
        --------
        >>> from pgmpy.inference import VariableElimination
        >>> from pgmpy.models import BayesianModel
        >>> import numpy as np
        >>> import pandas as pd
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> inference = VariableElimination(model)
        >>> phi_query = inference.map_query(['A', 'B'])
        """
        # TODO:Check the note in docstring. Change that behavior to return the joint MAP
        final_distribution = self._variable_elimination(variables, 'marginalize',
                                                        evidence=evidence,
                                                        elimination_order=elimination_order)
        # To handle the case when no argument is passed then
        # _variable_elimination returns a dict.
        if isinstance(final_distribution, dict):
            final_distribution = final_distribution.values()
        distribution = factor_product(*final_distribution)
        argmax = np.argmax(distribution.values)
        assignment = distribution.assignment([argmax])[0]

        map_query_results = {}
        for var_assignment in assignment:
            var, value = var_assignment
            map_query_results[var] = value

        if not variables:
            return map_query_results
        else:
            return_dict = {}
            for var in variables:
                return_dict[var] = map_query_results[var]
            return return_dict

    def induced_graph(self, elimination_order):
        """
        Returns the induced graph formed by running Variable Elimination on the network.

        Parameters
        ----------
        elimination_order: list, array like
            List of variables in the order in which they are to be eliminated.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.inference import VariableElimination
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> inference = VariableElimination(model)
        >>> inference.induced_graph(['C', 'D', 'A', 'B', 'E'])
        <networkx.classes.graph.Graph at 0x7f34ac8c5160>
        """
        # If the elimination order does not contain the same variables as the model
        if set(elimination_order) != set(self.variables):
            raise ValueError("Set of variables in elimination order"
                             " different from variables in model")

        eliminated_variables = set()
        working_factors = {node: [factor.scope() for factor in self.factors[node]]
                           for node in self.factors}

        # The set of cliques that should be in the induced graph
        cliques = set()
        for factors in working_factors.values():
            for factor in factors:
                cliques.add(tuple(factor))

        # Removing all the factors containing the variables which are
        # eliminated (as all the factors should be considered only once)
        for var in elimination_order:
            factors = [factor for factor in working_factors[var]
                       if not set(factor).intersection(eliminated_variables)]
            phi = set(itertools.chain(*factors)).difference({var})
            cliques.add(tuple(phi))
            del working_factors[var]
            for variable in phi:
                working_factors[variable].append(list(phi))
            eliminated_variables.add(var)

        edges_comb = [itertools.combinations(c, 2)
                      for c in filter(lambda x: len(x) > 1, cliques)]
        return nx.Graph(itertools.chain(*edges_comb))

    def induced_width(self, elimination_order):
        """
        Returns the width (integer) of the induced graph formed by running Variable Elimination on the network.
        The width is the defined as the number of nodes in the largest clique in the graph minus 1.

        Parameters
        ----------
        elimination_order: list, array like
            List of variables in the order in which they are to be eliminated.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.inference import VariableElimination
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> inference = VariableElimination(model)
        >>> inference.induced_width(['C', 'D', 'A', 'B', 'E'])
        3
        """
        induced_graph = self.induced_graph(elimination_order)
        return nx.graph_clique_number(induced_graph) - 1


class BeliefPropagation(Inference):
    """
    Class for performing inference using Belief Propagation method.

    Creates a Junction Tree or Clique Tree (JunctionTree class) for the input
    probabilistic graphical model and performs calibration of the junction tree
    so formed using belief propagation.

    Parameters
    ----------
    model: BayesianModel, MarkovModel, FactorGraph, JunctionTree
        model for which inference is to performed
    """

    def __init__(self, model):
        super(BeliefPropagation, self).__init__(model)

        if not isinstance(model, JunctionTree):
            self.junction_tree = model.to_junction_tree()
        else:
            self.junction_tree = copy.deepcopy(model)

        self.clique_beliefs = {}
        self.sepset_beliefs = {}

    def get_cliques(self):
        """
        Returns cliques used for belief propagation.
        """
        return self.junction_tree.nodes()

    def get_clique_beliefs(self):
        """
        Returns clique beliefs. Should be called after the clique tree (or
        junction tree) is calibrated.
        """
        return self.clique_beliefs

    def get_sepset_beliefs(self):
        """
        Returns sepset beliefs. Should be called after clique tree (or junction
        tree) is calibrated.
        """
        return self.sepset_beliefs

    def _update_beliefs(self, sending_clique, recieving_clique, operation):
        """
        This is belief-update method.

        Parameters
        ----------
        sending_clique: node (as the operation is on junction tree, node should be a tuple)
            Node sending the message
        recieving_clique: node (as the operation is on junction tree, node should be a tuple)
            Node recieving the message
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.

        Takes belief of one clique and uses it to update the belief of the
        neighboring ones.
        """
        sepset = frozenset(sending_clique).intersection(frozenset(recieving_clique))
        sepset_key = frozenset((sending_clique, recieving_clique))

        # \sigma_{i \rightarrow j} = \sum_{C_i - S_{i, j}} \beta_i
        # marginalize the clique over the sepset
        sigma = getattr(self.clique_beliefs[sending_clique], operation)(list(frozenset(sending_clique) - sepset),
                                                                        inplace=False)

        # \beta_j = \beta_j * \frac{\sigma_{i \rightarrow j}}{\mu_{i, j}}
        self.clique_beliefs[recieving_clique] *= (sigma / self.sepset_beliefs[sepset_key]
                                                  if self.sepset_beliefs[sepset_key] else sigma)

        # \mu_{i, j} = \sigma_{i \rightarrow j}
        self.sepset_beliefs[sepset_key] = sigma

    def _is_converged(self, operation):
        """
        Checks whether the calibration has converged or not. At convergence
        the sepset belief would be precisely the sepset marginal.

        Parameters
        ----------
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.
            if operation == marginalize, it checks whether the junction tree is calibrated or not
            else if operation == maximize, it checks whether the juction tree is max calibrated or not

        Formally, at convergence or at calibration this condition would be satisified for

        .. math:: \sum_{C_i - S_{i, j}} \beta_i = \sum_{C_j - S_{i, j}} \beta_j = \mu_{i, j}

        and at max calibration this condition would be satisfied

        .. math:: \max_{C_i - S_{i, j}} \beta_i = \max_{C_j - S_{i, j}} \beta_j = \mu_{i, j}
        """
        # If no clique belief, then the clique tree is not calibrated
        if not self.clique_beliefs:
            return False

        for edge in self.junction_tree.edges():
            sepset = frozenset(edge[0]).intersection(frozenset(edge[1]))
            sepset_key = frozenset(edge)
            if (edge[0] not in self.clique_beliefs or edge[1] not in self.clique_beliefs or
                    sepset_key not in self.sepset_beliefs):
                return False

            marginal_1 = getattr(self.clique_beliefs[edge[0]], operation)(list(frozenset(edge[0]) - sepset),
                                                                          inplace=False)
            marginal_2 = getattr(self.clique_beliefs[edge[1]], operation)(list(frozenset(edge[1]) - sepset),
                                                                          inplace=False)
            if marginal_1 != marginal_2 or marginal_1 != self.sepset_beliefs[sepset_key]:
                return False
        return True

    def _calibrate_junction_tree(self, operation):
        """
        Generalized calibration of junction tree or clique using belief propagation. This method can be used for both
        calibrating as well as max-calibrating.
        Uses Lauritzen-Spiegelhalter algorithm or belief-update message passing.

        Parameters
        ----------
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.

        Reference
        ---------
        Algorithm 10.3 Calibration using belief propagation in clique tree
        Probabilistic Graphical Models: Principles and Techniques
        Daphne Koller and Nir Friedman.
        """
        # Initialize clique beliefs as well as sepset beliefs
        self.clique_beliefs = {clique: self.junction_tree.get_factors(clique)
                               for clique in self.junction_tree.nodes()}
        self.sepset_beliefs = {frozenset(edge): None for edge in self.junction_tree.edges()}

        for clique in self.junction_tree.nodes():
            if not self._is_converged(operation=operation):
                neighbors = self.junction_tree.neighbors(clique)
                # update root's belief using nieighbor clique's beliefs
                # upward pass
                for neighbor_clique in neighbors:
                    self._update_beliefs(neighbor_clique, clique, operation=operation)
                bfs_edges = nx.algorithms.breadth_first_search.bfs_edges(self.junction_tree, clique)
                # update the beliefs of all the nodes starting from the root to leaves using root's belief
                # downward pass
                for edge in bfs_edges:
                    self._update_beliefs(edge[0], edge[1], operation=operation)
            else:
                break

    def calibrate(self):
        """
        Calibration using belief propagation in junction tree or clique tree.

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.inference import BeliefPropagation
        >>> G = BayesianModel([('diff', 'grade'), ('intel', 'grade'),
        ...                    ('intel', 'SAT'), ('grade', 'letter')])
        >>> diff_cpd = TabularCPD('diff', 2, [[0.2], [0.8]])
        >>> intel_cpd = TabularCPD('intel', 3, [[0.5], [0.3], [0.2]])
        >>> grade_cpd = TabularCPD('grade', 3,
        ...                        [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...                         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...                         [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
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
        >>> bp = BeliefPropagation(G)
        >>> bp.calibrate()
        """
        self._calibrate_junction_tree(operation='marginalize')

    def max_calibrate(self):
        """
        Max-calibration of the junction tree using belief propagation.

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.inference import BeliefPropagation
        >>> G = BayesianModel([('diff', 'grade'), ('intel', 'grade'),
        ...                    ('intel', 'SAT'), ('grade', 'letter')])
        >>> diff_cpd = TabularCPD('diff', 2, [[0.2], [0.8]])
        >>> intel_cpd = TabularCPD('intel', 3, [[0.5], [0.3], [0.2]])
        >>> grade_cpd = TabularCPD('grade', 3,
        ...                        [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...                         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...                         [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
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
        >>> bp = BeliefPropagation(G)
        >>> bp.max_calibrate()
        """
        self._calibrate_junction_tree(operation='maximize')

    def _query(self, variables, operation, evidence=None):
        """
        This is a generalized query method that can be used for both query and map query.

        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        Examples
        --------
        >>> from pgmpy.inference import BeliefPropagation
        >>> from pgmpy.models import BayesianModel
        >>> import numpy as np
        >>> import pandas as pd
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> inference = BeliefPropagation(model)
        >>> phi_query = inference.query(['A', 'B'])

        References
        ----------
        Algorithm 10.4 Out-of-clique inference in clique tree
        Probabilistic Graphical Models: Principles and Techniques Daphne Koller and Nir Friedman.
        """

        is_calibrated = self._is_converged(operation=operation)
        # Calibrate the junction tree if not calibrated
        if not is_calibrated:
            self.calibrate()

        if not isinstance(variables, (list, tuple, set)):
            query_variables = [variables]
        else:
            query_variables = list(variables)
        query_variables.extend(evidence.keys() if evidence else [])

        # Find a tree T' such that query_variables are a subset of scope(T')
        nodes_with_query_variables = set()
        for var in query_variables:
            nodes_with_query_variables.update(filter(lambda x: var in x, self.junction_tree.nodes()))
        subtree_nodes = nodes_with_query_variables

        # Conversion of set to tuple just for indexing
        nodes_with_query_variables = tuple(nodes_with_query_variables)
        # As junction tree is a tree, that means that there would be only path between any two nodes in the tree
        # thus we can just take the path between any two nodes; no matter there order is
        for i in range(len(nodes_with_query_variables) - 1):
            subtree_nodes.update(nx.shortest_path(self.junction_tree, nodes_with_query_variables[i],
                                                  nodes_with_query_variables[i + 1]))
        subtree_undirected_graph = self.junction_tree.subgraph(subtree_nodes)
        # Converting subtree into a junction tree
        if len(subtree_nodes) == 1:
            subtree = JunctionTree()
            subtree.add_node(subtree_nodes.pop())
        else:
            subtree = JunctionTree(subtree_undirected_graph.edges())

        # Selecting a node is root node. Root node would be having only one neighbor
        if len(subtree.nodes()) == 1:
            root_node = subtree.nodes()[0]
        else:
            root_node = tuple(filter(lambda x: len(subtree.neighbors(x)) == 1, subtree.nodes()))[0]
        clique_potential_list = [self.clique_beliefs[root_node]]

        # For other nodes in the subtree compute the clique potentials as follows
        # As all the nodes are nothing but tuples so simple set(root_node) won't work at it would update the set with'
        # all the elements of the tuple; instead use set([root_node]) as it would include only the tuple not the
        # internal elements within it.
        parent_nodes = set([root_node])
        nodes_traversed = set()
        while parent_nodes:
            parent_node = parent_nodes.pop()
            for child_node in set(subtree.neighbors(parent_node)) - nodes_traversed:
                clique_potential_list.append(self.clique_beliefs[child_node] /
                                             self.sepset_beliefs[frozenset([parent_node, child_node])])
                parent_nodes.update([child_node])
            nodes_traversed.update([parent_node])

        # Add factors to the corresponding junction tree
        subtree.add_factors(*clique_potential_list)

        # Sum product variable elimination on the subtree
        variable_elimination = VariableElimination(subtree)
        if operation == 'marginalize':
            return variable_elimination.query(variables=variables, evidence=evidence)
        elif operation == 'maximize':
            return variable_elimination.map_query(variables=variables, evidence=evidence)

    def query(self, variables, evidence=None):
        """
        Query method using belief propagation.

        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.inference import BeliefPropagation
        >>> bayesian_model = BayesianModel([('A', 'J'), ('R', 'J'), ('J', 'Q'),
        ...                                 ('J', 'L'), ('G', 'L')])
        >>> cpd_a = TabularCPD('A', 2, [[0.2], [0.8]])
        >>> cpd_r = TabularCPD('R', 2, [[0.4], [0.6]])
        >>> cpd_j = TabularCPD('J', 2,
        ...                    [[0.9, 0.6, 0.7, 0.1],
        ...                     [0.1, 0.4, 0.3, 0.9]],
        ...                    ['R', 'A'], [2, 2])
        >>> cpd_q = TabularCPD('Q', 2,
        ...                    [[0.9, 0.2],
        ...                     [0.1, 0.8]],
        ...                    ['J'], [2])
        >>> cpd_l = TabularCPD('L', 2,
        ...                    [[0.9, 0.45, 0.8, 0.1],
        ...                     [0.1, 0.55, 0.2, 0.9]],
        ...                    ['G', 'J'], [2, 2])
        >>> cpd_g = TabularCPD('G', 2, [[0.6], [0.4]])
        >>> bayesian_model.add_cpds(cpd_a, cpd_r, cpd_j, cpd_q, cpd_l, cpd_g)
        >>> belief_propagation = BeliefPropagation(bayesian_model)
        >>> belief_propagation.query(variables=['J', 'Q'],
        ...                          evidence={'A': 0, 'R': 0, 'G': 0, 'L': 1})
        """
        return self._query(variables=variables, operation='marginalize', evidence=evidence)

    def map_query(self, variables=None, evidence=None):
        """
        MAP Query method using belief propagation.

        Note: When multiple variables are passed, it returns the map_query for each
        of them individually.

        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.inference import BeliefPropagation
        >>> bayesian_model = BayesianModel([('A', 'J'), ('R', 'J'), ('J', 'Q'),
        ...                                 ('J', 'L'), ('G', 'L')])
        >>> cpd_a = TabularCPD('A', 2, [[0.2], [0.8]])
        >>> cpd_r = TabularCPD('R', 2, [[0.4], [0.6]])
        >>> cpd_j = TabularCPD('J', 2,
        ...                    [[0.9, 0.6, 0.7, 0.1],
        ...                     [0.1, 0.4, 0.3, 0.9]],
        ...                    ['R', 'A'], [2, 2])
        >>> cpd_q = TabularCPD('Q', 2,
        ...                    [[0.9, 0.2],
        ...                     [0.1, 0.8]],
        ...                    ['J'], [2])
        >>> cpd_l = TabularCPD('L', 2,
        ...                    [[0.9, 0.45, 0.8, 0.1],
        ...                     [0.1, 0.55, 0.2, 0.9]],
        ...                    ['G', 'J'], [2, 2])
        >>> cpd_g = TabularCPD('G', 2, [[0.6], [0.4]])
        >>> bayesian_model.add_cpds(cpd_a, cpd_r, cpd_j, cpd_q, cpd_l, cpd_g)
        >>> belief_propagation = BeliefPropagation(bayesian_model)
        >>> belief_propagation.map_query(variables=['J', 'Q'],
        ...                              evidence={'A': 0, 'R': 0, 'G': 0, 'L': 1})
        """
        # TODO:Check the note in docstring. Change that behavior to return the joint MAP
        if not variables:
            variables = set(self.variables)

        final_distribution = self._query(variables=variables, operation='marginalize', evidence=evidence)

        # To handle the case when no argument is passed then
        # _variable_elimination returns a dict.
        if isinstance(final_distribution, dict):
            final_distribution = final_distribution.values()
        distribution = factor_product(*final_distribution)
        argmax = np.argmax(distribution.values)
        assignment = distribution.assignment([argmax])[0]

        map_query_results = {}
        for var_assignment in assignment:
            var, value = var_assignment
            map_query_results[var] = value

        if not variables:
            return map_query_results
        else:
            return_dict = {}
            for var in variables:
                return_dict[var] = map_query_results[var]
            return return_dict
