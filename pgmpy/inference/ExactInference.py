#!/usr/bin/env python3

import itertools
import numpy as np
import networkx as nx
from pgmpy.inference import Inference
from pgmpy.factors.Factor import factor_product
from networkx import graph_clique_number


class VariableElimination(Inference):
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
                    factor_reduced = factor.reduce('{evidence_var}_{state}'.format(evidence_var=evidence_var,
                                                                                   state=evidence[evidence_var]),
                                                   inplace=False)
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
            phi = getattr(phi, operation)(var, inplace=False)
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
            phi.marginalize(list(set(variables) - set([query_var])))
            query_var_factor[query_var] = phi.normalize(inplace=False)
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

    def map_query(self, variables=None, evidence=None, elimination_order=None):
        """
        Computes the MAP Query over the variables given the evidence.

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
        elimination_variables = set(self.variables) - set(evidence.keys()) if evidence else set()
        final_distribution = self._variable_elimination(elimination_variables, 'maximize',
                                                        evidence=evidence,
                                                        elimination_order=elimination_order)
        # To handle the case when no argument is passed then
        # _variable_elimination returns a dict.
        if isinstance(final_distribution, dict):
            final_distribution = final_distribution.values()
        distribution = factor_product(*final_distribution)
        argmax = np.argmax(distribution.values)
        assignment = distribution.assignment(argmax)[0]

        map_query_results = {}
        for var_assignment in assignment:
            var, value = var_assignment.split('_')
            map_query_results[var] = int(value)

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
        return graph_clique_number(induced_graph) - 1


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
        from pgmpy.models import JunctionTree

        super().__init__(model)

        if not isinstance(model, JunctionTree):
            self.junction_tree = model.to_junction_tree()
        else:
            self.junction_tree = model

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

    def calibrate(self):
        """
        Calibration using belief propagation in junction tree or clique tree.

        Uses Lauritzen-Spiegelhalter algorithm or belief-update message passing.

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors import TabularCPD
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

        Reference
        ---------
        Algorithm 10.3 Calibration using belief propagation in clique tree
        Probabilistic Graphical Models: Principles and Techniques
        Daphne Koller and Nir Friedman.
        """
        # Initialize clique beliefs as well as sepset beliefs
        self.clique_beliefs = {clique: self.junction_tree.get_factors(clique)
                               for clique in self.junction_tree.nodes()}
        self.sepset_beliefs = {frozenset(x[0]).intersection(frozenset(x[1])): None
                               for x in self.junction_tree.edges()}

        def _update_beliefs(sending_clique, recieving_clique):
            """
            This is belief-update method.

            Takes belief of one clique and uses it to update the belief of the
            neighboring ones.
            """
            sepset = frozenset(sending_clique).intersection(frozenset(recieving_clique))
            print(sending_clique, sepset, recieving_clique)

            # \sigma_{i \rightarrow j} = \sum_{C_i - S_{i, j}} \beta_i
            # marginalize the clique over the sepset
            sigma = self.clique_beliefs[sending_clique].marginalize(
                list(frozenset(sending_clique) - sepset), inplace=False)
            print(sigma)

            # \beta_j = \beta_j * \frac{\sigma_{i \rightarrow j}}{\mu_{i, j}}
            self.clique_beliefs[recieving_clique] *= (sigma / self.sepset_beliefs[sepset]
                                                      if self.sepset_beliefs[sepset] else sigma)
            print(self.clique_beliefs[recieving_clique])

            # \mu_{i, j} = \sigma_{i \rightarrow j}
            self.sepset_beliefs[sepset] = sigma
            print(self.sepset_beliefs[sepset])

        def _converged():
            """
            Checks whether the calibration has converged or not. At convergence
            the sepset belief would be precisely the sepset marginal.

            Formally, at convergence this condition would be satisified

            \sum_{C_i - S_{i, j}} \beta_i = \sum_{C_j - S_{i, j}} \beta_j = \mu_{i, j}
            """
            for edge in self.junction_tree.edges():
                sepset = frozenset(edge[0]).intersection(frozenset(edge[1]))
                marginal_1 = self.clique_beliefs[edge[0]].marginalize(list(frozenset(edge[0]) - sepset), inplace=False)
                marginal_2 = self.clique_beliefs[edge[1]].marginalize(list(frozenset(edge[1]) - sepset), inplace=False)
                if not np.allclose(marginal_1.values, marginal_2.values, rtol=1e-4):
                    return False

            return True

        for clique in self.junction_tree.nodes():
            if not _converged():
                neighbors = self.junction_tree.neighbors(clique)
                # update root's belief using nieighbor clique's beliefs
                # upward pass
                for neighbor_clique in neighbors:
                    _update_beliefs(neighbor_clique, clique)
                bfs_edges = nx.algorithms.breadth_first_search.bfs_edges(self.junction_tree, clique)
                # update the beliefs of all the nodes starting from the root to leaves using root's belief
                # downward pass
                for edge in bfs_edges:
                    _update_beliefs(edge[0], edge[1])
            else:
                break
