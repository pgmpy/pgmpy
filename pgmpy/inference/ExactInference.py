#!/usr/bin/env python3
import copy
import itertools
from functools import reduce

import networkx as nx
import numpy as np
from opt_einsum import contract
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.factors import factor_product
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import Inference
from pgmpy.inference.EliminationOrder import (
    MinFill,
    MinNeighbors,
    MinWeight,
    WeightedMinFill,
)
from pgmpy.models import (
    BayesianNetwork,
    DynamicBayesianNetwork,
    FactorGraph,
    JunctionTree,
)
from pgmpy.utils import compat_fns


class VariableElimination(Inference):
    def _get_working_factors(self, evidence):
        """
        Uses the evidence given to the query methods to modify the factors before running
        the variable elimination algorithm.

        Parameters
        ----------
        evidence: dict
            Dict of the form {variable: state}

        Returns
        -------
        dict: Modified working factors.
        """

        working_factors = {
            node: {(factor, None) for factor in self.factors[node]}
            for node in self.factors
        }

        # Dealing with evidence. Reducing factors over it before VE is run.
        if evidence:
            for evidence_var in evidence:
                for factor, origin in working_factors[evidence_var]:
                    factor_reduced = factor.reduce(
                        [(evidence_var, evidence[evidence_var])], inplace=False
                    )
                    for var in factor_reduced.scope():
                        working_factors[var].remove((factor, origin))
                        working_factors[var].add((factor_reduced, evidence_var))
                del working_factors[evidence_var]
        return working_factors

    def _get_elimination_order(
        self, variables, evidence, elimination_order, show_progress=True
    ):
        """
        Deals with all elimination order parameters given to _variable_elimination method
        and returns a list of variables that are to be eliminated

        Parameters
        ----------
        elimination_order: str or list

        Returns
        -------
        list: A list of variables names in the order they need to be eliminated.
        """
        to_eliminate = (
            set(self.variables)
            - set(variables)
            - set(evidence.keys() if evidence else [])
        )

        # Step 1: If elimination_order is a list, verify it's correct and return.
        # Step 1.1: Check that not of the `variables` and `evidence` is in the elimination_order.
        if hasattr(elimination_order, "__iter__") and (
            not isinstance(elimination_order, str)
        ):
            if any(
                var in elimination_order
                for var in set(variables).union(
                    set(evidence.keys() if evidence else [])
                )
            ):
                raise ValueError(
                    "Elimination order contains variables which are in"
                    " variables or evidence args"
                )
            # Step 1.2: Check if elimination_order has variables which are not in the model.
            elif any(var not in self.model.nodes() for var in elimination_order):
                elimination_order = list(
                    filter(lambda t: t in self.model.nodes(), elimination_order)
                )

            # Step 1.3: Check if the elimination_order has all the variables that need to be eliminated.
            elif to_eliminate != set(elimination_order):
                raise ValueError(
                    f"Elimination order doesn't contain all the variables"
                    f"which need to be eliminated. The variables which need to"
                    f"be eliminated are {to_eliminate}"
                )

            return elimination_order

        # Step 2: If elimination order is None or a Markov model, return a random order.
        elif (elimination_order is None) or (
            not isinstance(self.model, BayesianNetwork)
        ):
            return to_eliminate

        # Step 3: If elimination order is a str, compute the order using the specified heuristic.
        elif isinstance(elimination_order, str) and isinstance(
            self.model, BayesianNetwork
        ):
            heuristic_dict = {
                "weightedminfill": WeightedMinFill,
                "minneighbors": MinNeighbors,
                "minweight": MinWeight,
                "minfill": MinFill,
            }
            elimination_order = heuristic_dict[elimination_order.lower()](
                self.model
            ).get_elimination_order(nodes=to_eliminate, show_progress=show_progress)
            return elimination_order

    def _variable_elimination(
        self,
        variables,
        operation,
        evidence=None,
        elimination_order="MinFill",
        joint=True,
        show_progress=True,
    ):
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

        elimination_order: str or list (array-like)
            If str: Heuristic to use to find the elimination order.
            If array-like: The elimination order to use.
            If None: A random elimination order is used.
        """
        # Step 1: Deal with the input arguments.
        if isinstance(variables, str):
            raise TypeError("variables must be a list of strings")
        if isinstance(evidence, str):
            raise TypeError("evidence must be a list of strings")

        # Dealing with the case when variables are not provided.
        if not variables:
            all_factors = []
            for factor_li in self.factors.values():
                all_factors.extend(factor_li)
            if joint:
                return factor_product(*set(all_factors))
            else:
                return set(all_factors)

        # Step 2: Prepare data structures to run the algorithm.
        eliminated_variables = set()
        # Get working factors and elimination order
        working_factors = self._get_working_factors(evidence)
        elimination_order = self._get_elimination_order(
            variables, evidence, elimination_order, show_progress=show_progress
        )

        # Step 3: Run variable elimination
        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(elimination_order)
        else:
            pbar = elimination_order

        for var in pbar:
            if show_progress and config.SHOW_PROGRESS:
                pbar.set_description(f"Eliminating: {var}")
            # Removing all the factors containing the variables which are
            # eliminated (as all the factors should be considered only once)
            factors = [
                factor
                for factor, _ in working_factors[var]
                if not set(factor.variables).intersection(eliminated_variables)
            ]
            phi = factor_product(*factors)
            phi = getattr(phi, operation)([var], inplace=False)
            del working_factors[var]
            for variable in phi.variables:
                working_factors[variable].add((phi, var))
            eliminated_variables.add(var)

        # Step 4: Prepare variables to be returned.
        final_distribution = set()
        for node in working_factors:
            for factor, origin in working_factors[node]:
                if not set(factor.variables).intersection(eliminated_variables):
                    final_distribution.add((factor, origin))
        final_distribution = [factor for factor, _ in final_distribution]

        if joint:
            if isinstance(self.model, BayesianNetwork):
                return factor_product(*final_distribution).normalize(inplace=False)
            else:
                return factor_product(*final_distribution)
        else:
            query_var_factor = {}
            if isinstance(self.model, BayesianNetwork):
                for query_var in variables:
                    phi = factor_product(*final_distribution)
                    query_var_factor[query_var] = phi.marginalize(
                        list(set(variables) - set([query_var])), inplace=False
                    ).normalize(inplace=False)
            else:
                for query_var in variables:
                    phi = factor_product(*final_distribution)
                    query_var_factor[query_var] = phi.marginalize(
                        list(set(variables) - set([query_var])), inplace=False
                    )
            return query_var_factor

    def query(
        self,
        variables,
        evidence=None,
        virtual_evidence=None,
        elimination_order="greedy",
        joint=True,
        show_progress=True,
    ):
        """
        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        virtual_evidence: list (default:None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual
            evidences.

        elimination_order: str or list (default='greedy')
            Order in which to eliminate the variables in the algorithm. If list is provided,
            should contain all variables in the model except the ones in `variables`. str options
            are: `greedy`, `WeightedMinFill`, `MinNeighbors`, `MinWeight`, `MinFill`. Please
            refer https://pgmpy.org/exact_infer/ve.html#module-pgmpy.inference.EliminationOrder
            for details.

        joint: boolean (default: True)
            If True, returns a Joint Distribution over `variables`.
            If False, returns a dict of distributions over each of the `variables`.

        show_progress: boolean
            If True, shows a progress bar.

        Examples
        --------
        >>> from pgmpy.inference import VariableElimination
        >>> from pgmpy.models import BayesianNetwork
        >>> import numpy as np
        >>> import pandas as pd
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> inference = VariableElimination(model)
        >>> phi_query = inference.query(['A', 'B'])
        """
        evidence = evidence if evidence is not None else dict()

        # Step 1: Parameter Checks
        common_vars = set(evidence if evidence is not None else []).intersection(
            set(variables)
        )
        if common_vars:
            raise ValueError(
                f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}"
            )

        # Step 2: If virtual_evidence is provided, modify the network.
        if isinstance(self.model, BayesianNetwork) and (virtual_evidence is not None):
            self._virtual_evidence(virtual_evidence)
            virt_evidence = {"__" + cpd.variables[0]: 0 for cpd in virtual_evidence}
            return self.query(
                variables=variables,
                evidence={**evidence, **virt_evidence},
                virtual_evidence=None,
                elimination_order=elimination_order,
                joint=joint,
                show_progress=show_progress,
            )

        # Step 3: Prune the network based on variables and evidence.
        if isinstance(self.model, BayesianNetwork):
            model_reduced, evidence = self._prune_bayesian_model(variables, evidence)
            factors = model_reduced.cpds
        else:
            model_reduced = self.model
            factors = self.model.factors

        # Step 4: If elimination_order is greedy, do a tensor contraction approach
        #         else do the classic Variable Elimination.
        if elimination_order == "greedy":
            # Step 5.1: Compute the values array for factors after reducing them to provided
            #           evidence.
            evidence_vars = set(evidence)
            reduce_indexes = []
            reshape_indexes = []
            for phi in factors:
                indexes_to_reduce = [
                    phi.variables.index(var)
                    for var in set(phi.variables).intersection(evidence_vars)
                ]
                indexer = [slice(None)] * len(phi.variables)
                for index in indexes_to_reduce:
                    indexer[index] = phi.get_state_no(
                        phi.variables[index], evidence[phi.variables[index]]
                    )
                reduce_indexes.append(tuple(indexer))

            # Step 5.2: Prepare values and index arrays to do use in einsum
            if isinstance(self.model, JunctionTree):
                var_int_map = {
                    var: i
                    for i, var in enumerate(
                        set(itertools.chain(*model_reduced.nodes()))
                    )
                }
            else:
                var_int_map = {var: i for i, var in enumerate(model_reduced.nodes())}

            evidence_var_set = set(evidence.keys())
            einsum_expr = []

            if isinstance(self.model, BayesianNetwork):
                for index, phi in enumerate(factors):
                    if len(set(phi.variables) - evidence_var_set) > 0:
                        # if phi.variable not in evidence_var_set:
                        einsum_expr.append((phi.values[reduce_indexes[index]]))
                        einsum_expr.append(
                            [
                                var_int_map[var]
                                for var in phi.variables
                                if var not in evidence.keys()
                            ]
                        )
            else:
                for index, phi in enumerate(factors):
                    einsum_expr.append((phi.values[reduce_indexes[index]]))
                    einsum_expr.append(
                        [
                            var_int_map[var]
                            for var in phi.variables
                            if var not in evidence.keys()
                        ]
                    )

            result_values = contract(
                *einsum_expr, [var_int_map[var] for var in variables], optimize="greedy"
            )

            # Step 5.3: Prepare return values.
            result = DiscreteFactor(
                variables,
                result_values.shape,
                result_values,
                state_names={var: model_reduced.states[var] for var in variables},
            )
            if joint:
                if isinstance(
                    self.model, (BayesianNetwork, JunctionTree, DynamicBayesianNetwork)
                ):
                    return result.normalize(inplace=False)
                else:
                    return result
            else:
                result_dict = {}
                all_vars = set(variables)
                if isinstance(
                    self.model, (BayesianNetwork, JunctionTree, DynamicBayesianNetwork)
                ):
                    for var in variables:
                        result_dict[var] = result.marginalize(
                            all_vars - {var}, inplace=False
                        ).normalize(inplace=False)
                else:
                    for var in variables:
                        result_dict[var] = result.marginalize(
                            all_vars - {var}, inplace=False
                        )

                return result_dict

        else:
            # Step 5.1: Initialize data structures for the reduced bn.
            reduced_ve = VariableElimination(model_reduced)
            reduced_ve._initialize_structures()

            # Step 5.2: Do the actual variable elimination
            result = reduced_ve._variable_elimination(
                variables=variables,
                operation="marginalize",
                evidence=evidence,
                elimination_order=elimination_order,
                joint=joint,
                show_progress=show_progress,
            )

        return result

    def max_marginal(
        self,
        variables=None,
        evidence=None,
        elimination_order="MinFill",
        show_progress=True,
    ):
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
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.inference import VariableElimination
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> inference = VariableElimination(model)
        >>> phi_query = inference.max_marginal(['A', 'B'])
        """
        if not variables:
            variables = []

        common_vars = set(evidence if evidence is not None else []).intersection(
            set(variables if variables is not None else [])
        )
        if common_vars:
            raise ValueError(
                f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}"
            )

        if isinstance(self.model, BayesianNetwork):
            model_reduced, evidence = self._prune_bayesian_model(variables, evidence)
        else:
            model_reduced = self.model

        reduced_ve = VariableElimination(model_reduced)
        reduced_ve._initialize_structures()

        final_distribution = reduced_ve._variable_elimination(
            variables=variables,
            operation="maximize",
            evidence=evidence,
            elimination_order=elimination_order,
            show_progress=show_progress,
        )

        return compat_fns.max(final_distribution.values)

    def map_query(
        self,
        variables=None,
        evidence=None,
        virtual_evidence=None,
        elimination_order="MinFill",
        show_progress=True,
    ):
        """
        Computes the MAP Query over the variables given the evidence. Returns the
        highest probable state in the joint distribution of `variables`.

        Parameters
        ----------
        variables: list
            list of variables over which we want to compute the max-marginal.

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        virtual_evidence: list (default:None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual
            evidences.

        elimination_order: list
            order of variable eliminations (if nothing is provided) order is
            computed automatically

        show_progress: boolean
            If True, shows a progress bar.

        Examples
        --------
        >>> from pgmpy.inference import VariableElimination
        >>> from pgmpy.models import BayesianNetwork
        >>> import numpy as np
        >>> import pandas as pd
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> inference = VariableElimination(model)
        >>> phi_query = inference.map_query(['A', 'B'])
        """
        variables = [] if variables is None else variables
        evidence = evidence if evidence is not None else dict()
        common_vars = set(evidence if evidence is not None else []).intersection(
            variables
        )
        if common_vars:
            raise ValueError(
                f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}"
            )

        if isinstance(self.model, BayesianNetwork) and (virtual_evidence is not None):
            self._virtual_evidence(virtual_evidence)
            virt_evidence = {"__" + cpd.variables[0]: 0 for cpd in virtual_evidence}
            return self.map_query(
                variables=variables,
                evidence={**evidence, **virt_evidence},
                virtual_evidence=None,
                elimination_order=elimination_order,
                show_progress=show_progress,
            )

        if isinstance(self.model, BayesianNetwork):
            model_reduced, evidence = self._prune_bayesian_model(variables, evidence)
        else:
            model_reduced = self.model

        reduced_ve = VariableElimination(model_reduced)
        reduced_ve._initialize_structures()

        final_distribution = reduced_ve._variable_elimination(
            variables=variables,
            operation="marginalize",
            evidence=evidence,
            elimination_order=elimination_order,
            joint=True,
            show_progress=show_progress,
        )
        argmax = compat_fns.argmax(final_distribution.values)
        assignment = final_distribution.assignment([argmax])[0]

        map_query_results = {}
        for var_assignment in assignment:
            var, value = var_assignment
            map_query_results[var] = value

        return map_query_results

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
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.inference import VariableElimination
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> inference = VariableElimination(model)
        >>> inference.induced_graph(['C', 'D', 'A', 'B', 'E'])
        """
        self._initialize_structures()

        # If the elimination order does not contain the same variables as the model
        if set(elimination_order) != set(self.variables):
            raise ValueError(
                "Set of variables in elimination order"
                " different from variables in model"
            )

        eliminated_variables = set()
        working_factors = {
            node: [factor.scope() for factor in self.factors[node]]
            for node in self.factors
        }

        # The set of cliques that should be in the induced graph
        cliques = set()
        for factors in working_factors.values():
            for factor in factors:
                cliques.add(tuple(factor))

        # Removing all the factors containing the variables which are
        # eliminated (as all the factors should be considered only once)
        for var in elimination_order:
            factors = [
                factor
                for factor in working_factors[var]
                if not set(factor).intersection(eliminated_variables)
            ]
            phi = set(itertools.chain(*factors)).difference({var})
            cliques.add(tuple(phi))
            del working_factors[var]
            for variable in phi:
                working_factors[variable].append(list(phi))
            eliminated_variables.add(var)

        edges_comb = [
            itertools.combinations(c, 2) for c in filter(lambda x: len(x) > 1, cliques)
        ]
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
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.inference import VariableElimination
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> inference = VariableElimination(model)
        >>> inference.induced_width(['C', 'D', 'A', 'B', 'E'])
        3
        """
        induced_graph = self.induced_graph(elimination_order)
        return max((len(clique) for clique in nx.find_cliques(induced_graph))) - 1


class BeliefPropagation(Inference):
    """
    Class for performing inference using Belief Propagation method.

    Creates a Junction Tree or Clique Tree (JunctionTree class) for the input
    probabilistic graphical model and performs calibration of the junction tree
    so formed using belief propagation.

    Parameters
    ----------
    model: BayesianNetwork, MarkovNetwork, FactorGraph, JunctionTree
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

    def _update_beliefs(self, sending_clique, receiving_clique, operation):
        """
        This is belief-update method.

        Parameters
        ----------
        sending_clique: node (as the operation is on junction tree, node should be a tuple)
            Node sending the message
        receiving_clique: node (as the operation is on junction tree, node should be a tuple)
            Node receiving the message
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.

        Takes belief of one clique and uses it to update the belief of the
        neighboring ones.
        """
        sepset = frozenset(sending_clique).intersection(frozenset(receiving_clique))
        sepset_key = frozenset((sending_clique, receiving_clique))

        # \sigma_{i \rightarrow j} = \sum_{C_i - S_{i, j}} \beta_i
        # marginalize the clique over the sepset
        sigma = getattr(self.clique_beliefs[sending_clique], operation)(
            list(frozenset(sending_clique) - sepset), inplace=False
        )

        # \beta_j = \beta_j * \frac{\sigma_{i \rightarrow j}}{\mu_{i, j}}
        self.clique_beliefs[receiving_clique] *= (
            sigma / self.sepset_beliefs[sepset_key]
            if self.sepset_beliefs[sepset_key]
            else sigma
        )

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
            else if operation == maximize, it checks whether the junction tree is max calibrated or not

        Formally, at convergence or at calibration this condition would be satisfied for

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
            if (
                edge[0] not in self.clique_beliefs
                or edge[1] not in self.clique_beliefs
                or sepset_key not in self.sepset_beliefs
            ):
                return False

            marginal_1 = getattr(self.clique_beliefs[edge[0]], operation)(
                list(frozenset(edge[0]) - sepset), inplace=False
            )
            marginal_2 = getattr(self.clique_beliefs[edge[1]], operation)(
                list(frozenset(edge[1]) - sepset), inplace=False
            )
            if (
                marginal_1 != marginal_2
                or marginal_1 != self.sepset_beliefs[sepset_key]
            ):
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
        self.clique_beliefs = {
            clique: self.junction_tree.get_factors(clique)
            for clique in self.junction_tree.nodes()
        }
        self.sepset_beliefs = {
            frozenset(edge): None for edge in self.junction_tree.edges()
        }

        for clique in self.junction_tree.nodes():
            if not self._is_converged(operation=operation):
                neighbors = self.junction_tree.neighbors(clique)
                # update root's belief using neighbor clique's beliefs
                # upward pass
                for neighbor_clique in neighbors:
                    self._update_beliefs(neighbor_clique, clique, operation=operation)
                bfs_edges = nx.algorithms.breadth_first_search.bfs_edges(
                    self.junction_tree, clique
                )
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
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.inference import BeliefPropagation
        >>> G = BayesianNetwork([('diff', 'grade'), ('intel', 'grade'),
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
        self._calibrate_junction_tree(operation="marginalize")

    def max_calibrate(self):
        """
        Max-calibration of the junction tree using belief propagation.

        Examples
        --------
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.inference import BeliefPropagation
        >>> G = BayesianNetwork([('diff', 'grade'), ('intel', 'grade'),
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
        self._calibrate_junction_tree(operation="maximize")

    def _query(
        self, variables, operation, evidence=None, joint=True, show_progress=True
    ):
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
        >>> from pgmpy.models import BayesianNetwork
        >>> import numpy as np
        >>> import pandas as pd
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
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
            nodes_with_query_variables.update(
                filter(lambda x: var in x, self.junction_tree.nodes())
            )
        subtree_nodes = nodes_with_query_variables

        # Conversion of set to tuple just for indexing
        nodes_with_query_variables = tuple(nodes_with_query_variables)
        # As junction tree is a tree, that means that there would be only path between any two nodes in the tree
        # thus we can just take the path between any two nodes; no matter there order is
        for i in range(len(nodes_with_query_variables) - 1):
            subtree_nodes.update(
                nx.shortest_path(
                    self.junction_tree,
                    nodes_with_query_variables[i],
                    nodes_with_query_variables[i + 1],
                )
            )
        subtree_undirected_graph = self.junction_tree.subgraph(subtree_nodes)
        # Converting subtree into a junction tree
        if len(subtree_nodes) == 1:
            subtree = JunctionTree()
            subtree.add_node(subtree_nodes.pop())
        else:
            subtree = JunctionTree(subtree_undirected_graph.edges())

        # Selecting a node is root node. Root node would be having only one neighbor
        if len(subtree.nodes()) == 1:
            root_node = list(subtree.nodes())[0]
        else:
            root_node = tuple(
                filter(lambda x: len(list(subtree.neighbors(x))) == 1, subtree.nodes())
            )[0]
        clique_potential_list = [self.clique_beliefs[root_node]]

        # For other nodes in the subtree compute the clique potentials as follows
        # As all the nodes are nothing but tuples so simple set(root_node) won't work at it would update the set with
        # all the elements of the tuple; instead use set([root_node]) as it would include only the tuple not the
        # internal elements within it.
        parent_nodes = set([root_node])
        nodes_traversed = set()
        while parent_nodes:
            parent_node = parent_nodes.pop()
            for child_node in set(subtree.neighbors(parent_node)) - nodes_traversed:
                clique_potential_list.append(
                    self.clique_beliefs[child_node]
                    / self.sepset_beliefs[frozenset([parent_node, child_node])]
                )
                parent_nodes.update([child_node])
            nodes_traversed.update([parent_node])

        # Add factors to the corresponding junction tree
        subtree.add_factors(*clique_potential_list)

        # Sum product variable elimination on the subtree
        variable_elimination = VariableElimination(subtree)
        if operation == "marginalize":
            return variable_elimination.query(
                variables=variables,
                evidence=evidence,
                joint=joint,
                show_progress=show_progress,
            )
        elif operation == "maximize":
            return variable_elimination.map_query(
                variables=variables, evidence=evidence, show_progress=show_progress
            )

    def query(
        self,
        variables,
        evidence=None,
        virtual_evidence=None,
        joint=True,
        show_progress=True,
    ):
        """
        Query method using belief propagation.

        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        virtual_evidence: list (default:None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual
            evidences.

        joint: boolean
            If True, returns a Joint Distribution over `variables`.
            If False, returns a dict of distributions over each of the `variables`.

        show_progress: boolean
            If True shows a progress bar.

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.inference import BeliefPropagation
        >>> bayesian_model = BayesianNetwork([('A', 'J'), ('R', 'J'), ('J', 'Q'),
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
        evidence = evidence if evidence is not None else dict()
        orig_model = self.model.copy()

        # Step 1: Parameter Checks
        common_vars = set(evidence if evidence is not None else []).intersection(
            set(variables)
        )
        if common_vars:
            raise ValueError(
                f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}"
            )

        # Step 2: If virtual_evidence is provided, modify model and evidence.
        if isinstance(self.model, BayesianNetwork) and (virtual_evidence is not None):
            self._virtual_evidence(virtual_evidence)
            virt_evidence = {"__" + cpd.variables[0]: 0 for cpd in virtual_evidence}
            return self.query(
                variables=variables,
                evidence={**evidence, **virt_evidence},
                virtual_evidence=None,
                joint=joint,
                show_progress=show_progress,
            )

        # Step 3: Do network pruning.
        if isinstance(self.model, BayesianNetwork):
            self.model, evidence = self._prune_bayesian_model(variables, evidence)
        self._initialize_structures()

        # Step 4: Run inference.
        result = self._query(
            variables=variables,
            operation="marginalize",
            evidence=evidence,
            joint=joint,
            show_progress=show_progress,
        )
        self.__init__(orig_model)

        if joint:
            return result.normalize(inplace=False)
        else:
            return result

    def map_query(
        self, variables=None, evidence=None, virtual_evidence=None, show_progress=True
    ):
        """
        MAP Query method using belief propagation. Returns the highest probable
        state in the joint distributon of `variables`.

        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability

        virtual_evidence: list (default:None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual
            evidences.

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        show_progress: boolean
            If True, shows a progress bar.

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.inference import BeliefPropagation
        >>> bayesian_model = BayesianNetwork([('A', 'J'), ('R', 'J'), ('J', 'Q'),
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
        variables = [] if variables is None else variables
        evidence = evidence if evidence is not None else dict()
        common_vars = set(evidence if evidence is not None else []).intersection(
            variables
        )

        if common_vars:
            raise ValueError(
                f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}"
            )

        # TODO:Check the note in docstring. Change that behavior to return the joint MAP
        if not variables:
            variables = list(self.model.nodes())

        # Make a copy of the original model and then replace self.model with it later.
        orig_model = self.model.copy()

        if isinstance(self.model, BayesianNetwork) and (virtual_evidence is not None):
            self._virtual_evidence(virtual_evidence)
            virt_evidence = {"__" + cpd.variables[0]: 0 for cpd in virtual_evidence}
            return self.map_query(
                variables=variables,
                evidence={**evidence, **virt_evidence},
                virtual_evidence=None,
                show_progress=show_progress,
            )

        if isinstance(self.model, BayesianNetwork):
            self.model, evidence = self._prune_bayesian_model(variables, evidence)
        self._initialize_structures()

        final_distribution = self._query(
            variables=variables,
            operation="marginalize",
            evidence=evidence,
            joint=True,
            show_progress=show_progress,
        )

        self.__init__(orig_model)

        # To handle the case when no argument is passed then
        # _variable_elimination returns a dict.
        argmax = compat_fns.argmax(final_distribution.values)
        assignment = final_distribution.assignment([argmax])[0]

        map_query_results = {}
        for var_assignment in assignment:
            var, value = var_assignment
            map_query_results[var] = value

        return map_query_results


class BeliefPropagationWithMessagePassing(Inference):
    """
    Class for performing efficient inference using Belief Propagation method on factor graphs with no loops.

    The message-passing algorithm recursively parses the factor graph to propagate the
    model's beliefs to infer the posterior distribution of the queried variable. The recursion
    stops when reaching an observed variable or a unobserved root/leaf variable.

    It does not work for loopy graphs.

    Parameters
    ----------
    model: FactorGraph
        Model on which to run the inference.

    References
    ----------
    Algorithm 2.1 in https://www.mbmlbook.com/LearningSkills_Testing_out_the_model.html
    by J Winn (Microsoft Research).
    """

    def __init__(self, model: FactorGraph, check_model=True):
        assert isinstance(
            model, FactorGraph
        ), "Model must be an instance of FactorGraph"
        if check_model:
            model.check_model()
        self.model = model

    class _RecursiveMessageSchedulingQuery(object):
        """
        Private class used in `BeliefPropagationWithMessagePassing.query()` to efficiently
        manage the message scheduling across the different queried variables, in a recursive way.

        Parameters
        ----------
        Same as in the query method.
        """

        def __init__(
            self,
            belief_propagation,
            variables,
            evidence,
            virtual_evidence,
            get_messages,
        ):
            self.bp = belief_propagation
            self.variables = variables
            self.evidence = evidence
            self.virtual_evidence = virtual_evidence
            self.all_messages = {} if get_messages else None

        def run(self):
            agg_res = {}
            for variable in self.variables:
                res = self.schedule_variable_node_messages(
                    variable,
                    from_factor=None,
                )
                agg_res[variable] = DiscreteFactor([variable], [len(res)], res)
            if self.all_messages is None:
                return agg_res
            else:
                return agg_res, self.all_messages

        def schedule_variable_node_messages(
            self,
            variable,
            from_factor,
        ):
            """
            Returns the message sent by the variable to the factor requesting it.
            For that, the variable requests the messages coming from its neighbouring
            factors, except the one making the request.

            Parameters
            ----------
            variable: str
                The variable node from which to compute the outgoing message
            from_factor: pgmpy.factors.discrete.DiscreteFactor or None.
                The factor requesting the message, as part of the recursion.
                None for the first time this function is called.
            """
            if self.evidence is not None and variable in self.evidence.keys():
                # Is an observed variable
                return self.bp.model.get_point_mass_message(
                    variable, self.evidence[variable]
                )

            virtual_messages = []
            if (
                self.virtual_evidence is not None
                and variable
                in self.bp._get_virtual_evidence_var_list(self.virtual_evidence)
            ):
                virtual_messages = [
                    cpd.values
                    for cpd in self.virtual_evidence
                    if cpd.variables[0] == variable
                ]

            incoming_factors = [
                factor
                for factor in list(self.bp.model.neighbors(variable))
                if factor != from_factor
            ]

            if len(incoming_factors) == 0:
                # Is an unobserved leaf variable
                return self.bp.calc_variable_node_message(
                    variable, [] + virtual_messages
                )
            else:
                # Else, get the incoming messages from all incoming factors
                incoming_messages = []
                for factor in incoming_factors:
                    incoming_message = self.schedule_factor_node_messages(
                        factor, variable
                    )

                    if self.all_messages is not None:
                        # Store the message if it's not already stored
                        factor_node_key = f"{factor.variables} -> {variable}"
                        if factor_node_key not in self.all_messages.keys():
                            self.all_messages[factor_node_key] = incoming_message

                    incoming_messages.append(incoming_message)
                return self.bp.calc_variable_node_message(
                    variable, incoming_messages + virtual_messages
                )

        def schedule_factor_node_messages(self, factor, from_variable):
            """
            Returns the message sent from the factor to the variable requesting it.
            For that, the factor requests the messages coming from its neighbouring
            variables, except the one making the request.

            Parameters
            ----------
            factor: pgmpy.factors.discrete.DiscreteFactor
                The factor from which we want to compute the outgoing message.
            from_variable: str
                The variable requesting the message, as part of the recursion.
            """
            assert from_variable is not None, "from_var must be specified"

            incoming_vars = [var for var in factor.variables if var != from_variable]
            if len(incoming_vars) == 0:
                # from_var is a root variable. The factor is its prior
                return self.bp.calc_factor_node_message(factor, [], from_variable)
            else:
                # Else, get the incoming messages from all incoming variables
                incoming_messages = []
                for var in incoming_vars:
                    incoming_messages.append(
                        self.schedule_variable_node_messages(var, factor)
                    )
                return self.bp.calc_factor_node_message(
                    factor, incoming_messages, from_variable
                )

    def query(
        self, variables, evidence=None, virtual_evidence=None, get_messages=False
    ):
        """
        Computes the posterior distributions for each of the queried variable,
        given the `evidence`, and the `virtual_evidence`. Optionally also returns
        the computed messages.

        Parameters
        ----------
        variables: list
            List of variables for which you want to compute the posterior.
        evidence: dict or None (default: None)
            A dict key, value pair as {var: state_of_var_observed}.
            None if no evidence.
        virtual_evidence: list or None (default: None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual
            evidences. Each virtual evidence becomes a virtual message that gets added to
            the list of computed messages incoming to the variable node.
            None if no virtual evidence.

        Returns
        -------
        If `get_messages` is False, returns a dict of the variables, posterior distributions
            pairs: {variable: pgmpy.factors.discrete.DiscreteFactor}.
        If `get_messages` is True, returns:
            1. A dict of the variables, posterior distributions pairs:
            {variable: pgmpy.factors.discrete.DiscreteFactor}
            2. A dict of all messages sent from a factor to a node:
            {"{pgmpy.factors.discrete.DiscreteFactor.variables} -> variable": np.array}.

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.inference import BeliefPropagation
        >>> factor_graph = FactorGraph()
        >>> factor_graph.add_nodes_from(["A", "B", "C", "D"])
        >>> phi1 = DiscreteFactor(["A"], [2], [0.4, 0.6])
        >>> phi2 = DiscreteFactor(
        ...     ["B", "A"], [3, 2], [[0.2, 0.05], [0.3, 0.15], [0.5, 0.8]]
        ... )
        >>> phi3 = DiscreteFactor(["C", "B"], [2, 3], [[0.4, 0.5, 0.1], [0.6, 0.5, 0.9]])
        >>> phi4 = DiscreteFactor(
        ...     ["D", "B"], [3, 3], [[0.1, 0.1, 0.2], [0.3, 0.2, 0.1], [0.6, 0.7, 0.7]]
        ... )
        >>> factor_graph.add_factors(phi1, phi2, phi3, phi4)
        >>> factor_graph.add_edges_from(
        ...     [
        ...         (phi1, "A"),
        ...         ("A", phi2),
        ...         (phi2, "B"),
        ...         ("B", phi3),
        ...         (phi3, "C"),
        ...         ("B", phi4),
        ...         (phi4, "D"),
        ...     ]
        ... )
        >>> belief_propagation = BeliefPropagation(factor_graph)
        >>> belief_propagation.query(variables=['B', 'C'],
        ...                          evidence={'D': 0},
        ...                          virtual_evidence=[TabularCPD(['A'], 2, [[0.3], [0.7]])])
        """
        common_vars = set(evidence if evidence is not None else []).intersection(
            set(variables)
        )
        if common_vars:
            raise ValueError(
                f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}"
            )

        # Can't have the same variables in both `evidence` and `virtual_evidence`
        if evidence is not None and virtual_evidence is not None:
            self._check_virtual_evidence(virtual_evidence)
            ve_names = self._get_virtual_evidence_var_list(virtual_evidence)
            common_vars = set(evidence).intersection(set(ve_names))
            if common_vars:
                raise ValueError(
                    f"Can't have the same variables in both `evidence` and `virtual_evidence`. Found in both: {common_vars}"
                )

        query = self._RecursiveMessageSchedulingQuery(
            self, variables, evidence, virtual_evidence, get_messages
        )
        return query.run()

    def calc_variable_node_message(self, variable, incoming_messages):
        """
        The outgoing message is the element wise product of all incoming messages

        If there are no incoming messages, returns a uniform message
        If there is only one incoming message, returns that message
        Otherwise, returns the product of all incoming messages

        Parameters
        ----------
        variable: str
            the variable node from which to compute the outgoing message
        incoming_messages: list
            list of messages coming to this variable node
        """
        if len(incoming_messages) == 0:
            return self.model.get_uniform_message(variable)
        elif len(incoming_messages) == 1:
            return incoming_messages[0]
        else:
            outgoing_message = reduce(np.multiply, incoming_messages)
        return outgoing_message / np.sum(outgoing_message)

    @staticmethod
    def calc_factor_node_message(factor, incoming_messages, target_var):
        """
        Returns the outgoing message for a factor node, which is the
        multiplication of the incoming messages with the factor function (CPT).

        The variables' order in the incoming messages list must match the
        variable's order in the CPT's dimensions

        Parameters
        ----------
        factor: str
            the factor node from which to compute the outgoing message
        incoming_messages: list
            list of messages coming to this factor node
        target_var: str
            the variable node to which the outgoing message is being sent to
        """
        cpt = factor.values

        assert (
            len(incoming_messages) == cpt.ndim - 1
        ), f"Error computing factor node message for {target_var}. The number of incoming messages must equal the card(CPT) - 1"

        if len(incoming_messages) == 0:
            return cpt

        # Ensure that the target var is on the CPT's 0th axis
        target_var_idx = factor.variables.index(target_var)
        if target_var_idx != 0:
            # Move target var to the 0th axis to allow the reduction
            cpt = np.moveaxis(cpt, target_var_idx, 0)

        # Invert incoming_messages, so that the first message corresponds to the last
        # dimension of the CPT
        incoming_messages = list(reversed(incoming_messages))

        # Reduce the CPT with the inverted list of incoming messages
        outgoing_message = reduce(
            lambda cpt_reduced, m: np.matmul(cpt_reduced, m), incoming_messages, cpt
        )
        # Normalise
        return outgoing_message / sum(outgoing_message)
