from pgmpy.inference import Inference
from pgmpy.models import MarkovModel
from pgmpy.factors import Factor
import numpy as np
import itertools as it
import copy


class Mplp(Inference):
    """
    Class for performing approximate inference using Max-Product Linear Programming method.

    We derive message passing updates that result in monotone decrease of the dual of the
    MAP LP Relaxation.

    Parameters
    ----------
    model: MarkovModel for which inference is to be performed.
    Examples
    --------
    >>> from pgmpy.models import MarkovModel
    >>> from pgmpy.factors import Factor
    >>> import numpy as np
    >>> student = MarkovModel()
    >>> student.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('E', 'F')])
    >>> factor_a = Factor(['A'], cardinality=[2], value=np.array([0.54577, 1.8323]))
    >>> factor_b = Factor(['B'], cardinality=[2], value=np.array([0.93894, 1.065]))
    >>> factor_c = Factor(['C'], cardinality=[2], value=np.array([0.89205, 1.121]))
    >>> factor_d = Factor(['D'], cardinality=[2], value=np.array([0.56292, 1.7765]))
    >>> factor_e = Factor(['E'], cardinality=[2], value=np.array([0.47117, 2.1224]))
    >>> factor_f = Factor(['F'], cardinality=[2], value=np.array([1.5093, 0.66257]))
    >>> factor_a_b = Factor(['A', 'B'], cardinality=[2, 2], value=np.array([1.3207, 0.75717, 0.75717, 1.3207]))
    >>> factor_b_c = Factor(['B', 'C'], cardinality=[2, 2], value=np.array([0.00024189, 4134.2, 4134.2, 0.00024189]))
    >>> factor_c_d = Factor(['C', 'D'], cardinality=[2, 2], value=np.array([0.0043227, 231.34, 231.34, 0.0043227]))
    >>> factor_d_e = Factor(['E', 'F'], cardinality=[2, 2], value=np.array([31.228, 0.032023, 0.032023, 31.228]))
    >>> student.add_factors(factor_a, factor_b, factor_c, factor_d, factor_e, factor_f, factor_a_b,
    ...    factor_b_c, factor_c_d, factor_d_e)
    >>> mplp = Mplp(student)
    """
    def __init__(self, model):
        if not isinstance(model, MarkovModel):
            raise TypeError('Only MarkovModel is supported')

        super().__init__(model)
        self.model = model

        # S = \{c \cap c^{'} : c, c^{'} \in C, c \cap c^{'} \neq \emptyset\}
        self.intersection_set_variables = set()
        # We generate the Intersections of all the pairwise edges taken one at a time to form S
        for edge_pair in it.combinations(model.edges(), 2):
            self.intersection_set_variables.add(frozenset(edge_pair[0]) & frozenset(edge_pair[1]))

        # The corresponding optimization problem = \min_{\delta}{L(\delta)}
        # Where L(\delta) = \sum_{i \in V}{max_{x_i}(Objective[nodes])} + \sum_{f /in F}{max_{x_f}(Objective[factors])
        # Objective[nodes] = \theta_i(x_i) + \sum_{f \mid i \in f}{\delta_{fi}(x_i)}
        # Objective[factors] = \theta_f(x_f) - \sum_{i \in f}{\delta_{fi}(x_i)}
        # In a way Objective stores the corresponding optimization problem for all the nodes and the factors.

        # Form Objective and cluster_set in the form of a dictionary.
        self.objective = {}
        self.cluster_set = {}
        for factor in model.get_factors():
            scope = tuple(factor.scope())
            self.objective[scope] = factor
            # For every factor consisting of more that a single node, we initialize a cluster.
            if len(scope) > 1:
                self.cluster_set[scope] = self.Cluster(self.intersection_set_variables, factor)

        # L(\delta)
        self.L = sum([max(self.objective[obj].values) for obj in self.objective])

        # Best integral value of the primal objective is stored here
        self.best_int_objective = 0

        # Assignment of the nodes that results in the "maximum" integral value of the primal objective
        self.best_assignment = {}
        # Results of the "maximum" integral value of the primal objective.
        self.best_decoded_result = {}

    class Cluster:

        """
        Inner class for representing a cluster.
        A cluster is a subset of variables.

        Parameters
        ----------
        set_of_variables: tuple
                          This is the set of variables that form the cluster.

        intersection_set_variables: set containing frozensets.
                                    collection of intersection of all pairs of cluster variables.
                        For eg: \{\{C_1 \cap C_2\}, \{C_2 \cap C_3\}, \{C_3 \cap C_1\} \} for clusters C_1, C_2 & C_3.

        cluster_potential: Factor
                           Each cluster has a initial probability distribution provided beforehand.
        """
        def __init__(self, intersection_set_variables, cluster_potential):
            """
            Initialization of the current cluster
            """

            # The variables with which the cluster is made of.
            self.cluster_variables = cluster_potential.scope()

            # The cluster potentials must be specified before only.
            self.cluster_potential = copy.deepcopy(cluster_potential)

            # Generate intersection sets for this cluster; S(c)
            self.intersection_sets_for_cluster_c = [i.intersection(self.cluster_variables)
                                                    for i in intersection_set_variables
                                                    if i.intersection(self.cluster_variables)]

            # Initialize messages from this cluster to its respective intersection sets
            # \lambda_{c \rightarrow \s} = 0
            self.message_from_cluster = {}
            for intersection in self.intersection_sets_for_cluster_c:
                # Present variable. It can be a node or an edge too. (that is ['A'] or ['A', 'C'] too)
                present_variables = list(intersection)

                # Present variables cardinality
                present_variables_card = [cluster_potential.cardinality[cluster_potential.scope().index(variable)]
                                          for variable in present_variables]

                # We need to create a new factor whose messages are blank
                self.message_from_cluster[intersection] = \
                    Factor(present_variables, present_variables_card, np.zeros(np.prod(present_variables_card)))

    def _update_message(self, sending_cluster):

        """
        This is the message-update method.

        Parameters
        ----------
        sending_cluster: The resulting messages are \lambda_{c \rightarrow s} from the given
            cluster 'c' to all of its intersection_sets 's'.
            Here 's' are the elements of intersection_sets_for_cluster_c.

        Reference
        ---------
        Fixing Max-Product: Convergent Message-Passing Algorithms for MAP LP Relaxations
        by Amir Globerson and Tommi Jaakkola.
        Section 6, Page: 5; Beyond pairwise potentials: Generalized MPLP
        Later Modified by Sontag in "Introduction to Dual decomposition for Inference" Pg: 7 & 17
        """

        # The new updates will take place for the intersection_sets of this cluster.
        # The new updates are:
        # \delta_{f \rightarrow i}(x_i) = - \delta_i^{-f} +
        # 1/{\| f \|} max_{x_{f-i}}\left[{\theta_f(x_f) + \sum_{i' in f}{\delta_{i'}^{-f}}(x_i')} \right ]

        # Step. 1) Calculate {\theta_f(x_f) + \sum_{i' in f}{\delta_{i'}^{-f}}(x_i')}
        objective_cluster = self.objective[tuple(sending_cluster.cluster_variables)]
        for current_intersect in sending_cluster.intersection_sets_for_cluster_c:
            objective_cluster += self.objective[tuple(current_intersect)]

        updated_results = []
        objective = []
        for current_intersect in sending_cluster.intersection_sets_for_cluster_c:
            # Step. 2) Maximize step.1 result wrt variables present in the cluster but not in the current intersect.
            phi = objective_cluster.maximize(list(set(sending_cluster.cluster_variables) - current_intersect),
                                             inplace=False)

            # Step. 3) Multiply 1/{\| f \|}
            intersection_length = len(sending_cluster.intersection_sets_for_cluster_c)
            phi *= (1 / intersection_length)
            objective.append(phi)

            # Step. 4) Subtract \delta_i^{-f}
            # These are the messages not emanating from the sending cluster but going into the current intersect.
            # which is = Objective[current_intersect_node] - messages from the cluster to the current intersect node.
            updated_results.append(
                phi + -1 * (self.objective[tuple(current_intersect)]
                            + -1 * sending_cluster.message_from_cluster[current_intersect])
            )

        # This loop is primarily for simultaneous updating:
        # 1. This cluster's message to each of the intersects.
        # 2. The value of the Objective for intersection_nodes.
        index = -1
        cluster_potential = copy.deepcopy(sending_cluster.cluster_potential)
        for current_intersect in sending_cluster.intersection_sets_for_cluster_c:
            index += 1
            sending_cluster.message_from_cluster[current_intersect] = updated_results[index]
            self.objective[tuple(current_intersect)] = objective[index]
            cluster_potential += (-1) * updated_results[index]

        # Here we update the Objective for the current factor.
        self.objective[tuple(sending_cluster.cluster_variables)] = cluster_potential

    def _run_mplp(self):
        """
        Update messages for each factor whose scope is greater than 1

        """
        # We take the clusters in the order they were added in the model.
        for factor in self.model.get_factors():
            if len(factor.scope()) > 1:
                self._update_message(self.cluster_set[tuple(factor.scope())])

    def _local_decode(self):
        """
        Finds the index of the maximum values for all the single node dual objectives.

        Reference:
        code presented by Sontag in 2012 here: http://cs.nyu.edu/~dsontag/code/README_v2.html
        """
        # The current assignment of the single node factors is stored in the form of a dictionary
        decoded_result_assignment = {node[0]: np.argmax(self.objective[node].values)
                                     for node in self.objective if len(node) == 1}

        # Use the original cluster_potentials of each cluster to find the primal integral value.
        integer_value = 0
        for cluster_key in self.cluster_set:
            cluster = self.cluster_set[cluster_key]
            index = [tuple([variable, decoded_result_assignment[variable]]) for variable in cluster.cluster_variables]
            integer_value += cluster.cluster_potential.reduce(index, inplace=False).values[0]

        # Check if this is the best assignment till now
        if self.best_int_objective < integer_value:
            self.best_int_objective = integer_value
            self.best_assignment = decoded_result_assignment

    def _is_converged(self, dual_threshold, integrality_gap_threshold):
        """
        This method checks the integrality gap to ensure either:
            * we have found a near to exact solution or
            * stuck on a local minima.

        Reference:
        code presented by Sontag in 2012 here: http://cs.nyu.edu/~dsontag/code/README_v2.html
        """
        # Find the new objective after the message updates
        new_L = sum([max(self.objective[obj].values) for obj in self.objective])
        # As the decrement of the L gets very low, we assume that we might have stuck in a local minima.
        if abs(self.L - new_L) < 0.0002:
            return True
        # Check the threshold for the integrality gap
        elif abs(self.L - self.best_int_objective) < 0.0002:
            return True
        else:
            self.L = new_L
            return False

    def map_query(self, niter=1000, dual_threshold=0.0002, integrality_gap_threshold=0.0002):
        """
        MAP query method using Max Product LP method.
        This returns the best assignment of the nodes in the form of a dictionary.

        Parameters
        ----------
        niter: Number of maximum iterations that we want MPLP to run if _is_converged() fails

        dual_threshold: This sets the minimum width between the dual objective decrements. Is the decrement is lesser
                        than the threshold, then that means we have stuck on a local minima.

        integrality_gap_threshold: This sets the threshold for the integrality gap below which we say that the solution
                                   is satisfactory.

        Reference:
        Section 3.3: The Dual Algorithm; Tightening LP Relaxation for MAP using Message Passing (2008)
        By Sontag Et al.
        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> from pgmpy.factors import Factor
        >>> import numpy as np
        >>> student = MarkovModel()
        >>> student.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('E', 'F')])
        >>> factor_a = Factor(['A'], cardinality=[2], value=np.array([0.54577, 1.8323]))
        >>> factor_b = Factor(['B'], cardinality=[2], value=np.array([0.93894, 1.065]))
        >>> factor_c = Factor(['C'], cardinality=[2], value=np.array([0.89205, 1.121]))
        >>> factor_d = Factor(['D'], cardinality=[2], value=np.array([0.56292, 1.7765]))
        >>> factor_e = Factor(['E'], cardinality=[2], value=np.array([0.47117, 2.1224]))
        >>> factor_f = Factor(['F'], cardinality=[2], value=np.array([1.5093, 0.66257]))
        >>> factor_a_b = Factor(['A', 'B'], cardinality=[2, 2], value=np.array([1.3207, 0.75717, 0.75717, 1.3207]))
        >>> factor_b_c = Factor(['B', 'C'], cardinality=[2, 2], value=np.array([0.00024189, 4134.2, 4134.2, 0.00024189]))
        >>> factor_c_d = Factor(['C', 'D'], cardinality=[2, 2], value=np.array([0.0043227, 231.34, 231.34, 0.0043227]))
        >>> factor_d_e = Factor(['E', 'F'], cardinality=[2, 2], value=np.array([31.228, 0.032023, 0.032023, 31.228]))
        >>> student.add_factors(factor_a, factor_b, factor_c, factor_d, factor_e, factor_f, factor_a_b,
        ...    factor_b_c, factor_c_d, factor_d_e)
        >>> mplp = Mplp(student)
        >>> result = mplp.map_query()
        {'B': 0.93894, 'C': 1.121, 'A': 1.8323, 'F': 1.5093, 'D': 1.7765, 'E': 2.12239}

        """
        # Run one iteration of MPLP initially
        self._run_mplp()

        # Run MPLP until convergence using pairwise clusters.
        for i in range(niter):
            # Find an integral solution by locally maximizing the single node beliefs
            self._local_decode()
            # If the dual objective is sufficiently close to the primal objective, terminate
            if self._is_converged(dual_threshold, integrality_gap_threshold):
                break
            self._run_mplp()

        # Get the best result from the best assignment
        self.best_decoded_result = {factor.scope()[0]: factor.values[self.best_assignment[factor.scope()[0]]]
                                    for factor in self.model.factors if len(factor.scope()) == 1}
        return self.best_decoded_result
