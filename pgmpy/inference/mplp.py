from pgmpy.inference import Inference
from pgmpy.models import MarkovModel
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
            raise ValueError('Only MarkovModel is supported')

        super().__init__(model)

        # S = \{c \cap c^{'} : c, c^{'} \in C, c \cap c^{'} \neq \emptyset\}
        self.intersection_set_variables = set()
        # We generate the Intersections of all the pairwise edges taken one at a time to form S
        for edge_pair in it.combinations(model.edges(), 2):
            self.intersection_set_variables.add(frozenset(edge_pair[0]) & frozenset(edge_pair[1]))

        # Initialize each cluster for variables appearing in each of the edges.
        # We also need the potentials of the current edges.
        self.cluster_set = []
        model_factors = model.get_factors()
        scope_list = [set(factor.scope()) for factor in model_factors]
        for edge in model.edges():
            index_of_factor = scope_list.index(set(edge))
            self.cluster_set.append(self.Cluster(edge, self.intersection_set_variables,
                                                 model_factors[index_of_factor]))

    class Cluster:

        """
        Inner class for representing a cluster.
        A cluster is a subset of variables.

        Parameters
        ----------
        set of variables: tuple
                          This is the set of variables that form the cluster.

        intersection_set_variables: set containing frozensets.
                                    collection of intersection of all pairs of cluster variables.
                        For eg: \{\{C_1 \cap C_2\}, \{C_2 \cap C_3\}, \{C_3 \cap C_1\} \} for clusters C_1, C_2 & C_3.

        cluster_potential: Factor
                           Each cluster has a initial probability distribution provided beforehand.
        """
        def __init__(self, set_of_variables_for_cluster_c, intersection_set_variables, cluster_potential):
            """
            Initialization of the current cluster
            """

            # The variables with which the cluster is made of.
            self.cluster_variables = set_of_variables_for_cluster_c

            # The cluster potentials must be specified before only.
            self.cluster_potential = copy.deepcopy(cluster_potential)

            # Generate intersection sets for this cluster; S(c)
            self.intersection_sets_for_cluster_c = [i.intersection(self.cluster_variables)
                                                    for i in intersection_set_variables
                                                    if i.intersection(self.cluster_variables)]

            # Initialize messages from this cluster to its respective intersection sets
            # \lambda_{c \rightarrow \s} = 1/|S(c)| * max_{x_{c-s}}{\theta_{ij}}
            self.message_from_cluster = {}
            for intersection in self.intersection_sets_for_cluster_c:
                other_variables = list(set(set_of_variables_for_cluster_c) - intersection)
                phi = copy.deepcopy(cluster_potential)
                # phi = max_{x_{c-s}}{\theta_{ij}}
                phi.maximize(other_variables)
                self.message_from_cluster[intersection] = (1 / len(self.intersection_sets_for_cluster_c)) * phi

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
        """
        updated_results = []
        # The new updates will take place for the intersection_sets of this cluster.
        for current_intersect in sending_cluster.intersection_sets_for_cluster_c:
            # Now we have to construct the crux of the equation which is used to update the messages there.
            # The terms which we need are:
            # 1. Message not originating from the current sending_cluster but going into the current intersect i
            message_not_from_cluster = []

            # 2. Summation of messages not originating from the current sending_cluster and going into other
            #    intersects than i of this cluster only.
            sum_other_intersects = []

            for cluster in self.cluster_set:
                if cluster == sending_cluster:
                    continue
                for intersection_set in cluster.intersection_sets_for_cluster_c:
                    if intersection_set == current_intersect:
                        message_not_from_cluster.append(cluster.message_from_cluster[intersection_set])

                for other_intersect in sending_cluster.intersection_sets_for_cluster_c:
                    if other_intersect == current_intersect:
                        continue
                    for intersection_set in cluster.intersection_sets_for_cluster_c:
                        if other_intersect == intersection_set:
                            sum_other_intersects.append(cluster.message_from_cluster[intersection_set])

            # lambda_1 = \lambda_{s}^{-c}
            # lambda_2 = max_{x_{c-s}}{\sum_{}{\lambda_{s'}^{-c}} + /theta_c}
            # here theta_c is the sending cluster potential
            lambda_2 = copy.deepcopy(sending_cluster.cluster_potential)
            for message in sum_other_intersects:
                lambda_2 += message

            # maximize w.r.t nodes in the present cluster but not in intersect
            lambda_2.maximize(list(set(sending_cluster.cluster_variables) - current_intersect))
            intersection_length = len(sending_cluster.intersection_sets_for_cluster_c)

            # lambda_c_s = K1(lambda_1) + K2(lambda_2)
            k1 = 1 / intersection_length
            k2 = -1 + k1
            # here lambda_c_s is the updated message
            lambda_c_s = k1 * lambda_2
            for message in message_not_from_cluster:
                lambda_c_s += k2 * message
            updated_results.append(lambda_c_s)

        # update for all the intersects of the sending cluster simultaneously.
        for i, j in zip(sending_cluster.intersection_sets_for_cluster_c, range(len(updated_results))):
            sending_cluster.message_from_cluster[i] = updated_results[j]
