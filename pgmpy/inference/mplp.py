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
        super().__init__(model)

        if not isinstance(model, MarkovModel):
            raise ValueError('Only MarkovModel is supported')

        # S = { c ∩ c': c, c' ∈ C, c ∩ c' != Null}
        # It should be ready before initializing Cluster.
        # Nothing but union of all the edge variables.
        self.intersection_set_variables = set()

        # The generation of all the Intersections of all
        # the pairwise edges taken one at a time
        for edge_pair in it.combinations([sorted(i) for i in model.edges()], 2):
            self.intersection_set_variables.add(frozenset(edge_pair[0]) & frozenset(edge_pair[1]))

        # Initialize each cluster for variables appearing in each of the edges.
        # We also need the potentials of the current edges.
        self.cluster_set = []
        scope_list = [i.scope() for i in model.get_factors()]
        for edge in [sorted(i) for i in model.edges()]:
            index_of_factor = scope_list.index(list(edge))
            self.cluster_set.append(self.Cluster(edge, self.intersection_set_variables,
                                                 model.get_factors()[index_of_factor]))

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
                                    For eg: {{C1 ∩ C2}, {C2 ∩ C3}, {C3 ∩ C1}} for clusters C1, C2 & C3.

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
            for i in self.intersection_sets_for_cluster_c:
                other_variables = list(set(set_of_variables_for_cluster_c)-i)
                phi = copy.deepcopy(cluster_potential)

                # phi = max_{x_{c-s}}{\theta_{ij}}
                phi.maximize(other_variables)
                self.message_from_cluster[i] = (1 / len(self.intersection_sets_for_cluster_c)) * phi

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
        for i in sending_cluster.intersection_sets_for_cluster_c:

            # Now we have to construct the crux of the equation which is used to update the messages there.

            # The terms which we need are:
            # 1. Message not originating from the current sending_cluster but going into the current intersect i
            message_not_from_cluster = []

            # 2. Summation of messages not originating from the current sending_cluster and going into other
            #    intersects than i of this cluster only.
            sum_other_intersects = []

            cluster_list_except_current = [j for j in self.cluster_set if j != sending_cluster]
            other_intersects = [j for j in sending_cluster.intersection_sets_for_cluster_c if j != i]

            for cluster in cluster_list_except_current:
                for intersection_set in cluster.intersection_sets_for_cluster_c:
                    if i == intersection_set:
                        message_not_from_cluster.append(cluster.message_from_cluster[intersection_set])

                for other_intersect in other_intersects:
                        for intersection_set in cluster.intersection_sets_for_cluster_c:
                            if other_intersect == intersection_set:
                                sum_other_intersects.append(cluster.message_from_cluster[intersection_set])

            # lambda_1 = \lambda_{s}^{-c}
            lambda_1 = copy.deepcopy(sending_cluster.message_from_cluster[i])
            lambda_1.values = np.zeros(np.prod(lambda_1.cardinality))
            # lambda_2 = \sum_{}{\lambda_{s'}^{-c}}
            lambda_2 = copy.deepcopy(lambda_1)
            for message in message_not_from_cluster:
                lambda_1 += message
            for message in sum_other_intersects:
                lambda_2 += message
            # maximize (lambda_2 + sending cluster potential) w.r.t nodes in the present cluster but not in intersect
            phi = sending_cluster.cluster_potential + lambda_2
            # phi = max_{x_{c-s}}{lambda_2 + /theta_c}
            phi.maximize(list(set(sending_cluster.cluster_variables)-i))
            intersection_length = len(sending_cluster.intersection_sets_for_cluster_c)
            # \lambda_{c \rightarrow \s} = -(1 - 1 / |S(c)|) lambda_1 + (1 / |S(c)|) phi
            lambda_c_s = -(1 - 1 / intersection_length) * lambda_1 + (1 / intersection_length) * phi
            updated_results.append(lambda_c_s)

        for i, j in zip(sending_cluster.intersection_sets_for_cluster_c, range(len(updated_results))):
            sending_cluster.message_from_cluster[i] = updated_results[j]
