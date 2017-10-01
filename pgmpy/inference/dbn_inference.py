from collections import defaultdict
from itertools import tee, chain, combinations

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors import factor_product
from pgmpy.inference import Inference, BeliefPropagation


class DBNInference(Inference):
    def __init__(self, model):
        """
        Class for performing inference using Belief Propagation method
        for the input Dynamic Bayesian Network.

        For the exact inference implementation, the interface algorithm
        is used which is adapted from [1].

        Parameters:
        ----------
        model: Dynamic Bayesian Network
            Model for which inference is to performed

        Examples:
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.inference import DBNInference
        >>> dbnet = DBN()
        >>> dbnet.add_edges_from([(('Z', 0), ('X', 0)), (('X', 0), ('Y', 0)),
        ...                       (('Z', 0), ('Z', 1))])
        >>> z_start_cpd = TabularCPD(('Z', 0), 2, [[0.5, 0.5]])
        >>> x_i_cpd = TabularCPD(('X', 0), 2, [[0.6, 0.9],
        ...                                    [0.4, 0.1]],
        ...                      evidence=[('Z', 0)],
        ...                      evidence_card=[2])
        >>> y_i_cpd = TabularCPD(('Y', 0), 2, [[0.2, 0.3],
        ...                                    [0.8, 0.7]],
        ...                      evidence=[('X', 0)],
        ...                      evidence_card=[2])
        >>> z_trans_cpd = TabularCPD(('Z', 1), 2, [[0.4, 0.7],
        ...                                        [0.6, 0.3]],
        ...                      evidence=[('Z', 0)],
        ...                      evidence_card=[2])
        >>> dbnet.add_cpds(z_start_cpd, z_trans_cpd, x_i_cpd, y_i_cpd)
        >>> dbnet.initialize_initial_state()
        >>> dbn_inf = DBNInference(dbnet)
        >>> dbn_inf.start_junction_tree.nodes()
        [(('X', 0), ('Z', 0)), (('X', 0), ('Y', 0))]
        >>> dbn_inf.one_and_half_junction_tree.nodes()
        [(('Z', 1), ('Z', 0)),
         (('Y', 1), ('X', 1)),
         (('Z', 1), ('X', 1))]

        References:
        ----------
        [1] Dynamic Bayesian Networks: Representation, Inference and Learning
            by Kevin Patrick Murphy
            http://www.cs.ubc.ca/~murphyk/Thesis/thesis.pdf

        Public Methods:
        --------------
        forward_inference
        backward_inference
        query
        """
        super(DBNInference, self).__init__(model)
        self.interface_nodes_0 = model.get_interface_nodes(time_slice=0)
        self.interface_nodes_1 = model.get_interface_nodes(time_slice=1)

        start_markov_model = self.start_bayesian_model.to_markov_model()
        one_and_half_markov_model = self.one_and_half_model.to_markov_model()

        combinations_slice_0 = tee(combinations(self.interface_nodes_0, 2), 2)
        combinations_slice_1 = combinations(self.interface_nodes_1, 2)

        start_markov_model.add_edges_from(combinations_slice_0[0])
        one_and_half_markov_model.add_edges_from(chain(combinations_slice_0[1], combinations_slice_1))

        self.one_and_half_junction_tree = one_and_half_markov_model.to_junction_tree()
        self.start_junction_tree = start_markov_model.to_junction_tree()

        self.start_interface_clique = self._get_clique(self.start_junction_tree, self.interface_nodes_0)
        self.in_clique = self._get_clique(self.one_and_half_junction_tree, self.interface_nodes_0)
        self.out_clique = self._get_clique(self.one_and_half_junction_tree, self.interface_nodes_1)

    def _shift_nodes(self, nodes, time_slice):
        """
        Shifting the nodes to a certain required timeslice.

        Parameters:
        ----------
        nodes: list, array-like
            List of node names.
            nodes that are to be shifted to some other time slice.

        time_slice: int
            time slice where to shift the nodes.
        """
        return [(node[0], time_slice) for node in nodes]

    def _get_clique(self, junction_tree, nodes):
        """
        Extracting the cliques from the junction tree which are a subset of
        the given nodes.

        Parameters:
        ----------
        junction_tree: Junction tree
            from which the nodes are to be extracted.

        nodes: iterable container
            A container of nodes (list, dict, set, etc.).
        """

        return [clique for clique in junction_tree.nodes() if set(nodes).issubset(clique)][0]

    def _get_evidence(self, evidence_dict, time_slice, shift):
        """
        Getting the evidence belonging to a particular timeslice.

        Parameters:
        ----------
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        time: int
            the evidence corresponding to the time slice

        shift: int
            shifting the evidence corresponding to the given time slice.
        """
        if evidence_dict:
            return {(node[0], shift): evidence_dict[node] for node in evidence_dict if node[1] == time_slice}

    def _marginalize_factor(self, nodes, factor):
        """
        Marginalizing the factor selectively for a set of variables.

        Parameters:
        ----------
        nodes: list, array-like
            A container of nodes (list, dict, set, etc.).

        factor: factor
            factor which is to be marginalized.
        """
        marginalizing_nodes = list(set(factor.scope()).difference(nodes))
        return factor.marginalize(marginalizing_nodes, inplace=False)

    def _update_belief(self, belief_prop, clique, clique_potential, message=None):
        """
        Method for updating the belief.

        Parameters:
        ----------
        belief_prop: Belief Propagation
            Belief Propagation which needs to be updated.

        in_clique: clique
            The factor which needs to be updated corresponding to the input clique.

        out_clique_potential: factor
            Multiplying factor which will be multiplied to the factor corresponding to the clique.
        """
        old_factor = belief_prop.junction_tree.get_factors(clique)
        belief_prop.junction_tree.remove_factors(old_factor)
        if message:
            if message.scope() and clique_potential.scope():
                new_factor = old_factor * message
                new_factor = new_factor / clique_potential
            else:
                new_factor = old_factor
        else:
            new_factor = old_factor * clique_potential
        belief_prop.junction_tree.add_factors(new_factor)
        belief_prop.calibrate()

    def _get_factor(self, belief_prop, evidence):
        """
        Extracts the required factor from the junction tree.

        Parameters:
        ----------
        belief_prop: Belief Propagation
            Belief Propagation which needs to be updated.

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
        """
        final_factor = factor_product(*belief_prop.junction_tree.get_factors())
        if evidence:
            for var in evidence:
                if var in final_factor.scope():
                    final_factor.reduce([(var, evidence[var])])
        return final_factor

    def _shift_factor(self, factor, shift):
        """
        Shifting the factor to a certain required time slice.

        Parameters:
        ----------
        factor: DiscreteFactor
           The factor which needs to be shifted.

        shift: int
           The new timeslice to which the factor should belong to.
        """
        new_scope = self._shift_nodes(factor.scope(), shift)
        return DiscreteFactor(new_scope, factor.cardinality, factor.values)

    def forward_inference(self, variables, evidence=None, args=None):
        """
        Forward inference method using belief propagation.

        Parameters:
        ----------
        variables: list
            list of variables for which you want to compute the probability

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        Examples:
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.inference import DBNInference
        >>> dbnet = DBN()
        >>> dbnet.add_edges_from([(('Z', 0), ('X', 0)), (('X', 0), ('Y', 0)),
        ...                       (('Z', 0), ('Z', 1))])
        >>> z_start_cpd = TabularCPD(('Z', 0), 2, [[0.5, 0.5]])
        >>> x_i_cpd = TabularCPD(('X', 0), 2, [[0.6, 0.9],
        ...                                    [0.4, 0.1]],
        ...                      evidence=[('Z', 0)],
        ...                      evidence_card=[2])
        >>> y_i_cpd = TabularCPD(('Y', 0), 2, [[0.2, 0.3],
        ...                                    [0.8, 0.7]],
        ...                      evidence=[('X', 0)],
        ...                      evidence_card=[2])
        >>> z_trans_cpd = TabularCPD(('Z', 1), 2, [[0.4, 0.7],
        ...                                        [0.6, 0.3]],
        ...                      evidence=[('Z', 0)],
        ...                      evidence_card=[2])
        >>> dbnet.add_cpds(z_start_cpd, z_trans_cpd, x_i_cpd, y_i_cpd)
        >>> dbnet.initialize_initial_state()
        >>> dbn_inf = DBNInference(dbnet)
        >>> dbn_inf.forward_inference([('X', 2)], {('Y', 0):1, ('Y', 1):0, ('Y', 2):1})[('X', 2)].values
        array([ 0.76738736,  0.23261264])
        """
        variable_dict = defaultdict(list)
        for var in variables:
            variable_dict[var[1]].append(var)

        time_range = max(variable_dict)
        if evidence:
            evid_time_range = max([time_slice for var, time_slice in evidence.keys()])
            time_range = max(time_range, evid_time_range)

        start_bp = BeliefPropagation(self.start_junction_tree)
        mid_bp = BeliefPropagation(self.one_and_half_junction_tree)
        evidence_0 = self._get_evidence(evidence, 0, 0)
        interface_nodes_dict = {}
        potential_dict = {}

        if evidence:
            interface_nodes_dict = {k: v for k, v in evidence_0.items() if k in self.interface_nodes_0}
        initial_factor = self._get_factor(start_bp, evidence_0)
        marginalized_factor = self._marginalize_factor(self.interface_nodes_0, initial_factor)
        potential_dict[0] = marginalized_factor
        self._update_belief(mid_bp, self.in_clique, marginalized_factor)

        if variable_dict[0]:
            factor_values = start_bp.query(variable_dict[0], evidence=evidence_0)
        else:
            factor_values = {}

        for time_slice in range(1, time_range + 1):
            evidence_time = self._get_evidence(evidence, time_slice, 1)
            if interface_nodes_dict:
                evidence_time.update(interface_nodes_dict)

            if variable_dict[time_slice]:
                variable_time = self._shift_nodes(variable_dict[time_slice], 1)
                new_values = mid_bp.query(variable_time, evidence=evidence_time)
                changed_values = {}
                for key in new_values.keys():
                    new_key = (key[0], time_slice)
                    new_factor = DiscreteFactor([new_key], new_values[key].cardinality, new_values[key].values)
                    changed_values[new_key] = new_factor
                factor_values.update(changed_values)

            clique_phi = self._get_factor(mid_bp, evidence_time)
            out_clique_phi = self._marginalize_factor(self.interface_nodes_1, clique_phi)
            new_factor = self._shift_factor(out_clique_phi, 0)
            potential_dict[time_slice] = new_factor
            mid_bp = BeliefPropagation(self.one_and_half_junction_tree)
            self._update_belief(mid_bp, self.in_clique, new_factor)

            if evidence_time:
                interface_nodes_dict = {(k[0], 0): v for k, v in evidence_time.items() if k in self.interface_nodes_1}
            else:
                interface_nodes_dict = {}

        if args == 'potential':
            return potential_dict

        return factor_values

    def backward_inference(self, variables, evidence=None):
        """
        Backward inference method using belief propagation.

        Parameters:
        ----------
        variables: list
            list of variables for which you want to compute the probability
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        Examples:
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.inference import DBNInference
        >>> dbnet = DBN()
        >>> dbnet.add_edges_from([(('Z', 0), ('X', 0)), (('X', 0), ('Y', 0)),
        ...                       (('Z', 0), ('Z', 1))])
        >>> z_start_cpd = TabularCPD(('Z', 0), 2, [[0.5, 0.5]])
        >>> x_i_cpd = TabularCPD(('X', 0), 2, [[0.6, 0.9],
        ...                                    [0.4, 0.1]],
        ...                      evidence=[('Z', 0)],
        ...                      evidence_card=[2])
        >>> y_i_cpd = TabularCPD(('Y', 0), 2, [[0.2, 0.3],
        ...                                    [0.8, 0.7]],
        ...                      evidence=[('X', 0)],
        ...                      evidence_card=[2])
        >>> z_trans_cpd = TabularCPD(('Z', 1), 2, [[0.4, 0.7],
        ...                                        [0.6, 0.3]],
        ...                      evidence=[('Z', 0)],
        ...                      evidence_card=[2])
        >>> dbnet.add_cpds(z_start_cpd, z_trans_cpd, x_i_cpd, y_i_cpd)
        >>> dbnet.initialize_initial_state()
        >>> dbn_inf = DBNInference(dbnet)
        >>> dbn_inf.backward_inference([('X', 0)], {('Y', 0):0, ('Y', 1):1, ('Y', 2):1})[('X', 0)].values
        array([ 0.66594382,  0.33405618])
        """
        variable_dict = defaultdict(list)
        for var in variables:
            variable_dict[var[1]].append(var)
        time_range = max(variable_dict)
        interface_nodes_dict = {}
        if evidence:
            evid_time_range = max([time_slice for var, time_slice in evidence.keys()])
            time_range = max(time_range, evid_time_range)
        end_bp = BeliefPropagation(self.start_junction_tree)
        potential_dict = self.forward_inference(variables, evidence, 'potential')
        update_factor = self._shift_factor(potential_dict[time_range], 1)
        factor_values = {}

        for time_slice in range(time_range, 0, -1):
            evidence_time = self._get_evidence(evidence, time_slice, 1)
            evidence_prev_time = self._get_evidence(evidence, time_slice - 1, 0)
            if evidence_prev_time:
                interface_nodes_dict = {k: v for k, v in evidence_prev_time.items() if k in self.interface_nodes_0}
            if evidence_time:
                evidence_time.update(interface_nodes_dict)
            mid_bp = BeliefPropagation(self.one_and_half_junction_tree)
            self._update_belief(mid_bp, self.in_clique, potential_dict[time_slice - 1])
            forward_factor = self._shift_factor(potential_dict[time_slice], 1)
            self._update_belief(mid_bp, self.out_clique, forward_factor, update_factor)

            if variable_dict[time_slice]:
                variable_time = self._shift_nodes(variable_dict[time_slice], 1)
                new_values = mid_bp.query(variable_time, evidence=evidence_time)
                changed_values = {}
                for key in new_values.keys():
                    new_key = (key[0], time_slice)
                    new_factor = DiscreteFactor([new_key], new_values[key].cardinality, new_values[key].values)
                    changed_values[new_key] = new_factor
                factor_values.update(changed_values)

            clique_phi = self._get_factor(mid_bp, evidence_time)
            in_clique_phi = self._marginalize_factor(self.interface_nodes_0, clique_phi)
            update_factor = self._shift_factor(in_clique_phi, 1)

        out_clique_phi = self._shift_factor(update_factor, 0)
        self._update_belief(end_bp, self.start_interface_clique, potential_dict[0], out_clique_phi)
        evidence_0 = self._get_evidence(evidence, 0, 0)
        if variable_dict[0]:
            factor_values.update(end_bp.query(variable_dict[0], evidence_0))
        return factor_values

    def query(self, variables, evidence=None, args='exact'):
        """
        Query method for Dynamic Bayesian Network using Interface Algorithm.

        Parameters:
        ----------
        variables: list
            list of variables for which you want to compute the probability

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        Examples:
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.inference import DBNInference
        >>> dbnet = DBN()
        >>> dbnet.add_edges_from([(('Z', 0), ('X', 0)), (('X', 0), ('Y', 0)),
        ...                       (('Z', 0), ('Z', 1))])
        >>> z_start_cpd = TabularCPD(('Z', 0), 2, [[0.5, 0.5]])
        >>> x_i_cpd = TabularCPD(('X', 0), 2, [[0.6, 0.9],
        ...                                    [0.4, 0.1]],
        ...                      evidence=[('Z', 0)],
        ...                      evidence_card=[2])
        >>> y_i_cpd = TabularCPD(('Y', 0), 2, [[0.2, 0.3],
        ...                                    [0.8, 0.7]],
        ...                      evidence=[('X', 0)],
        ...                      evidence_card=[2])
        >>> z_trans_cpd = TabularCPD(('Z', 1), 2, [[0.4, 0.7],
        ...                                        [0.6, 0.3]],
        ...                      evidence=[('Z', 0)],
        ...                      evidence_card=[2])
        >>> dbnet.add_cpds(z_start_cpd, z_trans_cpd, x_i_cpd, y_i_cpd)
        >>> dbnet.initialize_initial_state()
        >>> dbn_inf = DBNInference(dbnet)
        >>> dbn_inf.query([('X', 0)], {('Y', 0):0, ('Y', 1):1, ('Y', 2):1})[('X', 0)].values
        array([ 0.66594382,  0.33405618])
        """
        if args == 'exact':
            return self.backward_inference(variables, evidence)
