import copy
import itertools
from collections import defaultdict

from pgmpy.models import BayesianModel, DynamicBayesianNetwork
from pgmpy.factors import Factor
from pgmpy.factors.Factor import factor_product
from pgmpy.inference import Inference, BeliefPropagation


class DBNInference(Inference):

    def __init__(self, model):
        """
        Class for performing inference using Belief Propagation method
        for the input Dynamic Bayesian Network.

        Parameters
        ----------
        model: Dynamic Bayesian Network
        model for which inference is to performed

        Examples
        --------
        >>> from pgmpy.models import BayesianModel as bm
        >>> from pgmpy.models import DynamicBayesianNetwork as dbn
        >>> dbnet = dbn()
        >>> grade_cpd = TabularCPD(('G',0),3, [[0.3, 0.05, 0.9, 0.5],
                                               [0.4, 0.25, 0.08, 0.3],
                                               [0.3, 0.7, 0.2, 0.2]],[('D',0),('I',0)],[2,2])
        >>> d_i_cpd = TabularCPD(('D',1),2,[[0.6,0.3],[0.4,0.7]],[('D',0)],2)
        >>> diff_cpd = TabularCPD(('D',0),2,[[0.6,0.4]])
        >>> intel_cpd = TabularCPD(('I',0),2,[[0.7,0.3]])
        >>> i_i_cpd = TabularCPD(('I',1),2,[[0.5,0.4],[0.5,0.6]],[('I',0)],2)
        >>> dbnet.add_edges_from([(('D', 0), ('D', 1)), (('I', 0), ('I', 1)), (('D',1),('G',1)), (('I',1), ('G',1))])
        >>> dbnet.add_cpds(grade_cpd, d_i_cpd, diff_cpd, intel_cpd, i_i_cpd)
        >>> dbnet.initialize_initial_state()
        >>> dbn_inf = DBNInference(dbnet)
        >>> dbn_inf.start_junction_tree.nodes()
        [(('I', 0), ('G', 0), ('D', 0))]
        >>> dbn_inf.one_and_half_junction_tree.nodes()
        [(('I', 1), ('D', 1), ('D', 0)),
        (('I', 1), ('D', 1), ('G', 1)),
        (('I', 1), ('I', 0), ('D', 0))]
        """
        super().__init__(model)
        self.interface_nodes_0 = model.get_interface_nodes(0)
        self.interface_nodes_1 = model.get_interface_nodes(1)

        start_markov_model = self.start_bayesian_model.to_markov_model()
        one_and_half_markov_model = self.one_and_half_model.to_markov_model()

        combinations_slice_0 = itertools.combinations(model.get_interface_nodes(0), 2)
        combinations_slice_1 = itertools.combinations(model.get_interface_nodes(1), 2)

        start_markov_model.add_edges_from(combinations_slice_0)
        one_and_half_markov_model.add_edges_from(itertools.chain(combinations_slice_0, combinations_slice_1))

        self.one_and_half_junction_tree = one_and_half_markov_model.to_junction_tree()
        self.start_junction_tree = start_markov_model.to_junction_tree()

        self.start_interface_clique = self._get_clique(self.start_junction_tree, self.interface_nodes_0)
        self.in_clique = self._get_clique(self.one_and_half_junction_tree, self.interface_nodes_0)
        self.out_clique = self._get_clique(self.one_and_half_junction_tree, self.interface_nodes_1)

    def _shift_nodes(self, nodes, time):
        """
        Shifting the nodes to a certain required timeslice.

        Parameters
        ----------
        shift: int
            shifting the evidence corresponding to the given time slice.
        """
        return [(node[0], time) for node in nodes]

    def _get_clique(self, junction_tree, nodes):
        """
        Extracting the cliques from the junction tree which are a subset of
        the given nodes.

        Parameters
        ----------
        junction_tree: Junction tree
            from which the nodes are to be extracted.
        nodes: iterable container
            A container of nodes (list, dict, set, etc.).
        """
        return [clique for clique in junction_tree.nodes() if set(nodes).issubset(clique)][0]

    def _get_evidence(self, evidence_dict, time, shift):
        """
        Getting the evidence belonging to a particular timeslice.

        Parameters
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
            return {(node[0], shift): evidence_dict[node] for node in evidence_dict if node[1] == time}
        else:
            return None

    def _marginalize_factor(self, nodes, factor):
        """
        Marginalizing the factor selectively for a set of variables.

        Parameters
        ----------
        nodes: iterable container
            A container of nodes (list, dict, set, etc.).
        factor: factor
            factor which is to be marginalized.
        """
        marginalizing_nodes = list(set(factor.scope()).difference(nodes))
        new_factor = factor.marginalize(marginalizing_nodes, inplace=False)
        return new_factor

    def _update_belief(self, belief_prop, in_clique, out_clique_potential):
        """
        Method for updating the belief.

        Parameters
        ----------
        belief_prop: Belief Propagation
            Belief Propagation which needs to be updated.
        in_clique: clique
            The factor which needs to be updated corresponding to the input clique.
        out_clique_potential: factor
            Multiplying factor which will be multiplied to the factor corresponding to the clique.
        """
        old_factor = belief_prop.junction_tree.get_factors(in_clique)
        belief_prop.junction_tree.remove_factors(old_factor)
        new_factor = old_factor*out_clique_potential
        belief_prop.junction_tree.add_factors(new_factor)
        belief_prop.calibrate()

    def _get_factor(self, belief_prop, evidence):
        """
        Extracts the required factor from the junction tree.

        Parameters
        ----------
        belief_prop: Belief Propagation
            Belief Propagation which needs to be updated.
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
        """
        final_factor = copy.deepcopy(factor_product(*belief_prop.junction_tree.get_factors()))
        if evidence:
            for var in evidence:
                if var in final_factor.scope():
                    final_factor.reduce((var, evidence[var]))
        return final_factor

    def forward_inference(self, variables, evidence=None):
        """
        Forward inference method using belief propagation.

        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        Examples
        --------
        >>> from pgmpy.models import BayesianModel as bm
        >>> from pgmpy.models import DynamicBayesianNetwork as dbn
        >>> dbnet = dbn()
        >>> grade_cpd = TabularCPD(('G',0),3, [[0.3, 0.05, 0.9, 0.5],
                                               [0.4, 0.25, 0.08, 0.3],
                                               [0.3, 0.7, 0.2, 0.2]],[('D',0),('I',0)],[2,2])
        >>> d_i_cpd = TabularCPD(('D',1),2,[[0.6,0.3],[0.4,0.7]],[('D',0)],2)
        >>> diff_cpd = TabularCPD(('D',0),2,[[0.6,0.4]])
        >>> intel_cpd = TabularCPD(('I',0),2,[[0.7,0.3]])
        >>> i_i_cpd = TabularCPD(('I',1),2,[[0.5,0.4],[0.5,0.6]],[('I',0)],2)
        >>> dbnet.add_edges_from([(('D', 0), ('D', 1)), (('I', 0), ('I', 1)), (('D',1),('G',1)), (('I',1), ('G',1))])
        >>> dbnet.add_cpds(grade_cpd, d_i_cpd, diff_cpd, intel_cpd, i_i_cpd)
        >>> dbnet.initialize_initial_state()
        >>> dbn_inf = DBNInference(dbnet)
        >>> dbn_inf.query([('G',0)], evidence={('D',0):0})
        """
        variable_dict = defaultdict(list)
        for var in variables:
                variable_dict[var[1]].append(var)
        time_range = max(variable_dict)
        start_bp = BeliefPropagation(self.start_junction_tree)
        mid_bp = BeliefPropagation(self.one_and_half_junction_tree)
        evidence_0 = self._get_evidence(evidence, 0, 0)
        interface_nodes_dict = {}
        if evidence:
            interface_nodes_dict = {k: v for k, v in evidence_0.items() if k in self.interface_nodes_0}
        initial_factor = self._get_factor(start_bp, evidence_0)
        marginalized_factor = self._marginalize_factor(self.interface_nodes_0, initial_factor)
        self._update_belief(mid_bp, self.in_clique, marginalized_factor)
        if variable_dict[0]:
            factor_values = start_bp.query(variable_dict[0], evidence=evidence_0)
        else:
            factor_values = {}

        for time in range(1, time_range + 1):
            evidence_time = self._get_evidence(evidence, time, 1)
            if interface_nodes_dict:
                if evidence_time is not None:
                    evidence_time.update(interface_nodes_dict)
                else:
                    evidence_time = interface_nodes_dict

            if variable_dict[time]:
                variable_time = self._shift_nodes(variable_dict[time], 1)
                new_values = mid_bp.query(variable_time, evidence=evidence_time)
                changed_values = {}
                for key in new_values.keys():
                    new_key = (key[0], time)
                    new_factor = Factor([new_key], new_values[key].cardinality, new_values[key].values)
                    changed_values[new_key] = new_factor
                factor_values.update(changed_values)

            clique_phi = self._get_factor(mid_bp, evidence_time)
            out_clique_phi = self._marginalize_factor(self.interface_nodes_1, clique_phi)
            new_scope = self._shift_nodes(out_clique_phi.scope(), 0)
            new_factor = Factor(new_scope, out_clique_phi.cardinality, out_clique_phi.values)
            self._update_belief(mid_bp, self.in_clique, new_factor)
            if evidence_time:
                interface_nodes_dict = {(k[0], 0): v for k, v in evidence_time.items() if k in self.interface_nodes_1}
            else:
                interface_nodes_dict = {}
        return factor_values
