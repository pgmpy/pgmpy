import itertools
from collections import defaultdict

from pgmpy.models import BayesianModel, DynamicBayesianNetwork
from pgmpy.factors import Factor
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
        """
        super().__init__(model)
        self.interface_nodes_0 = model.get_interface_nodes(0)
        self.interface_nodes_1 = model.get_interface_nodes(1)

        start_markov_model = self.start_bayesian_model.to_markov_model()
        one_and_half_markov_model = self.one_and_half_model.to_markov_model()

        combinations_slice_0 = list(itertools.combinations(model.get_interface_nodes(0), 2))
        combinations_slice_1 = list(itertools.combinations(model.get_interface_nodes(1), 2))

        start_markov_model.add_edges_from(combinations_slice_0)
        one_and_half_markov_model.add_edges_from(combinations_slice_0 + combinations_slice_1)

        self.one_and_half_junction_tree = one_and_half_markov_model.to_junction_tree()
        self.start_junction_tree = start_markov_model.to_junction_tree()

        self.start_interface_clique = [clique for clique in self.start_junction_tree.nodes() if set(model.get_interface_nodes(0)).issubset(clique)][0]
        self.in_clique = [clique for clique in self.one_and_half_junction_tree.nodes() if set(model.get_interface_nodes(0)).issubset(clique)][0]
        self.out_clique = [clique for clique in self.one_and_half_junction_tree.nodes() if set(model.get_interface_nodes(1)).issubset(clique)][0]

    def _shift_nodes(self, nodes, time):
        """
        shifting the nodes to a certain required timeslice
        Parameters
        ----------
        shift: int
            shifting the evidence corresponding to the given time slice.
        """
        return [(node[0], time) for node in nodes]

    def _get_evidence(self, evidence_dict, time, shift):
        """
        getting the evidence belonging to a particular timeslice
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
        marginalizing the factor selectively for a set of variables.
        Parameters
        ----------
        nodes: list
           The nodes which will be marginalized in the factor.
        factor: factor
           factor which is to be marginalized. 
        """
        marginalizing_nodes = list(set(factor.scope()).difference(nodes))
        new_factor = factor.marginalize(marginalizing_nodes, inplace=False)
        return new_factor

    def _update_belief(self, belief_prop, in_clique, out_clique_potential):
        """
        method for updating the belief.
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

    def forward_inference(self, variables, evidence=None):
        """
        forward inference method using belief propagation.
        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence
        """

        variable_dict = defaultdict(list)
        for var in variables:
                variable_dict[var[1]].append(var)
        time_range = max(variable_dict)
        start_bp = BeliefPropagation(self.start_junction_tree)
        mid_bp = BeliefPropagation(self.one_and_half_junction_tree)
        start_bp.calibrate()
        mid_bp.calibrate()
        evidence_0 = self._get_evidence(evidence, 0, 0)
        interface_nodes_dict = {}
        if evidence is not None:
            interface_nodes_dict = {k: v for k, v in evidence_0.items() if k in self.interface_nodes_0}
        initial_factor = start_bp.clique_beliefs[self.start_interface_clique]
        initial_factor = self._marginalize_factor(self.interface_nodes_0, initial_factor)
        self._update_belief(mid_bp, self.in_clique, initial_factor)
        if variable_dict[0] != []:
            factor_values = start_bp.query(variable_dict[0], evidence=evidence_0)
        else:
            factor_values = {}

        for time in range(1, time_range + 1):
            evidence_time = self._get_evidence(evidence, time, 1)
            if interface_nodes_dict != {}:
                if evidence_time is not None:
                    evidence_time.update(interface_nodes_dict)
                else:
                    evidence_time = interface_nodes_dict

            if variable_dict[time] != []:
                variable_time = self._shift_nodes(variable_dict[time], 1)
                for clique in mid_bp.clique_beliefs:
                    factor = mid_bp.clique_beliefs[clique]
                new_values = mid_bp.query(variable_time, evidence=evidence_time)
                changed_values = {}
                for key in new_values.keys():
                    new_key = (key[0], time)
                    new_factor = Factor([new_key], new_values[key].cardinality, new_values[key].values)
                    changed_values[new_key] = new_factor
                factor_values.update(changed_values)

            out_clique_phi = mid_bp.clique_beliefs[self.out_clique]
            out_clique_phi = self._marginalize_factor(self.interface_nodes_1, out_clique_phi)
            new_scope = self._shift_nodes(out_clique_phi.scope(), 0)
            new_factor = Factor(new_scope, out_clique_phi.cardinality, out_clique_phi.values)
            self._update_belief(mid_bp, self.in_clique, new_factor)
            if evidence_time is not None:
                interface_nodes_dict = {(k[0], 0): v for k, v in evidence_time.items() if k in self.interface_nodes_1}
            else:
                interface_nodes_dict = {}
        return factor_values
