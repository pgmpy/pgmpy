import itertools
from collections import defaultdict

from pgmpy.inference import Inference
from pgmpy.inference import VariableElimination
from pgmpy.factors import TabularCPD



class DBNInference(Inference):

    def query(self, variables, evidence=None, elimination_order=None):
        variable_dict = defaultdict(list)
        for var in variables:
            variable_dict[var[1]].append(var[0])

        # This method extracts the max value in time slices
        timerange = max(variable_dict)
        if evidence:
            for node, timeslice in list(evidence):
                timerange = max(timerange, timeslice)

        int_nodes = self.interface_nodes
        inference = VariableElimination(self.start_bayesian_model)

        def vect(variables, time):
            values = [(var, time) for var in variables]
            return values

        def factor_to_cpd(factor):
            # This method converts factors to CPD
            # The problem arises when the CPD having one variable
            # is reduced and is not converted to an appropriate factor
            if isinstance(factor, TabularCPD):
                return factor
            variable = list(factor.variables)[0]
            variable = (variable[0], 0)
            variable_card = int(factor.cardinality[0])
            values = [factor.values.tolist()]
            cpd = TabularCPD(variable, variable_card, values)
            return cpd

        def dict_factor(time, shift):
            # This method filters out the nodes that are present in the evidence
            # and presents them in that format required for inference
            temp_dict = {}
            if evidence:
                for evid in evidence:
                    if evid[1] == time:
                        temp_dict[(evid[0], shift)] = evidence[evid]
                return temp_dict
            else:
                return None

        def filter_nodes(nodes, time, shift):
            # This method filters out the nodes that are not present in the evidence
            if not evidence:
                return nodes
            list_of_nodes = []
            for node in nodes:
                if node not in dict_factor(time, shift):
                    list_of_nodes.append(node)
            return list_of_nodes

        def assign_nodes(time, shift):
            # This method extracts the values for inference. In case the evidence is applied to
            # the evidence nodes, they need to be reduced appropriately.
            extracted_nodes = filter_nodes(int_nodes, time, shift) + vect(variable_dict[time], shift)
            s_values = inference.query(extracted_nodes, evidence=dict_factor(time, shift))

            if extracted_nodes != int_nodes and dict_factor(time, shift) is not None:
                for node in list(set(dict_factor(time, shift)).intersection(int_nodes)):
                    cpd = self.start_bayesian_model.get_cpds(node)
                    cpd = cpd.reduce([(node, evidence[node])], inplace=False)
                    s_values[node] = cpd
            return s_values

        start_values = assign_nodes(0, 0)
        final_values = {node: start_values[node] for node in vect(variable_dict[0], 0)}
        new_cpds = [factor_to_cpd(start_values[node]) for node in int_nodes]
        new_nodes = [(node[0], 1) for node in int_nodes]
        self.one_and_half_model.add_cpds(*new_cpds)

        for time in range(1, timerange + 1):
            inference = VariableElimination(self.one_and_half_model)
            int_time_nodes = [(var, 1) for var, time in int_nodes]
            if dict_factor(time, 1):
                new_values = inference.query(int_time_nodes, dict_factor)
            else:
                new_values = inference.query(int_time_nodes)
            new_cpds = [factor_to_cpd(new_values[node]) for node in new_nodes]
            old_cpds = [self.one_and_half_model.get_cpds(node) for node in int_nodes]
            self.one_and_half_model.remove_cpds(*old_cpds)
            self.one_and_half_model.add_cpds(*new_cpds)

            for node in variable_dict[time]:
                key = (node, time)
                cpd = new_values[(node, 1)]
                new_cpd = TabularCPD(key, int(cpd.cardinality[0]), [cpd.values])
                final_values[key] = new_cpd
        return final_values

class DBNJunctionTreeInference(Inference):
    """
    Class for performing inference using Junction Tree method.
    Creates a Junction Trees or Clique Trees (JunctionTree class) for the input
    Dynamic Bayesian Network and performs inference for the network
    Parameters
    ----------
    model: Dynamic Bayesian Network
        model for which inference is to performed
    """
    def __init__(self, model):
        super().__init__(model)

        #adding the edges between the interface nodes so as to form the cliques required for inference
        #after converting it to the markov model
        start_markov_model = self.start_bayesian_model.to_markov_model()
        one_and_half_markov_model = self.one_and_half_model.to_markov_model()

        combinations_slice_0 = list(itertools.combinations(model.get_interface_nodes(0), 2))
        combinations_slice_1 = list(itertools.combinations(model.get_interface_nodes(1), 2))

        start_markov_model.add_edges_from(combinations_slice_0)
        one_and_half_markov_model.add_edges_from(combinations_slice_0 + combinations_slice_1)

        #forming junction trees
        self.one_and_half_junction_tree = one_and_half_markov_model.to_junction_tree()
        self.start_junction_tree = start_markov_model.to_junction_tree()

        #classifying the cliques
        self.start_interface_clique = [clique for clique in self.start_junction_tree.nodes() if set(model.get_interface_nodes(0)).issubset(clique)]
        self.in_clique = [clique for clique in self.start_junction_tree.nodes() if set(model.get_interface_nodes(0)).issubset(clique)]
        self.out_clique = [clique for clique in self.one_and_half_junction_tree.nodes() if set(model.get_interface_nodes(1)).issubset(cliqu
