#!/usr/bin/env python3

from pgmpy.Inference import Inference
from pgmpy.factors.Factor import factor_product
from copy import deepcopy


class VariableElimination(Inference):
    def query(self, variables, conditions=None, elimination_order=None):
        """
        Examples
        --------
        For a Bayesian Model
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors import TabularCPD
        >>> bm = BayesianModel([('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')])
        >>> cpd_a = TabularCPD('a', 2, [[0.4, 0.6]])
        >>> cpd_b = TabularCPD('b', 2, [[0.4, 0.8], [0.6, 0.2]],
        ...                    evidence=['a'], evidence_card=[2])
        >>> cpd_c = TabularCPD('c', 2, [[0.2, 0.2], [0.8, 0.8]],
        ...                    evidence='b', evidence_card=[2])
        >>> cpd_d = TabularCPD('d', 2, [[0.5, 0.4], [0.5, 0.6]],
        ...                    evidence='c', evidence_card=[2])
        >>> cpd_e = TabularCPD('e', 2, [[0.1, 0.8], [0.9, 0.2]],
        ...                    evidence='d', evidence_card=[2])
        >>> bm.add_cpd([cpd_a, cpd_b, cpd_c, cpd_d, cpd_e])
        >>> VariableElimination(bm).query(variables={'c':{}})
        """
        if not elimination_order:
            if conditions:
                elimination_order = list(set(self.variables) - set(variables.keys()) - set(conditions.keys()))
            else:
                elimination_order = list(set(self.variables) - set(variables.keys()))

        factors_dict = deepcopy(self.factors)
        for node in elimination_order:
            import pdb; pdb.set_trace()
            if len(factors_dict[node]) == 1:
                phi = factors_dict[node][0].marginalize(node)
            else:
                phi = factor_product(*factors_dict[node]).marginalize(node)
            del factors_dict[node]
            import pdb; pdb.set_trace()
            for var in phi.variables:
                try:
                    factors_dict[var].append(phi)
                except KeyError:
                    factors_dict[var] = [phi]

        remaining_factors = set()
        for factor_lists in factors_dict.values():
            remaining_factors.union(set(factor_lists))

        marginalized = factor_product(*remaining_factors)
        if not conditions:
            return marginalized

        else:
            for node in variables.keys():
                phi = factor_product(*factors_dict[node].marginalize(node))
                del factors_dict[node]
                for var in phi.variables:
                    try:
                        factors_dict[var].append(phi)
                    except KeyError:
                        factors_dict[var] = [phi]

            remaining_factors = set()
            for factor_lists in factors_dict.values():
                remaining_factors.union(set(factor_lists))

            return marginalized / factor_product(*remaining_factors)