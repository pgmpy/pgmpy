#!/usr/bin/env python3

from pgmpy.Inference import Inference
from pgmpy.factors.Factor import factor_product


class VariableElimination(Inference):
    def query(self, variables, conditions, elimination_order=None):
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
            elimination_order = list(set(self.variables) - set(variables.keys()) - set(conditions.keys()))

        for node in elimination_order:
            working_factors = self.factors
            phi = factor_product(*working_factors[node]).marginalize(node)
            del working_factors[node]
            for var in phi.variables:
                try:
                    working_factors[var].append(phi)
                except KeyError:
                    working_factors[var] = [phi]
