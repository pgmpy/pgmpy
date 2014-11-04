#!/usr/bin/env python3

from pgmpy.Inference import Inference
from pgmpy.factors.Factor import factor_product


class VariableElimination(Inference):
    def query(self, variables, conditions, elimination_order=None):
        """
        Examples
        --------
        >>> from pgmpy.BayesianModel import BayesianModel
        >>> bm = BayesianModel([('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')])
        >>> # add cpds
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
