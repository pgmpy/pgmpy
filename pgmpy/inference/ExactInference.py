#!/usr/bin/env python3

from pgmpy.inference import Inference
from pgmpy.factors.Factor import factor_product


class VariableElimination(Inference):
    def query(self, variables, evidence=None, elimination_order=None):
        """
        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence
        elimination_order: list
            order of variable eliminations (if nothing is provided) order is
            computed automatically

        Examples
        --------
        >>> from pgmpy.inference import VariableElimination
        >>> from pgmpy.models import BayesianModel
        >>> import numpy as np
        >>> import pandas as pd
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> inference = VariableElimination(model)
        >>> phi_query = inference.query(['A', 'B'])
        """
        eliminated_variables = set()
        working_factors = {node: [factor for factor in self.factors[node]]
                           for node in self.factors}

        # TODO: Modify it to find the optimal elimination order
        if not elimination_order:
            elimination_order = list(set(self.variables) -
                                     set(variables) -
                                     set(evidence.keys() if evidence else []))

        elif any(var in elimination_order for var in set(variables).union(set(evidence.keys() if evidence else []))):
            raise ValueError("elimination_order contains variables which are in variables or evidence args")

        for var in elimination_order:
            # Removing all the factors containing the variables which are
            # eliminated (as all the factors should be considered only once)
            factors = [factor for factor in working_factors[var]
                       if not set(factor.variables).intersection(eliminated_variables)]
            phi = factor_product(*factors)
            phi.marginalize(var)
            del working_factors[var]
            for variable in phi.variables:
                working_factors[variable].append(phi)
            eliminated_variables.add(var)

        final_distribution = set()
        for node in working_factors:
            factors = working_factors[node]
            for factor in factors:
                if not set(factor.variables).intersection(eliminated_variables):
                    final_distribution.add(factor)

        query_var_factor = {}
        for query_var in variables:
            phi = factor_product(*final_distribution).marginalize(
                list(set(variables) - set([query_var])), inplace=False)
            if evidence:
                phi.reduce(['{evidence_var}_{evidence}'.format(
                    evidence_var=evidence_var, evidence=evidence[evidence_var]) for evidence_var in evidence])
                phi.normalize()

            query_var_factor[query_var] = phi

        return query_var_factor

