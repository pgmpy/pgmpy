from itertools import product

import numpy as np

from pgmpy.factors.discrete import TabularCPD


class NoisyORCPD(TabularCPD):
    """
    Initializes the NoisyORCPD class.

    The NoisyOR CPD is a special case of TabularCPD for binary variables
    where a given variable can be activated by each of the parent variables
    with a specified probability. This activation probability is defined
    in the `prob_values` argument.

    Parameters
    ----------
    variable: str
        The variable for which the CPD is to be defined.

    prob_values: iterable
        A list of probabilities values for each `evidence` variable
        to activate `variable`.

    evidence: list
        List of evidence variables, i.e., conditional variables.

    Examples
    --------
    >>> from pgmpy.factors.discrete import NoisyORCPD
    >>> cpd = NoisyORCPD(variable='Y', prob_values=[0.3, 0.5], evidence=['X1', 'X2'])
    # Defining a model containing NoisyORCPD
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> model = DiscreteBayesianNetwork(['A', 'B'])
    # With nodes with no parents, we can not define a NoisyORCPD.
    >>> cpd_a = TabularCPD('A', 2, [[0.2], [0.8]], state_names={'A': ['True', 'False']})
    >>> cpd_b = NoisyORCPD('B', [0.8], evidence=['A'])
    >>> model.add_cpds(cpd_a, cpd_b)
    """

    def __init__(self, variable, prob_values, evidence):
        if len(prob_values) != len(evidence):
            raise ValueError(
                "Number of prob_values should be equal to number of evidence variables"
            )

        self.prob_values = np.array(prob_values)
        if any(self.prob_values > 1) or any(self.prob_values < 0):
            raise ValueError("Values in prob_values should be between 0 and 1")

        inv_prob_values = 1 - self.prob_values
        tabular_values = np.zeros(2 ** len(evidence))
        state_comb = product([True, False], repeat=len(evidence))
        for i, states in enumerate(state_comb):
            tabular_values[i] = 1 - np.prod(inv_prob_values[np.array(states)])

        tabular_values = np.stack((tabular_values, 1 - tabular_values))

        state_names = {variable: ["True", "False"]}
        state_names.update({var: ["True", "False"] for var in evidence})

        super(NoisyORCPD, self).__init__(
            variable=variable,
            variable_card=2,
            values=tabular_values,
            evidence=evidence,
            evidence_card=[2] * len(evidence),
            state_names=state_names,
        )
