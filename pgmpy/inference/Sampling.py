#!/usr/bin/env python3

import networkx as nx

from pgmpy.models import BayesianModel
from pgmpy.utils.mathext import sample_discrete
from pgmpy.inference import Inference


class BayesianModelSampling(Inference):
    """
    Class for sampling methods specific to Bayesian Models

    Parameters
    ----------
    model: instance of BayesianModel
        model on which inference queries will be computed


    Public Methods
    --------------
    forward_sample(size)
    """
    def __init__(self, model):
        if not isinstance(model, BayesianModel):
            raise TypeError("model must an instance of BayesianModel")
        super(BayesianModelSampling, self).__init__(model)
        self.topological_order = nx.topological_sort(model)
        self.cpds = {}
        for node in model.nodes():
            self.cpds[node] = model.get_cpds(node)

    def forward_sample(self, size=1):
        """
        Generates sample(s) from joint distribution of the bayesian network.

        Parameters
        ----------
        size: int
            size of sample to be generated

        Returns
        -------
        sampled: list of dicts
            the generated samples

        Examples
        --------
        >>> from pgmpy.models.BayesianModel import BayesianModel
        >>> from pgmpy.factors.CPD import TabularCPD
        >>> from pgmpy.inference.Sampling import BayesianModelSampling
        >>> student = BayesianModel([('diff', 'grade'), ('intel', 'grade')])
        >>> cpd_d = TabularCPD('diff', 2, [[0.6], [0.4]])
        >>> cpd_i = TabularCPD('intel', 2, [[0.7], [0.3]])
        >>> cpd_g = TabularCPD('grade', 3, [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25,
        ...                0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
        ...                ['intel', 'diff'], [2, 2])
        >>> student.add_cpds(cpd_d, cpd_i, cpd_g)
        >>> inference = BayesianModelSampling(student)
        >>> inference.forward_sample(2)
        [{'intel': 'intel_0', 'grade': 'grade_1', 'diff': 'diff_1'},
        {'intel':  'intel_1', 'grade': 'grade_0', 'diff': 'diff_0'}]
        """
        sampled = []
        for _ in range(size):
            particle = {}
            for node in self.topological_order:
                cpd = self.cpds[node]
                if cpd.evidence:
                    evid = []
                    for var in cpd.evidence:
                        evid.append(particle[var])
                    weights = cpd.reduce(evid, inplace=False).values
                else:
                    weights = cpd.values
                particle[node] = sample_discrete(cpd.variables[cpd.variable], weights)
            sampled.append(particle)
        return sampled
