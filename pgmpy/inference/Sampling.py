#!/usr/bin/env python3
from collections import namedtuple

import networkx as nx
import numpy as np
from pandas import DataFrame

from pgmpy.inference import Inference
from pgmpy.models import BayesianModel
from pgmpy.utils.mathext import sample_discrete


State = namedtuple('State', ['var', 'state'])


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
        super().__init__(model)
        self.topological_order = nx.topological_sort(model)
        self.cpds = {node: model.get_cpds(node) for node in model.nodes()}

    def forward_sample(self, size=1):
        """
        Generates sample(s) from joint distribution of the bayesian network.

        Parameters
        ----------
        size: int
            size of sample to be generated

        Returns
        -------
        sampled: pandas.DataFrame
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
                diff       intel       grade
        0  (diff, 1)  (intel, 0)  (grade, 1)
        1  (diff, 1)  (intel, 0)  (grade, 2)
        """
        sampled = DataFrame(index=range(size), columns=self.topological_order)
        for node in self.topological_order:
            cpd = self.cpds[node]
            states = [st for var, st in cpd.variables[node]]
            if cpd.evidence:
                indices = [i for i, x in enumerate(self.topological_order) if x in cpd.evidence]
                evidence = sampled.values[:, [indices]].tolist()
                weights = list(map(lambda t: cpd.reduce(t[0], inplace=False).values, evidence))
                sampled[node] = list(map(lambda t: State(node, t), sample_discrete(states, weights)))
            else:
                sampled[node] = list(map(lambda t: State(node, t),
                                     sample_discrete(states, cpd.values, size)))
        return sampled

    def rejection_sample(self, evidence=None, size=1):
        """
        Generates sample(s) from joint distribution of the bayesian network,
        given the evidence.

        Parameters
        ----------
        evidence: list of `pgmpy.factor.State` namedtuples
            None if no evidence
        size: int
            size of sample to be generated

        Returns
        -------
        sampled: pandas.DataFrame
            the generated samples

        Examples
        --------
        >>> from pgmpy.models.BayesianModel import BayesianModel
        >>> from pgmpy.factors.CPD import TabularCPD
        >>> from pgmpy.factors.Factor import State
        >>> from pgmpy.inference.Sampling import BayesianModelSampling
        >>> student = BayesianModel([('diff', 'grade'), ('intel', 'grade')])
        >>> cpd_d = TabularCPD('diff', 2, [[0.6], [0.4]])
        >>> cpd_i = TabularCPD('intel', 2, [[0.7], [0.3]])
        >>> cpd_g = TabularCPD('grade', 3, [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25,
        ...                0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
        ...                ['intel', 'diff'], [2, 2])
        >>> student.add_cpds(cpd_d, cpd_i, cpd_g)
        >>> inference = BayesianModelSampling(student)
        >>> evidence = [State(var='diff', state=0)]
        >>> inference.rejection_sample(evidence, 2)
                intel       diff       grade
        0  (intel, 0)  (diff, 0)  (grade, 1)
        1  (intel, 0)  (diff, 0)  (grade, 1)
        """
        if evidence is None:
            return self.forward_sample(size)
        sampled = DataFrame(columns=self.topological_order)
        prob = 1
        while len(sampled) < size:
            _size = int(((size - len(sampled)) / prob) * 1.5)
            _sampled = self.forward_sample(_size)
            for evid in evidence:
                _sampled = _sampled[_sampled.ix[:, evid.var] == evid]
            prob = max(len(_sampled) / _size, 0.01)
            sampled = sampled.append(_sampled)
        sampled.reset_index(inplace=True, drop=True)
        return sampled[:size]

    def likelihood_weighted_sample(self, evidence=None, size=1):
        """
        Generates weighted sample(s) from joint distribution of the bayesian
        network, that comply with the given evidence.
        'Probabilistic Graphical Model Principles and Techniques', Koller and
        Friedman, Algorithm 12.2 pp 493.

        Parameters
        ----------
        evidence: list of `pgmpy.factor.State` namedtuples
            None if no evidence
        size: int
            size of sample to be generated

        Returns
        -------
        sampled: pandas.DataFrame
            the generated samples with corresponding weights

        Examples
        --------
        >>> from pgmpy.factors.Factor import State
        >>> from pgmpy.models.BayesianModel import BayesianModel
        >>> from pgmpy.factors.CPD import TabularCPD
        >>> from pgmpy.inference.Sampling import BayesianModelSampling
        >>> student = BayesianModel([('diff', 'grade'), ('intel', 'grade')])
        >>> cpd_d = TabularCPD('diff', 2, [[0.6], [0.4]])
        >>> cpd_i = TabularCPD('intel', 2, [[0.7], [0.3]])
        >>> cpd_g = TabularCPD('grade', 3, [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25,
        ...         0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
        ...         ['intel', 'diff'], [2, 2])
        >>> student.add_cpds(cpd_d, cpd_i, cpd_g)
        >>> inference = BayesianModelSampling(student)
        >>> evidence = {'diff': State(var='diff',state=0)}
        >>> inference.likelihood_weighted_sample(evidence, 2)
                intel       diff       grade  _weight
        0  (intel, 0)  (diff, 0)  (grade, 1)      0.6
        1  (intel, 1)  (diff, 0)  (grade, 1)      0.6
        """
        sampled = DataFrame(index=range(size), columns=self.topological_order)
        sampled['_weight'] = np.ones(size)
        evidence_dict = {var: st for var, st in evidence}
        for node in self.topological_order:
            cpd = self.cpds[node]
            states = [st for var, st in cpd.variables[node]]
            if cpd.evidence:
                indices = [i for i, x in enumerate(self.topological_order) if x in cpd.evidence]
                evidence = sampled.values[:, [indices]].tolist()
                weights = list(map(lambda t: cpd.reduce(t[0], inplace=False).values, evidence))
                if node in evidence_dict:
                    sampled[node] = (State(node, evidence_dict[node]), ) * size
                    for i in range(size):
                        sampled.loc[i, '_weight'] *= weights[i][evidence_dict[node]]
                else:
                    sampled[node] = list(map(lambda t: State(node, t), sample_discrete(states, weights)))
            else:
                if node in evidence_dict:
                    sampled[node] = (State(node, evidence_dict[node]), ) * size
                    for i in range(size):
                        sampled.loc[i, '_weight'] *= cpd.values[evidence_dict[node]]
                else:
                    sampled[node] = list(map(lambda t: State(node, t),
                                         sample_discrete(states, cpd.values, size)))
        return sampled
