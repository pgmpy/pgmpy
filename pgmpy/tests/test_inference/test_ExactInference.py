#!/usr/bin/env python3

import unittest
import numpy as np
import numpy.testing as np_test
from pgmpy.models import BayesianModel
from pgmpy.models import MarkovModel
from pgmpy.factors import TabularCPD
from pgmpy.factors import Factor


class TestVariableElimination(unittest.TestCase):
    def setUp(self):
        self.bayesian = BayesianModel([('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')])
        cpd_a = TabularCPD('a', 2, [[0.4, 0.6]])
        cpd_b = TabularCPD('b', 2, [[0.2, 0.4], [0.8, 0.6]], evidence=['a'], evidence_card=[2])
        cpd_c = TabularCPD('c', 2, [[0.3, 0.4], [0.7, 0.6]], evidence=['b'], evidence_card=[2])
        cpd_d = TabularCPD('d', 2, [[0.7, 0.5], [0.3, 0.5]], evidence=['c'], evidence_card=[2])
        cpd_e = TabularCPD('e', 2, [[0.5, 0.1], [0.5, 0.9]], evidence=['d'], evidence_card=[2])

        self.bayesian.add_cpd([cpd_a, cpd_b, cpd_c, cpd_d, cpd_e])

        self.markov = MarkovModel([('a', 'b'), ('b', 'c'), ('c', 'd'), ('a', 'd')])
        factor_a_b = Factor(['a', 'b'], [2, 2], [100, 40, 40, 100])
        factor_b_c = Factor(['b', 'c'], [2, 2], [60, 20, 20, 40])
        factor_c_d = Factor(['c', 'd'], [2, 2], [80, 50, 50, 60])
        factor_d_a = Factor(['d', 'a'], [2, 2], [10, 90, 40 ,20])

        self.markov.add_factors(factor_a_b, factor_b_c, factor_c_d, factor_d_a)

