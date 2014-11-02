#!/usr/bin/env python3
import unittest
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.models import MarkovModel
from pgmpy.factors import Factor
from pgmpy.Inference import Inference


class TestInferenceBase(unittest.TestCase):
    def setUp(self):
        self.bayesian = BayesianModel([('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')])
        self.bayesian.set_cpd('a', np.array([[0.4, 0.6]]))
        self.bayesian.set_cpd('b', np.array([[0.2, 0.4], [0.3, 0.4]]))
        self.bayesian.set_cpd('c', np.array([[0.1, 0.2], [0.3, 0.4]]))
        self.bayesian.set_cpd('d', np.array([[0.4, 0.3], [0.2, 0.1]]))
        self.bayesian.set_cpd('e', np.array([[0.3, 0.2], [0.4, 0.1]]))

        self.markov = MarkovModel([('a', 'b'), ('b', 'd'), ('a', 'c'), ('c', 'd')])
        self.markov.add_factors(Factor(['a', 'b'], [2, 2], np.array([100, 1, 1, 100])))
        self.markov.add_factors(Factor(['a', 'c'], [2, 2], np.array([40, 30, 100, 20])))
        self.markov.add_factors(Factor(['b', 'd'], [2, 2], np.array([1, 100, 100, 1])))
        self.markov.add_factors(Factor(['c', 'd'], [2, 2], np.array([60, 60, 40, 40])))

    def test_inference_init(self):
        infer_bayesian = Inference(self.bayesian)
        self.assertEqual(infer_bayesian.variables, ['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(infer_bayesian.factors, {'a': [self.bayesian.get_cpd('a'), self.bayesian.get_cpd('b')],
                                                  'b': [self.bayesian.get_cpd('b'), self.bayesian.get_cpd('c')],
                                                  'c': [self.bayesian.get_cpd('c'), self.bayesian.get_cpd('d')],
                                                  'd': [self.bayesian.get_cpd('d'), self.bayesian.get_cpd('e')],
                                                  'e': [self.bayesian.get_cpd('e')]})

        infer_markov = Inference(self.markov)
        self.assertEqual(infer_markov.variables, ['a', 'b', 'c', 'd'])
        self.assertEqual(infer_markov.factors, {'a': [Factor(['a', 'b'], [2, 2], np.array([100, 1, 1, 100])),
                                                      Factor(['a', 'c'], [2, 2], np.array([40, 30, 100, 20]))],
                                                'b': [Factor(['a', 'b'], [2, 2], np.array([100, 1, 1, 100])),
                                                      Factor(['b', 'd'], [2, 2], np.array([1, 100, 100, 1]))],
                                                'c': [Factor(['a', 'c'], [2, 2], np.array([40, 30, 100, 20])),
                                                      Factor(['c', 'd'], [2, 2], np.array([60, 60, 40, 40]))],
                                                'd': [Factor(['b', 'd'], [2, 2], np.array([1, 100, 100, 1])),
                                                      Factor(['c', 'd'], [2, 2], np.array([60, 60, 40, 40]))]})
