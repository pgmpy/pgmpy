#!/usr/bin/env python3
import unittest
from unittest.mock import patch

from pgmpy.models import MarkovChainMonteCarlo as MCMC


class TestMarkovChainMonteCarlo(unittest.TestCase):
    def setUp(self):
        self.edges = [(-2, -2, 0.75), (-2, -1, 0.25), (-1, -2, 0.25), (-1, -1, 0.5), (-1, 0, 0.25),
                      (0, -1, 0.25), (0, 0, 0.5), (0, 1, 0.25), (1, 0, 0.25), (1, 1, 0.5), (1, 2, 0.25),
                      (2, 1, 0.25), (2, 2, 0.75)]
        self.nodes = [-2, -1, 0, 1, 2]
        self.edges_dict = {-2: {-2: {'weight': 0.75}, -1: {'weight': 0.25}},
                           -1: {-2: {'weight': 0.25}, -1: {'weight': 0.5}, 0: {'weight': 0.25}},
                           0: {-1: {'weight': 0.25}, 0: {'weight': 0.5}, 1: {'weight': 0.25}},
                           1: {0: {'weight': 0.25}, 1: {'weight': 0.5}, 2: {'weight': 0.25}},
                           2: {1: {'weight': 0.25}, 2: {'weight': 0.75}}}
        self.model = MCMC()


    def test_init(self):
        model = MCMC(self.edges, start_state=0)
        self.assertEquals(model.start_state, 0)
        self.assertDictEqual(model.edge, self.edges_dict)

    def test_init_no_args(self):
        model = MCMC()
        self.assertIsNone(model.start_state)
        self.assertDictEqual(model.edge, {})

    def test_init_wrong_type(self):
        with self.assertRaises(ValueError):
            model = MCMC(start_state=0)

    def test_add_node(self):
        self.model.add_node('some_node')
        self.assertIn('some_node', self.model.nodes())
        self.assertEquals(self.model._weights['some_node'], 0)

    @patch("pgmpy.models.MarkovChainMonteCarlo.reset_weights")
    def test_add_weighted_edges_from(self, reset_weights):
        self.model.add_weighted_edges_from([(1, 2, 0.3)])
        reset_weights.assert_called_once()
        self.assertEquals(self.model.edge[1][2]['weight'], 0.3)
        self.assertSetEqual(set(self.model._weights.values()), {0})

    @patch("pgmpy.models.MarkovChainMonteCarlo.reset_weights")
    def test_set_start_state(self, reset_weights):
        model = MCMC(self.edges)
        model.set_start_state(0)
        reset_weights.assert_called_once()
        self.assertEquals(model._weights[0], 1)
        self.assertEquals(model.start_state, 0)

    def test_set_start_state_failure(self):
        model = MCMC()
        with self.assertRaises(ValueError):
            model.set_start_state(0)

    def test_reset_weights(self):
        model = MCMC(self.edges)
        model.reset_weights()
        self.assertTrue(set(model._weights.values()).issubset({0}))
        self.assertIsNone(model.start_state)

    def test_check_markov_chain_true(self):
        model = MCMC(self.edges)
        self.assertTrue(model.check_markov_chain())

    def test_check_markov_chain_false_1(self):
        model = MCMC([(1, 1, -0.5)])
        self.assertFalse(model.check_markov_chain())

    def test_check_markov_chain_false_2(self):
        model = MCMC([(1, 2, 0.4), (1, 1, 0.5), (2, 2, 1)])
        self.assertFalse(model.check_markov_chain())

    def test_weights_getter(self):
        model = MCMC()
        model._weights = {1: 2, 3: 4, 5: 6}
        self.assertDictEqual(model.weights, {1: 2, 3: 4, 5: 6})

    def test_weights_setter_bad_type(self):
        model = MCMC()
        with self.assertRaises(ValueError):
            model.weights = 10

    def test_weights_setter_bad_keys(self):
        model = MCMC()
        with self.assertRaises(ValueError):
            model.weights = {'a': 0}

    def test_weights_setter_bad_values(self):
        model = MCMC()
        model.add_node('a')
        with self.assertRaises(ValueError):
            model.weights = {'a': 100}

    def test_weights_setter_basic(self):
        model = MCMC()
        model.add_node('a')
        model.weights = {'a': 0.5}
        self.assertDictEqual(model._weights, {'a': 0.5})

    def test_sample(self):
        self.model = MCMC(self.edges, 0)
        sample = self.model.sample(5)
        self.assertTrue(set(sample).issubset(set(self.nodes)))
        self.assertEquals(len(sample), 5)
