#!/usr/bin/env python3
import unittest
from unittest.mock import patch, call

from pgmpy.factors import State
from pgmpy.models import MarkovChain as MC


class TestMarkovChain(unittest.TestCase):
    def setUp(self):
        self.variables = ['intel', 'diff', 'grade']
        self.card = [3, 2, 3]
        self.cardinalities = {'intel': 3, 'diff': 2, 'grade': 3}
        self.intel_tm = {0: {0: 0.1, 1: 0.25, 2: 0.65}, 1: {0: 0.5, 1: 0.3, 2: 0.2}, 2: {0: 0.3, 1: 0.3, 2: 0.4}}
        self.diff_tm = {0: {0: 0.3, 1: 0.7}, 1: {0: 0.75, 1: 0.25}}
        self.grade_tm = {0: {0: 0.4, 1: 0.2, 2: 0.4}, 1: {0: 0.9, 1: 0.05, 2: 0.05}, 2: {0: 0.1, 1: 0.4, 2: 0.5}}
        self.start_state = [State('intel', 0), State('diff', 1), State('grade', 2)]
        self.model = MC()

    def tearDown(self):
        del self.variables
        del self.card
        del self.cardinalities
        del self.intel_tm
        del self.diff_tm
        del self.grade_tm
        del self.start_state
        del self.model

    @patch("pgmpy.models.MarkovChain._check_state", autospec=True)
    def test_init(self, check_state):
        model = MC(self.variables, self.card, self.start_state)
        self.assertListEqual(model.variables, self.variables)
        self.assertDictEqual(model.cardinalities, self.cardinalities)
        self.assertDictEqual(model.transition_models, {var: {} for var in self.variables})
        check_state.assert_called_once_with(model, self.start_state)
        self.assertListEqual(model.state, self.start_state)

    def test_init_bad_args1(self):
        self.assertRaises(ValueError, MC, variables=123)

    def test_init_bad_args2(self):
        self.assertRaises(ValueError, MC, card=123)

    def test_init_less_args(self):
        model = MC()
        self.assertListEqual(model.variables, [])
        self.assertDictEqual(model.cardinalities, {})
        self.assertDictEqual(model.transition_models, {})
        self.assertIsNone(model.state)

    @patch("pgmpy.models.MarkovChain._check_state", autospec=True)
    def test_set_start_state_list(self, check_state):
        model = MC(['b', 'a'], [1, 2])
        check_state.return_value = True
        model.set_start_state([State('a', 0), State('b', 1)])
        model_state = [State('b', 1), State('a', 0)]
        check_state.assert_called_once_with(model, model_state)
        self.assertEqual(model.state, model_state)

    def test_set_start_state_none(self):
        model = MC()
        model.state = 'state'
        model.set_start_state(None)
        self.assertIsNone(model.state)

    def test_check_state_fail1(self):
        model = MC()
        self.assertRaises(ValueError, model._check_state, 123)

    def test_check_state_fail2(self):
        model = MC()
        self.assertRaises(ValueError, model._check_state, [State(1, 2)])

    def test_check_state_fail3(self):
        model = MC(['a'], [2])
        self.assertRaises(ValueError, model._check_state, [State('a', 3)])

    def test_check_state(self):
        model = MC(['a'], [2])
        self.assertTrue(model._check_state([State('a', 1)]))

    def test_add_variable(self):
        model = MC(['a'], [2])
        model.add_variable('p', 3)
        self.assertIn('p', model.variables)
        self.assertEqual(model.cardinalities['p'], 3)
        self.assertDictEqual(model.transition_models['p'], {})

    @patch("pgmpy.models.MarkovChain.add_variable", autospec=True)
    def test_add_variables_from(self, add_var):
        model = MC()
        model.add_variables_from(self.variables, self.card)
        calls = [call(model, *p) for p in zip(self.variables, self.card)]
        add_var.assert_has_calls(calls)

    def test_add_transition_model_fail1(self):
        model = MC()
        self.assertRaises(ValueError, model.add_transition_model, 'var', 123)

    def test_add_transition_model_fail2(self):
        model = MC(['var'], [2])
        transition_model = {0: {0.1, 0.9}}
        self.assertRaises(ValueError, model.add_transition_model, 'var', transition_model)

    def test_add_transition_model_fail3(self):
        pass
