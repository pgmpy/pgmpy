#!/usr/bin/env python3
import sys
import unittest

import numpy as np
from mock import call, patch
from pandas import DataFrame

from pgmpy.factors.discrete import State
from pgmpy.models import MarkovChain as MC


class TestMarkovChain(unittest.TestCase):
    def setUp(self):
        self.variables = ["intel", "diff", "grade"]
        self.card = [3, 2, 3]
        self.cardinalities = {"intel": 3, "diff": 2, "grade": 3}
        self.intel_tm = {
            0: {0: 0.1, 1: 0.25, 2: 0.65},
            1: {0: 0.5, 1: 0.3, 2: 0.2},
            2: {0: 0.3, 1: 0.3, 2: 0.4},
        }
        self.intel_tm_matrix = np.array(
            [[0.1, 0.25, 0.65], [0.5, 0.3, 0.2], [0.3, 0.3, 0.4]]
        )
        self.diff_tm = {0: {0: 0.3, 1: 0.7}, 1: {0: 0.75, 1: 0.25}}
        self.diff_tm_matrix = np.array([[0.3, 0.7], [0.75, 0.25]])
        self.grade_tm = {
            0: {0: 0.4, 1: 0.2, 2: 0.4},
            1: {0: 0.9, 1: 0.05, 2: 0.05},
            2: {0: 0.1, 1: 0.4, 2: 0.5},
        }
        self.grade_tm_matrix = [[0.4, 0.2, 0.4], [0.9, 0.05, 0.05], [0.1, 0.4, 0.5]]
        self.start_state = [State("intel", 0), State("diff", 1), State("grade", 2)]
        self.model = MC()

        self.sample = DataFrame(index=range(200), columns=["a", "b"])
        self.sample.a = [1] * 100 + [0] * 100
        self.sample.b = [0] * 100 + [1] * 100

    def tearDown(self):
        del self.variables
        del self.card
        del self.cardinalities
        del self.intel_tm
        del self.diff_tm
        del self.grade_tm
        del self.start_state
        del self.model
        del self.sample

    @patch("pgmpy.models.MarkovChain._check_state", autospec=True)
    def test_init(self, check_state):
        model = MC(self.variables, self.card, self.start_state)
        self.assertListEqual(model.variables, self.variables)
        self.assertDictEqual(model.cardinalities, self.cardinalities)
        self.assertDictEqual(
            model.transition_models, {var: {} for var in self.variables}
        )
        check_state.assert_called_once_with(model, self.start_state)
        self.assertListEqual(model.state, self.start_state)

    def test_init_bad_variables_type(self):
        # variables is non-iterable
        self.assertRaises(ValueError, MC, variables=123)
        # variables is a string
        self.assertRaises(ValueError, MC, variables="abc")

    def test_init_bad_card_type(self):
        # card is non-iterable
        self.assertRaises(ValueError, MC, card=123)
        # card is a string
        self.assertRaises(ValueError, MC, card="abc")

    def test_init_less_args(self):
        model = MC()
        self.assertListEqual(model.variables, [])
        self.assertDictEqual(model.cardinalities, {})
        self.assertDictEqual(model.transition_models, {})
        self.assertIsNone(model.state)

    @patch("pgmpy.models.MarkovChain._check_state", autospec=True)
    def test_set_start_state_list(self, check_state):
        model = MC(["b", "a"], [1, 2])
        check_state.return_value = True
        model.set_start_state([State("a", 0), State("b", 1)])
        model_state = [State("b", 1), State("a", 0)]
        check_state.assert_called_once_with(model, model_state)
        self.assertEqual(model.state, model_state)

    def test_set_start_state_none(self):
        model = MC()
        model.state = "state"
        model.set_start_state(None)
        self.assertIsNone(model.state)

    def test_check_state_bad_type(self):
        model = MC()
        # state is non-iterable
        self.assertRaises(ValueError, model._check_state, 123)
        # state is a string
        self.assertRaises(ValueError, model._check_state, "abc")

    def test_check_state_bad_vars(self):
        model = MC()
        # state_vars and model_vars differ
        self.assertRaises(ValueError, model._check_state, [State(1, 2)])

    def test_check_state_bad_var_value(self):
        model = MC(["a"], [2])
        # value of variable >= cardinality
        self.assertRaises(ValueError, model._check_state, [State("a", 3)])

    def test_check_state_success(self):
        model = MC(["a"], [2])
        self.assertTrue(model._check_state([State("a", 1)]))

    def test_add_variable_new(self):
        model = MC(["a"], [2])
        model.add_variable("p", 3)
        self.assertIn("p", model.variables)
        self.assertEqual(model.cardinalities["p"], 3)
        self.assertDictEqual(model.transition_models["p"], {})

    def test_copy(self):
        model = MC(["a", "b"], [2, 2], [State("a", 0), State("b", 1)])
        model.add_transition_model("a", {0: {0: 0.1, 1: 0.9}, 1: {0: 0.2, 1: 0.8}})
        model.add_transition_model("b", {0: {0: 0.3, 1: 0.7}, 1: {0: 0.4, 1: 0.6}})
        copy = model.copy()

        self.assertIsInstance(copy, MC)
        self.assertEqual(sorted(model.variables), sorted(copy.variables))
        self.assertEqual(model.cardinalities, copy.cardinalities)
        self.assertEqual(model.transition_models, copy.transition_models)
        self.assertEqual(model.state, copy.state)

        model.add_variable("p", 1)
        model.set_start_state([State("a", 0), State("b", 1), State("p", 0)])
        model.add_transition_model("p", {0: {0: 1}})

        self.assertNotEqual(sorted(model.variables), sorted(copy.variables))
        self.assertEqual(sorted(["a", "b"]), sorted(copy.variables))
        self.assertNotEqual(model.cardinalities, copy.cardinalities)
        self.assertEqual({"a": 2, "b": 2}, copy.cardinalities)
        self.assertNotEqual(model.state, copy.state)
        self.assertEqual([State("a", 0), State("b", 1)], copy.state)
        self.assertNotEqual(model.transition_models, copy.transition_models)
        self.assertEqual(len(copy.transition_models), 2)
        self.assertEqual(
            copy.transition_models["a"], {0: {0: 0.1, 1: 0.9}, 1: {0: 0.2, 1: 0.8}}
        )
        self.assertEqual(
            copy.transition_models["b"], {0: {0: 0.3, 1: 0.7}, 1: {0: 0.4, 1: 0.6}}
        )

    @patch("pgmpy.models.MarkovChain.add_variable", autospec=True)
    def test_add_variables_from(self, add_var):
        model = MC()
        model.add_variables_from(self.variables, self.card)
        calls = [call(model, *p) for p in zip(self.variables, self.card)]
        add_var.assert_has_calls(calls)

    def test_add_transition_model_bad_type(self):
        model = MC()
        grade_tm_matrix_bad = [[0.1, 0.5, 0.4], [0.2, 0.2, 0.6], "abc"]
        # if transition_model is not a dict or np.array
        self.assertRaises(ValueError, model.add_transition_model, "var", 123)
        self.assertRaises(
            ValueError, model.add_transition_model, "var", grade_tm_matrix_bad
        )

    def test_add_transition_model_bad_states(self):
        model = MC(["var"], [2])
        # transition for state=1 not defined
        transition_model = {0: {0: 0.1, 1: 0.9}}
        self.assertRaises(
            ValueError, model.add_transition_model, "var", transition_model
        )

    def test_add_transition_model_bad_transition(self):
        model = MC(["var"], [2])
        # transition for state=1 is not a dict
        transition_model = {0: {0: 0.1, 1: 0.9}, 1: "abc"}
        self.assertRaises(
            ValueError, model.add_transition_model, "var", transition_model
        )

    def test_add_transition_model_bad_probability(self):
        model = MC(["var"], [2])
        transition_model = {0: {0: -0.1, 1: 1.1}, 1: {0: 0.5, 1: 0.5}}
        self.assertRaises(
            ValueError, model.add_transition_model, "var", transition_model
        )

    def test_add_transition_model_bad_probability_sum(self):
        model = MC(["var"], [2])
        # transition probabilities from state=0 do not sum to 1.0
        transition_model = {0: {0: 0.1, 1: 0.2}, 1: {0: 0.5, 1: 0.5}}
        self.assertRaises(
            ValueError, model.add_transition_model, "var", transition_model
        )

    def test_add_transition_model_success(self):
        model = MC(["var"], [2])
        transition_model = {0: {0: 0.3, 1: 0.7}, 1: {0: 0.5, 1: 0.5}}
        model.add_transition_model("var", transition_model)
        self.assertDictEqual(model.transition_models["var"], transition_model)

    def test_transition_model_bad_matrix_dimension(self):
        model = MC(["var"], [2])
        transition_model = np.array([0.3, 0.7])
        # check for square dimension of the matrix
        self.assertRaises(
            ValueError, model.add_transition_model, "var", transition_model
        )
        transition_model = np.array([[0.3, 0.6, 0.1], [0.3, 0.3, 0.4]])
        self.assertRaises(
            ValueError, model.add_transition_model, "var", transition_model
        )

    def test_transition_model_dict_to_matrix(self):
        model = MC(["var"], [2])
        transition_model = {0: {0: 0.3, 1: 0.7}, 1: {0: 0.5, 1: 0.5}}
        transition_model_matrix = np.array([[0.3, 0.7], [0.5, 0.5]])
        model.add_transition_model("var", transition_model_matrix)
        self.assertDictEqual(model.transition_models["var"], transition_model)

    def test_sample(self):
        model = MC(["a", "b"], [2, 2])
        model.transition_models["a"] = {0: {0: 0.1, 1: 0.9}, 1: {0: 0.2, 1: 0.8}}
        model.transition_models["b"] = {0: {0: 0.3, 1: 0.7}, 1: {0: 0.4, 1: 0.6}}
        sample = model.sample(start_state=[State("a", 0), State("b", 1)], size=2)
        self.assertEqual(len(sample), 2)
        self.assertEqual(list(sample.columns), ["a", "b"])
        self.assertTrue(list(sample.loc[0]) in [[0, 0], [0, 1], [1, 0], [1, 1]])
        self.assertTrue(list(sample.loc[1]) in [[0, 0], [0, 1], [1, 0], [1, 1]])

    @patch("pgmpy.models.MarkovChain.random_state", autospec=True)
    def test_sample_less_arg(self, random_state):
        model = MC(["a", "b"], [2, 2])
        random_state.return_value = [State("a", 0), State("b", 1)]
        sample = model.sample(size=1)
        random_state.assert_called_once_with(model)
        self.assertEqual(model.state, random_state.return_value)
        self.assertEqual(len(sample), 1)
        self.assertEqual(list(sample.columns), ["a", "b"])
        self.assertEqual(list(sample.loc[0]), [0, 1])

    @patch("pgmpy.models.MarkovChain.sample", autospec=True)
    def test_prob_from_sample(self, sample):
        model = MC(["a", "b"], [2, 2])
        sample.return_value = self.sample
        probabilities = model.prob_from_sample([State("a", 1), State("b", 0)])
        self.assertEqual(list(probabilities), [1] * 50 + [0] * 50)

    def test_is_stationarity_success(self):
        model = MC(["intel", "diff"], [2, 3])
        model.set_start_state([State("intel", 0), State("diff", 2)])
        intel_tm = {0: {0: 0.25, 1: 0.75}, 1: {0: 0.5, 1: 0.5}}
        model.add_transition_model("intel", intel_tm)
        diff_tm = {
            0: {0: 0.1, 1: 0.5, 2: 0.4},
            1: {0: 0.2, 1: 0.2, 2: 0.6},
            2: {0: 0.7, 1: 0.15, 2: 0.15},
        }
        model.add_transition_model("diff", diff_tm)
        self.assertTrue(model.is_stationarity)

    def test_is_stationarity_failure(self):
        model = MC(["intel", "diff"], [2, 3])
        model.set_start_state([State("intel", 0), State("diff", 2)])
        intel_tm = {0: {0: 0.25, 1: 0.75}, 1: {0: 0.5, 1: 0.5}}
        model.add_transition_model("intel", intel_tm)
        diff_tm = {
            0: {0: 0.1, 1: 0.5, 2: 0.4},
            1: {0: 0.2, 1: 0.2, 2: 0.6},
            2: {0: 0.7, 1: 0.15, 2: 0.15},
        }
        model.add_transition_model("diff", diff_tm)
        self.assertFalse(model.is_stationarity(0.002, None))

    @patch.object(sys.modules["pgmpy.models.MarkovChain"], "sample_discrete")
    def test_generate_sample(self, sample_discrete):
        model = MC(["a", "b"], [2, 2])
        model.transition_models["a"] = {0: {0: 0.1, 1: 0.9}, 1: {0: 0.2, 1: 0.8}}
        model.transition_models["b"] = {0: {0: 0.3, 1: 0.7}, 1: {0: 0.4, 1: 0.6}}
        sample_discrete.side_effect = [[1], [0]] * 2
        gen = model.generate_sample(start_state=[State("a", 0), State("b", 1)], size=2)
        samples = [sample for sample in gen]
        expected_samples = [[State("a", 1), State("b", 0)]] * 2
        self.assertEqual(samples, expected_samples)

    @patch.object(sys.modules["pgmpy.models.MarkovChain"], "sample_discrete")
    @patch("pgmpy.models.MarkovChain.random_state", autospec=True)
    def test_generate_sample_less_arg(self, random_state, sample_discrete):
        model = MC(["a", "b"], [2, 2])
        model.transition_models["a"] = {0: {0: 0.1, 1: 0.9}, 1: {0: 0.2, 1: 0.8}}
        model.transition_models["b"] = {0: {0: 0.3, 1: 0.7}, 1: {0: 0.4, 1: 0.6}}
        random_state.return_value = [State("a", 0), State("b", 1)]
        sample_discrete.side_effect = [[1], [0]] * 2
        gen = model.generate_sample(size=2)
        samples = [sample for sample in gen]
        expected_samples = [[State("a", 1), State("b", 0)]] * 2
        self.assertEqual(samples, expected_samples)

    def test_random_state(self):
        model = MC(["a", "b"], [2, 3])
        state = model.random_state()
        vars = [v for v, s in state]
        self.assertEqual(vars, ["a", "b"])
        self.assertGreaterEqual(state[0].state, 0)
        self.assertGreaterEqual(state[1].state, 0)
        self.assertLessEqual(state[0].state, 1)
        self.assertLessEqual(state[1].state, 2)
