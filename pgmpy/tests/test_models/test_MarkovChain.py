#!/usr/bin/env python3
import numpy as np
import pytest
from pandas import DataFrame

from pgmpy.factors.discrete import State
from pgmpy.models import MarkovChain as MC


class TestMarkovChainInit:
    def test_init_bad_variables_type(self):
        # variables is non-iterable
        with pytest.raises(ValueError):
            MC(variables=123)
        # variables is a string
        with pytest.raises(ValueError):
            MC(variables="abc")

    def test_init_bad_card_type(self):
        # card is non-iterable
        with pytest.raises(ValueError):
            MC(card=123)
        # card is a string
        with pytest.raises(ValueError):
            MC(card="abc")

    def test_init_less_args(self):
        model = MC()
        assert model.variables == []
        assert model.cardinalities == {}
        assert model.transition_models == {}
        assert model.state is None

    def test_init_with_valid_args(self):
        """Test initialization with valid arguments."""
        variables = ["intel", "diff", "grade"]
        card = [3, 2, 3]
        start_state = [State("intel", 0), State("diff", 1), State("grade", 2)]
        model = MC(variables, card, start_state)
        assert model.variables == variables
        assert model.cardinalities == {"intel": 3, "diff": 2, "grade": 3}


class TestMarkovChainStartState:
    def test_set_start_state_list(self):
        model = MC(["b", "a"], [2, 2])
        model.set_start_state([State("a", 0), State("b", 1)])
        # State is reordered to match model.variables order
        assert model.state == [State("b", 1), State("a", 0)]

    def test_set_start_state_none(self):
        model = MC()
        model.state = "state"
        model.set_start_state(None)
        assert model.state is None


class TestMarkovChainCheckState:
    def test_check_state_bad_type(self):
        model = MC()
        # state is non-iterable
        with pytest.raises(ValueError):
            model._check_state(123)
        # state is a string
        with pytest.raises(ValueError):
            model._check_state("abc")

    def test_check_state_bad_vars(self):
        model = MC()
        # state_vars and model_vars differ
        with pytest.raises(ValueError):
            model._check_state([State(1, 2)])

    def test_check_state_bad_var_value(self):
        model = MC(["a"], [2])
        # value of variable >= cardinality
        with pytest.raises(ValueError):
            model._check_state([State("a", 3)])

    def test_check_state_success(self):
        model = MC(["a"], [2])
        assert model._check_state([State("a", 1)]) is True


class TestMarkovChainVariables:
    def test_add_variable_new(self):
        model = MC(["a"], [2])
        model.add_variable("p", 3)
        assert "p" in model.variables
        assert model.cardinalities["p"] == 3
        assert model.transition_models["p"] == {}

    def test_copy(self):
        model = MC(["a", "b"], [2, 2], [State("a", 0), State("b", 1)])
        model.add_transition_model("a", {0: {0: 0.1, 1: 0.9}, 1: {0: 0.2, 1: 0.8}})
        model.add_transition_model("b", {0: {0: 0.3, 1: 0.7}, 1: {0: 0.4, 1: 0.6}})
        copy = model.copy()

        assert isinstance(copy, MC)
        assert sorted(model.variables) == sorted(copy.variables)
        assert model.cardinalities == copy.cardinalities
        assert model.transition_models == copy.transition_models
        assert model.state == copy.state

        model.add_variable("p", 1)
        model.set_start_state([State("a", 0), State("b", 1), State("p", 0)])
        model.add_transition_model("p", {0: {0: 1}})

        assert sorted(model.variables) != sorted(copy.variables)
        assert sorted(["a", "b"]) == sorted(copy.variables)
        assert model.cardinalities != copy.cardinalities
        assert {"a": 2, "b": 2} == copy.cardinalities
        assert model.state != copy.state
        assert [State("a", 0), State("b", 1)] == copy.state
        assert model.transition_models != copy.transition_models
        assert len(copy.transition_models) == 2
        assert copy.transition_models["a"] == {
            0: {0: 0.1, 1: 0.9},
            1: {0: 0.2, 1: 0.8},
        }
        assert copy.transition_models["b"] == {
            0: {0: 0.3, 1: 0.7},
            1: {0: 0.4, 1: 0.6},
        }

    def test_add_variables_from(self):
        """Test adding multiple variables at once."""
        model = MC()
        model.add_variables_from(["a", "b", "c"], [2, 3, 4])
        assert model.variables == ["a", "b", "c"]
        assert model.cardinalities == {"a": 2, "b": 3, "c": 4}


class TestMarkovChainTransitionModel:
    def test_add_transition_model_bad_type(self):
        model = MC()
        grade_tm_matrix_bad = [[0.1, 0.5, 0.4], [0.2, 0.2, 0.6], "abc"]
        # if transition_model is not a dict or np.array
        with pytest.raises(ValueError):
            model.add_transition_model("var", 123)
        with pytest.raises(ValueError):
            model.add_transition_model("var", grade_tm_matrix_bad)

    def test_add_transition_model_bad_states(self):
        model = MC(["var"], [2])
        # transition for state=1 not defined
        transition_model = {0: {0: 0.1, 1: 0.9}}
        with pytest.raises(ValueError):
            model.add_transition_model("var", transition_model)

    def test_add_transition_model_bad_transition(self):
        model = MC(["var"], [2])
        # transition for state=1 is not a dict
        transition_model = {0: {0: 0.1, 1: 0.9}, 1: "abc"}
        with pytest.raises(ValueError):
            model.add_transition_model("var", transition_model)

    def test_add_transition_model_bad_probability(self):
        model = MC(["var"], [2])
        transition_model = {0: {0: -0.1, 1: 1.1}, 1: {0: 0.5, 1: 0.5}}
        with pytest.raises(ValueError):
            model.add_transition_model("var", transition_model)

    def test_add_transition_model_bad_probability_sum(self):
        model = MC(["var"], [2])
        # transition probabilities from state=0 do not sum to 1.0
        transition_model = {0: {0: 0.1, 1: 0.2}, 1: {0: 0.5, 1: 0.5}}
        with pytest.raises(ValueError):
            model.add_transition_model("var", transition_model)

    def test_add_transition_model_success(self):
        model = MC(["var"], [2])
        transition_model = {0: {0: 0.3, 1: 0.7}, 1: {0: 0.5, 1: 0.5}}
        model.add_transition_model("var", transition_model)
        assert model.transition_models["var"] == transition_model

    def test_transition_model_bad_matrix_dimension(self):
        model = MC(["var"], [2])
        transition_model = np.array([0.3, 0.7])
        # check for square dimension of the matrix
        with pytest.raises(ValueError):
            model.add_transition_model("var", transition_model)
        transition_model = np.array([[0.3, 0.6, 0.1], [0.3, 0.3, 0.4]])
        with pytest.raises(ValueError):
            model.add_transition_model("var", transition_model)

    def test_transition_model_dict_to_matrix(self):
        model = MC(["var"], [2])
        transition_model = {0: {0: 0.3, 1: 0.7}, 1: {0: 0.5, 1: 0.5}}
        transition_model_matrix = np.array([[0.3, 0.7], [0.5, 0.5]])
        model.add_transition_model("var", transition_model_matrix)
        assert model.transition_models["var"] == transition_model


class TestMarkovChainSampling:
    def test_sample(self):
        model = MC(["a", "b"], [2, 2])
        model.transition_models["a"] = {0: {0: 0.1, 1: 0.9}, 1: {0: 0.2, 1: 0.8}}
        model.transition_models["b"] = {0: {0: 0.3, 1: 0.7}, 1: {0: 0.4, 1: 0.6}}
        sample = model.sample(start_state=[State("a", 0), State("b", 1)], size=2)
        assert len(sample) == 2
        assert list(sample.columns) == ["a", "b"]
        assert list(sample.loc[0]) in [[0, 0], [0, 1], [1, 0], [1, 1]]
        assert list(sample.loc[1]) in [[0, 0], [0, 1], [1, 0], [1, 1]]

    def test_sample_less_arg(self):
        """Test sampling without explicit start_state uses current state."""
        model = MC(["a", "b"], [2, 2])
        model.state = [State("a", 0), State("b", 1)]
        model.transition_models["a"] = {0: {0: 0.1, 1: 0.9}, 1: {0: 0.2, 1: 0.8}}
        model.transition_models["b"] = {0: {0: 0.3, 1: 0.7}, 1: {0: 0.4, 1: 0.6}}
        sample = model.sample(size=1)
        assert len(sample) == 1
        assert list(sample.columns) == ["a", "b"]

    def test_prob_from_sample(self):
        """Test probability calculation from sample data."""
        model = MC(["a", "b"], [2, 2])
        # Add transition models
        model.add_transition_model("a", {0: {0: 0.5, 1: 0.5}, 1: {0: 0.5, 1: 0.5}})
        model.add_transition_model("b", {0: {0: 0.5, 1: 0.5}, 1: {0: 0.5, 1: 0.5}})
        sample = DataFrame(index=range(200), columns=["a", "b"])
        sample["a"] = [1] * 100 + [0] * 100
        sample["b"] = [0] * 100 + [1] * 100
        probabilities = model.prob_from_sample([State("a", 1), State("b", 0)])
        # Check that probabilities are calculated (returns one prob per variable, 100 samples)
        assert len(probabilities) == 100
        # Check that probabilities are reasonable values between 0 and 1
        assert all(0 <= p <= 1 for p in probabilities)


class TestMarkovChainStationarity:
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
        assert model.is_stationarity() is True

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
        assert model.is_stationarity(0.0002, None) is False


class TestMarkovChainGenerateSample:
    def test_generate_sample(self):
        """Test generating samples from the model."""
        model = MC(["a", "b"], [2, 2])
        model.transition_models["a"] = {0: {0: 0.1, 1: 0.9}, 1: {0: 0.2, 1: 0.8}}
        model.transition_models["b"] = {0: {0: 0.3, 1: 0.7}, 1: {0: 0.4, 1: 0.6}}
        gen = model.generate_sample(start_state=[State("a", 0), State("b", 1)], size=2)
        samples = [sample for sample in gen]
        assert len(samples) == 2

    def test_generate_sample_less_arg(self):
        """Test generating samples without explicit start state."""
        model = MC(["a", "b"], [2, 2])
        model.state = [State("a", 0), State("b", 1)]
        model.transition_models["a"] = {0: {0: 0.1, 1: 0.9}, 1: {0: 0.2, 1: 0.8}}
        model.transition_models["b"] = {0: {0: 0.3, 1: 0.7}, 1: {0: 0.4, 1: 0.6}}
        gen = model.generate_sample(size=2)
        samples = [sample for sample in gen]
        assert len(samples) == 2


class TestMarkovChainRandomState:
    def test_random_state(self):
        model = MC(["a", "b"], [2, 3])
        state = model.random_state()
        vars = [v for v, s in state]
        assert vars == ["a", "b"]
        assert state[0].state >= 0
        assert state[1].state >= 0
        assert state[0].state <= 1
        assert state[1].state <= 2
