import numpy as np
import pytest

from pgmpy.utils.mathext import (
    _adjusted_weights,
    cartesian,
    powerset,
    sample_discrete,
    sample_discrete_maps,
)


class TestMathext:
    def test_cartesian(self):
        arrays = ([1, 2, 3], [4, 5], [6, 7])
        expected = np.array(
            [
                [1, 4, 6],
                [1, 4, 7],
                [1, 5, 6],
                [1, 5, 7],
                [2, 4, 6],
                [2, 4, 7],
                [2, 5, 6],
                [2, 5, 7],
                [3, 4, 6],
                [3, 4, 7],
                [3, 5, 6],
                [3, 5, 7],
            ]
        )
        result = cartesian(arrays)
        np.testing.assert_array_equal(result, expected)

    def test_cartesian_with_out(self):
        arrays = ([1, 2], [3, 4])
        out = np.zeros((4, 2), dtype=int)
        expected = np.array([[1, 3], [1, 4], [2, 3], [2, 4]])
        result = cartesian(arrays, out=out)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(out, expected)

    def test_adjusted_weights_exact_sum(self):
        weights = np.array([0.2, 0.3, 0.5])
        result = _adjusted_weights(weights.copy())
        np.testing.assert_array_almost_equal(result, weights)

    def test_adjusted_weights_minor_roundoff(self):
        # Slightly off but within normal roundoff tolerance
        weights = np.array([0.3333333333333333, 0.3333333333333333, 0.3333333333333333])
        result = _adjusted_weights(weights.copy())
        assert np.isclose(result.sum(), 1.0)

    def test_adjusted_weights_warning(self):
        # Off by more than float tolerance, less than 1e-3
        weights = np.array([0.1111111] * 9)  # sum = 0.9999999, off by 1e-7
        with pytest.warns(UserWarning, match="Probability values sum to"):
            result = _adjusted_weights(weights.copy())
            assert np.isclose(result.sum(), 1.0)

    def test_adjusted_weights_value_error(self):
        # Off by more than 1e-3
        weights = np.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="The probability values do not sum to 1."):
            _adjusted_weights(weights)

    def test_sample_discrete_1d(self):
        values = np.array(["v_0", "v_1", "v_2"])
        weights = np.array([0.2, 0.5, 0.3])
        samples = sample_discrete(values, weights, size=10, seed=42)
        assert len(samples) == 10
        assert all(s in values for s in samples)

    def test_sample_discrete_2d(self):
        values = np.array([0, 1])
        weights = np.array([[0.2, 0.8], [0.8, 0.2], [0.5, 0.5]])
        # When passing 2D weights, size should equal len(weights)
        samples = sample_discrete(values, weights, size=3, seed=42)
        assert len(samples) == 3

        # Depending on numpy type inference, samples could be string or int
        # if string array is not created, we can just check size
        assert samples.shape == (3,)

    def test_sample_discrete_maps(self):
        states = np.array([0, 1])
        weight_indices = np.array([0, 1, 0, 1, 1])
        index_to_weight = {0: np.array([0.1, 0.9]), 1: np.array([0.8, 0.2])}
        samples = sample_discrete_maps(states, weight_indices, index_to_weight, size=5, seed=42)
        assert len(samples) == 5
        assert all(s in states for s in samples)

    def test_powerset(self):
        l_input = [1, 2, 3]
        expected = [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
        result = list(powerset(l_input))
        assert result == expected
