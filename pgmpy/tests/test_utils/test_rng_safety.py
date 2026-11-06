import unittest

import numpy as np

from pgmpy.factors.discrete import State
from pgmpy.models import MarkovChain
from pgmpy.utils import sample_discrete


class TestRNGSafety(unittest.TestCase):
    def test_sample_discrete_safety(self):
        # Set global seed
        np.random.seed(42)  # noqa: NPY002
        _ = np.random.random()  # noqa: S311

        # Call sample_discrete with its own rng
        sample_discrete(["a", "b"], [0.5, 0.5], size=10, rng=123)

        # Check if global state was affected
        v2 = np.random.random()  # noqa: S311

        # If the rng resets global state, v2 will be the first value from rng 123.
        # If the rng is isolated, v2 will be the second value from seed 42.
        np.random.seed(42)  # noqa: NPY002
        _ = np.random.random()  # noqa: S311
        expected_v2 = np.random.random()  # noqa: S311

        self.assertEqual(v2, expected_v2, "Global RNG state was polluted by sample_discrete!")

    def test_sample_discrete_seed_deprecated_safety(self):
        # Set global seed
        np.random.seed(42)  # noqa: NPY002
        _ = np.random.random()  # noqa: S311

        # Call sample_discrete with legacy seed kwarg
        sample_discrete(["a", "b"], [0.5, 0.5], size=10, seed=123)

        v2 = np.random.random()  # noqa: S311

        np.random.seed(42)  # noqa: NPY002
        _ = np.random.random()  # noqa: S311
        expected_v2 = np.random.random()  # noqa: S311

        self.assertEqual(v2, expected_v2, "Global RNG state was polluted by sample_discrete with seed!")

    def test_markov_chain_safety(self):
        model = MarkovChain(variables=["A", "B"], card=[2, 2])
        tm = {0: {0: 0.1, 1: 0.9}, 1: {0: 0.8, 1: 0.2}}
        model.add_transition_model("A", tm)
        model.add_transition_model("B", tm)

        np.random.seed(42)  # noqa: NPY002
        _ = np.random.random()  # noqa: S311

        # Call sample with its own rng (without start state)
        model.sample(size=5, rng=123)

        # Call sample with an existing start state
        model.set_start_state([State("A", 0), State("B", 1)])
        model.sample(size=5, rng=456)

        # Call generate_sample with its own rng
        _ = list(model.generate_sample(size=5, rng=789))

        # Call sample with legacy seed kwarg
        model.sample(size=5, seed=123)

        v2 = np.random.random()  # noqa: S311

        np.random.seed(42)  # noqa: NPY002
        _ = np.random.random()  # noqa: S311
        expected_v2 = np.random.random()  # noqa: S311

        self.assertEqual(v2, expected_v2, "Global RNG state was polluted by MarkovChain.sample!")
