import unittest

import numpy as np

from pgmpy.models import MarkovChain
from pgmpy.utils import sample_discrete


class TestRNGSafety(unittest.TestCase):
    def test_sample_discrete_safety(self):
        # Set global seed
        np.random.seed(42)  # noqa: NPY002
        v1 = np.random.random()  # noqa: S311

        # Call sample_discrete with its own seed
        sample_discrete(["a", "b"], [0.5, 0.5], size=10, seed=123)

        # Check if global state was affected
        v2 = np.random.random()  # noqa: S311

        # If the seed resets global state, v2 will be the first value from seed 123.
        # If the seed is isolated, v2 will be the second value from seed 42.
        np.random.seed(42)  # noqa: NPY002
        _ = np.random.random()  # noqa: S311
        expected_v2 = np.random.random()  # noqa: S311

        self.assertEqual(v2, expected_v2, "Global RNG state was polluted by sample_discrete!")

    def test_markov_chain_safety(self):
        model = MarkovChain(variables=["A", "B"], card=[2, 2])
        tm = {0: {0: 0.1, 1: 0.9}, 1: {0: 0.8, 1: 0.2}}
        model.add_transition_model("A", tm)
        model.add_transition_model("B", tm)

        np.random.seed(42)  # noqa: NPY002
        v1 = np.random.random()  # noqa: S311

        # Call sample with its own seed
        model.sample(size=5, seed=123)

        v2 = np.random.random()  # noqa: S311

        np.random.seed(42)  # noqa: NPY002
        _ = np.random.random()  # noqa: S311
        expected_v2 = np.random.random()  # noqa: S311

        self.assertEqual(v2, expected_v2, "Global RNG state was polluted by MarkovChain.sample!")


if __name__ == "__main__":
    unittest.main()
