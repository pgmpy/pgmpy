import unittest

import numpy as np
import pandas as pd

from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.inference.CausalInference import CausalInference


class TestCausalGraphMethods(unittest.TestCase):
    def setUp(self):
        self.game = BayesianModel(
            [("A", "X"), ("A", "B"), ("C", "B"), ("C", "Y"), ("X", "Y"), ("B", "X")]
        )
        self.inference = CausalInference(self.game)

    def test_is_d_separated(self):
        self.assertFalse(self.inference._is_d_separated("X", "Y", Z=None))
        self.assertTrue(self.inference._is_d_separated("B", "Y", Z=("C", "X")))

    def test_backdoor_validation(self):
        self.inference.is_valid_backdoor_adjustment_set("X", "Y", Z="C")


class TestBackdoorPaths(unittest.TestCase):
    """
    These tests are drawn from games presented in The Book of Why by Judea Pearl. See the Jupyter Notebook called
    Causal Games in the examples folder for further explanation about each of these.
    """

    def test_game1(self):
        game1 = BayesianModel([("X", "A"), ("A", "Y"), ("A", "B")])
        inference = CausalInference(game1)
        self.assertTrue(inference.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(deconfounders, frozenset())

    def test_game2(self):
        game2 = BayesianModel(
            [
                ("X", "E"),
                ("E", "Y"),
                ("A", "B"),
                ("A", "X"),
                ("B", "C"),
                ("D", "B"),
                ("D", "E"),
            ]
        )
        inference = CausalInference(game2)
        self.assertTrue(inference.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(deconfounders, frozenset())

    def test_game3(self):
        game3 = BayesianModel(
            [("X", "Y"), ("X", "A"), ("B", "A"), ("B", "Y"), ("B", "X")]
        )
        inference = CausalInference(game3)
        self.assertFalse(inference.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(deconfounders, frozenset({frozenset({"B"})}))

    def test_game4(self):
        game4 = BayesianModel([("A", "X"), ("A", "B"), ("C", "B"), ("C", "Y")])
        inference = CausalInference(game4)
        self.assertTrue(inference.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(deconfounders, frozenset())

    def test_game5(self):
        game5 = BayesianModel(
            [("A", "X"), ("A", "B"), ("C", "B"), ("C", "Y"), ("X", "Y"), ("B", "X")]
        )
        inference = CausalInference(game5)
        self.assertFalse(inference.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(
            deconfounders, frozenset({frozenset({"C"}), frozenset({"A", "B"})})
        )

    def test_game6(self):
        game6 = BayesianModel(
            [
                ("X", "F"),
                ("C", "X"),
                ("A", "C"),
                ("A", "D"),
                ("B", "D"),
                ("B", "E"),
                ("D", "X"),
                ("D", "Y"),
                ("E", "Y"),
                ("F", "Y"),
            ]
        )
        inference = CausalInference(game6)
        self.assertFalse(inference.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(
            deconfounders,
            frozenset(
                {
                    frozenset({"C", "D"}),
                    frozenset({"A", "D"}),
                    frozenset({"D", "E"}),
                    frozenset({"B", "D"}),
                }
            ),
        )


class TestEstimator(unittest.TestCase):
    def test_create_estimator(self):
        game1 = BayesianModel([("X", "A"), ("A", "Y"), ("A", "B")])
        data = pd.DataFrame(
            np.random.randint(2, size=(1000, 4)), columns=["X", "A", "B", "Y"]
        )
        inference = CausalInference(model=game1)
        ate = inference.estimate_ate("X", "Y", data=data, estimator_type="linear")
        self.assertAlmostEqual(ate, 0, places=0)
