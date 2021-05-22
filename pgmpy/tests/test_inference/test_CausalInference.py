import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd

from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference.CausalInference import CausalInference


class TestCausalGraphMethods(unittest.TestCase):
    def setUp(self):
        self.game = BayesianModel(
            [("A", "X"), ("A", "B"), ("C", "B"), ("C", "Y"), ("X", "Y"), ("B", "X")]
        )
        self.inference = CausalInference(self.game)

    def test_is_d_separated(self):
        self.assertTrue(self.inference.model.is_dconnected("X", "Y", observed=None))
        self.assertFalse(
            self.inference.model.is_dconnected("B", "Y", observed=("C", "X"))
        )

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


class TestDoQuery(unittest.TestCase):
    def setUp(self):
        self.simpson_model = BayesianModel([("S", "T"), ("T", "C"), ("S", "C")])
        cpd_s = TabularCPD(
            variable="S",
            variable_card=2,
            values=[[0.5], [0.5]],
            state_names={"S": ["m", "f"]},
        )
        cpd_t = TabularCPD(
            variable="T",
            variable_card=2,
            values=[[0.25, 0.75], [0.75, 0.25]],
            evidence=["S"],
            evidence_card=[2],
            state_names={"S": ["m", "f"], "T": [0, 1]},
        )
        cpd_c = TabularCPD(
            variable="C",
            variable_card=2,
            values=[[0.3, 0.4, 0.7, 0.8], [0.7, 0.6, 0.3, 0.2]],
            evidence=["S", "T"],
            evidence_card=[2, 2],
            state_names={"S": ["m", "f"], "T": [0, 1], "C": [0, 1]},
        )

        self.simpson_model.add_cpds(cpd_s, cpd_t, cpd_c)
        self.infer = CausalInference(self.simpson_model)

    def test_query(self):
        for algo in ["ve", "bp"]:
            query_nodo1 = self.infer.query(
                variables=["C"], do=None, evidence={"T": 1}, inference_algo=algo
            )
            np_test.assert_array_almost_equal(query_nodo1.values, np.array([0.5, 0.5]))

            query_nodo2 = self.infer.query(
                variables=["C"], do=None, evidence={"T": 0}, inference_algo=algo
            )
            np_test.assert_array_almost_equal(query_nodo2.values, np.array([0.6, 0.4]))

            query1 = self.infer.query(variables=["C"], do={"T": 1}, inference_algo=algo)
            np_test.assert_array_almost_equal(query1.values, np.array([0.6, 0.4]))

            query2 = self.infer.query(variables=["C"], do={"T": 0}, inference_algo=algo)
            np_test.assert_array_almost_equal(query2.values, np.array([0.5, 0.5]))

            query_evi1 = self.infer.query(
                variables=["C"], do={"T": 1}, evidence={"S": "m"}, inference_algo=algo
            )
            np_test.assert_array_almost_equal(query_evi1.values, np.array([0.4, 0.6]))

            query_evi2 = self.infer.query(
                variables=["C"], do={"T": 0}, evidence={"S": "m"}, inference_algo=algo
            )
            np_test.assert_array_almost_equal(query_evi2.values, np.array([0.3, 0.7]))

    def test_query_error(self):
        self.assertRaises(ValueError, self.infer.query, variables="C", do={"T": 1})
        self.assertRaises(ValueError, self.infer.query, variables=["E"], do={"T": 1})
        self.assertRaises(ValueError, self.infer.query, variables=["C"], do="T")
        self.assertRaises(
            ValueError, self.infer.query, variables=["C"], do={"T": 1}, evidence="S"
        )
        self.assertRaises(
            ValueError,
            self.infer.query,
            variables=["C"],
            do={"T": 1},
            inference_algo="random",
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
