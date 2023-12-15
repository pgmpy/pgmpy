import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference.CausalInference import CausalInference
from pgmpy.models import BayesianNetwork

np.random.seed(42)


class TestCausalGraphMethods(unittest.TestCase):
    def setUp(self):
        self.game = BayesianNetwork(
            [("A", "X"), ("A", "B"), ("C", "B"), ("C", "Y"), ("X", "Y"), ("B", "X")]
        )
        self.inference = CausalInference(self.game)

        self.dag_bd1 = BayesianNetwork([("X", "Y"), ("Z1", "X"), ("Z1", "Y")])
        self.inference_bd = CausalInference(self.dag_bd1)

        self.dag_bd2 = BayesianNetwork(
            [("X", "Y"), ("Z1", "X"), ("Z1", "Z2"), ("Z2", "Y")]
        )
        self.inference_bd2 = CausalInference(self.dag_bd2)

    def test_is_d_separated(self):
        self.assertTrue(self.inference.model.is_dconnected("X", "Y", observed=None))
        self.assertFalse(
            self.inference.model.is_dconnected("B", "Y", observed=("C", "X"))
        )

    def test_backdoor_validation(self):
        self.assertTrue(
            self.inference.is_valid_backdoor_adjustment_set("X", "Y", Z="C")
        )

        # Z accepts str or set[str]
        self.assertTrue(
            self.inference_bd.is_valid_backdoor_adjustment_set("X", "Y", Z="Z1")
        )
        self.assertTrue(
            self.inference_bd2.is_valid_backdoor_adjustment_set(
                "X", "Y", Z={"Z1", "Z2"}
            )
        )


class TestAdjustmentSet(unittest.TestCase):
    def setUp(self):
        # Model example taken from Constructing Separators and Adjustment Sets
        # in Ancestral Graphs UAI 2014.
        self.model = BayesianNetwork(
            [("x1", "y1"), ("x1", "z1"), ("z1", "z2"), ("z2", "x2"), ("y2", "z2")]
        )
        self.infer = CausalInference(self.model)

    def test_proper_backdoor_graph_error(self):
        self.assertRaises(
            ValueError, self.infer.get_proper_backdoor_graph, X=["x3"], Y=["y1", "y2"]
        )
        self.assertRaises(
            ValueError, self.infer.get_proper_backdoor_graph, X=["x2"], Y=["y1", "y3"]
        )
        self.assertRaises(
            ValueError,
            self.infer.get_proper_backdoor_graph,
            X=["x3", "x2"],
            Y=["y1", "y3"],
        )

    def test_proper_backdoor_graph(self):
        bd_graph = self.infer.get_proper_backdoor_graph(X=["x1", "x2"], Y=["y1", "y2"])
        self.assertTrue(("x1", "y1") not in bd_graph.edges())
        self.assertEqual(len(bd_graph.edges()), 4)
        self.assertTrue(
            set(bd_graph.edges()),
            set([("x1", "z1"), ("z1", "z2"), ("z2", "x2"), ("y2", "z2")]),
        )

    def test_is_valid_adjustment_set(self):
        self.assertTrue(
            self.infer.is_valid_adjustment_set(
                X=["x1", "x2"], Y=["y1", "y2"], adjustment_set=["z1", "z2"]
            )
        )

        self.assertFalse(
            self.infer.is_valid_adjustment_set(
                X=["x1", "x2"], Y=["y1", "y2"], adjustment_set=["z1"]
            )
        )

        self.assertTrue(
            self.infer.is_valid_adjustment_set(
                X=["x1", "x2"], Y=["y1", "y2"], adjustment_set=["z2"]
            )
        )

    def test_get_minimal_adjustment_set(self):
        # Without latent variables
        dag1 = BayesianNetwork([("X", "Y"), ("Z", "X"), ("Z", "Y")])
        infer = CausalInference(dag1)
        adj_set = infer.get_minimal_adjustment_set(X="X", Y="Y")
        self.assertEqual(adj_set, {"Z"})

        self.assertRaises(ValueError, infer.get_minimal_adjustment_set, X="W", Y="Y")

        # M graph
        dag2 = BayesianNetwork(
            [("X", "Y"), ("Z1", "X"), ("Z1", "Z3"), ("Z2", "Z3"), ("Z2", "Y")]
        )
        infer = CausalInference(dag2)
        adj_set = infer.get_minimal_adjustment_set(X="X", Y="Y")
        self.assertEqual(adj_set, set())

        # With latents
        dag_lat1 = BayesianNetwork([("X", "Y"), ("Z", "X"), ("Z", "Y")], latents={"Z"})
        infer = CausalInference(dag_lat1)
        adj_set = infer.get_minimal_adjustment_set(X="X", Y="Y")
        self.assertIsNone(adj_set)

        # Pearl's Simpson machine
        dag_lat2 = BayesianNetwork(
            [
                ("X", "Y"),
                ("Z1", "U"),
                ("U", "X"),
                ("Z1", "Z3"),
                ("Z3", "Y"),
                ("U", "Z2"),
                ("Z3", "Z2"),
            ],
            latents={"U"},
        )
        infer = CausalInference(dag_lat2)
        adj_set = infer.get_minimal_adjustment_set(X="X", Y="Y")
        self.assertTrue((adj_set == {"Z1"}) or (adj_set == {"Z3"}))

    def test_issue_1710(self):
        dag = BayesianNetwork([("X_1", "X_2"), ("Z", "X_1"), ("Z", "X_2")])
        infer = CausalInference(dag)
        adj_set = infer.get_minimal_adjustment_set("X_1", "X_2")

        self.assertEqual(adj_set, {"Z"})
        self.assertRaises(ValueError, infer.get_minimal_adjustment_set, X="X_3", Y="Y")


class TestBackdoorPaths(unittest.TestCase):
    """
    These tests are drawn from games presented in The Book of Why by Judea Pearl. See the Jupyter Notebook called
    Causal Games in the examples folder for further explanation about each of these.
    """

    def test_game1(self):
        game1 = BayesianNetwork([("X", "A"), ("A", "Y"), ("A", "B")])
        inference = CausalInference(game1)
        self.assertTrue(inference.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(deconfounders, frozenset())

    def test_game2(self):
        game2 = BayesianNetwork(
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
        game3 = BayesianNetwork(
            [("X", "Y"), ("X", "A"), ("B", "A"), ("B", "Y"), ("B", "X")]
        )
        inference = CausalInference(game3)
        self.assertFalse(inference.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(deconfounders, frozenset({frozenset({"B"})}))

    def test_game4(self):
        game4 = BayesianNetwork([("A", "X"), ("A", "B"), ("C", "B"), ("C", "Y")])
        inference = CausalInference(game4)
        self.assertTrue(inference.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(deconfounders, frozenset())

    def test_game5(self):
        game5 = BayesianNetwork(
            [("A", "X"), ("A", "B"), ("C", "B"), ("C", "Y"), ("X", "Y"), ("B", "X")]
        )
        inference = CausalInference(game5)
        self.assertFalse(inference.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(
            deconfounders, frozenset({frozenset({"C"}), frozenset({"A", "B"})})
        )

    def test_game6(self):
        game6 = BayesianNetwork(
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
        self.simpson_model = self.get_simpson_model()
        self.simp_infer = CausalInference(self.simpson_model)

        self.example_model = self.get_example_model()
        self.example_infer = CausalInference(self.example_model)

        self.iv_model = self.get_iv_model()
        self.iv_infer = CausalInference(self.iv_model)

    def get_simpson_model(self):
        simpson_model = BayesianNetwork([("S", "T"), ("T", "C"), ("S", "C")])
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
        simpson_model.add_cpds(cpd_s, cpd_t, cpd_c)

        return simpson_model

    def get_example_model(self):
        # Model structure: Z -> X -> Y; Z -> W -> Y
        example_model = BayesianNetwork(
            [("X", "Y"), ("Z", "X"), ("Z", "W"), ("W", "Y")]
        )
        cpd_z = TabularCPD(variable="Z", variable_card=2, values=[[0.2], [0.8]])

        cpd_x = TabularCPD(
            variable="X",
            variable_card=2,
            values=[[0.1, 0.3], [0.9, 0.7]],
            evidence=["Z"],
            evidence_card=[2],
        )

        cpd_w = TabularCPD(
            variable="W",
            variable_card=2,
            values=[[0.2, 0.9], [0.8, 0.1]],
            evidence=["Z"],
            evidence_card=[2],
        )

        cpd_y = TabularCPD(
            variable="Y",
            variable_card=2,
            values=[[0.3, 0.4, 0.7, 0.8], [0.7, 0.6, 0.3, 0.2]],
            evidence=["X", "W"],
            evidence_card=[2, 2],
        )

        example_model.add_cpds(cpd_z, cpd_x, cpd_w, cpd_y)

        return example_model

    def get_iv_model(self):
        # Model structure: Z -> X -> Y; X <- U -> Y
        example_model = BayesianNetwork(
            [("Z", "X"), ("X", "Y"), ("U", "X"), ("U", "Y")]
        )
        cpd_z = TabularCPD(variable="Z", variable_card=2, values=[[0.2], [0.8]])
        cpd_u = TabularCPD(variable="U", variable_card=2, values=[[0.7], [0.3]])
        cpd_x = TabularCPD(
            variable="X",
            variable_card=2,
            values=[[0.1, 0.3, 0.2, 0.9], [0.9, 0.7, 0.8, 0.1]],
            evidence=["U", "Z"],
            evidence_card=[2, 2],
        )
        cpd_y = TabularCPD(
            variable="Y",
            variable_card=2,
            values=[[0.5, 0.8, 0.2, 0.7], [0.5, 0.2, 0.8, 0.3]],
            evidence=["U", "X"],
            evidence_card=[2, 2],
        )

        example_model.add_cpds(cpd_z, cpd_u, cpd_x, cpd_y)

        return example_model

    def test_query(self):
        for algo in ["ve", "bp"]:
            # Simpson model queries
            query_nodo1 = self.simp_infer.query(
                variables=["C"], do=None, evidence={"T": 1}, inference_algo=algo
            )
            np_test.assert_array_almost_equal(query_nodo1.values, np.array([0.5, 0.5]))

            query_nodo2 = self.simp_infer.query(
                variables=["C"], do=None, evidence={"T": 0}, inference_algo=algo
            )
            np_test.assert_array_almost_equal(query_nodo2.values, np.array([0.6, 0.4]))

            query1 = self.simp_infer.query(
                variables=["C"], do={"T": 1}, inference_algo=algo
            )
            np_test.assert_array_almost_equal(query1.values, np.array([0.6, 0.4]))

            query2 = self.simp_infer.query(
                variables=["C"], do={"T": 0}, inference_algo=algo
            )
            np_test.assert_array_almost_equal(query2.values, np.array([0.5, 0.5]))

            query3 = self.simp_infer.query(["C"], adjustment_set=["S"])
            np_test.assert_array_almost_equal(query3.values, np.array([0.55, 0.45]))

            # IV model queries
            query_nodo1 = self.iv_infer.query(["Z"], do=None, inference_algo=algo)
            np_test.assert_array_almost_equal(query_nodo1.values, np.array([0.2, 0.8]))

            query_nodo2 = self.iv_infer.query(["X"], do=None, evidence={"Z": 1})
            np_test.assert_array_almost_equal(
                query_nodo2.values, np.array([0.48, 0.52])
            )

            query1 = self.iv_infer.query(["X"], do={"Z": 1})
            np_test.assert_array_almost_equal(query1.values, np.array([0.48, 0.52]))

            query2 = self.iv_infer.query(["Y"], do={"X": 1})
            np_test.assert_array_almost_equal(query2.values, np.array([0.77, 0.23]))

            query3 = self.iv_infer.query(["Y"], do={"X": 1}, adjustment_set={"U"})
            np_test.assert_array_almost_equal(query3.values, np.array([0.77, 0.23]))

    def test_adjustment_query(self):
        for algo in ["ve", "bp"]:
            # Test adjustment with do operation.
            query1 = self.example_infer.query(
                variables=["Y"], do={"X": 1}, adjustment_set={"Z"}, inference_algo=algo
            )
            np_test.assert_array_almost_equal(query1.values, np.array([0.7240, 0.2760]))

            query2 = self.example_infer.query(
                variables=["Y"], do={"X": 1}, adjustment_set={"W"}, inference_algo=algo
            )
            np_test.assert_array_almost_equal(query2.values, np.array([0.7240, 0.2760]))

            # Test adjustment without do operation.
            query3 = self.example_infer.query(["Y"], adjustment_set=["W"])
            np_test.assert_array_almost_equal(query3.values, np.array([0.62, 0.38]))

            query4 = self.example_infer.query(["Y"], adjustment_set=["Z"])
            np_test.assert_array_almost_equal(query4.values, np.array([0.62, 0.38]))

            query5 = self.example_infer.query(["Y"], adjustment_set=["W", "Z"])
            np_test.assert_array_almost_equal(query5.values, np.array([0.62, 0.38]))

    def test_issue_1459(self):
        bn = BayesianNetwork([("X", "Y"), ("W", "X"), ("W", "Y")])
        cpd_w = TabularCPD(variable="W", variable_card=2, values=[[0.7], [0.3]])
        cpd_x = TabularCPD(
            variable="X",
            variable_card=2,
            values=[[0.7, 0.4], [0.3, 0.6]],
            evidence=["W"],
            evidence_card=[2],
        )
        cpd_y = TabularCPD(
            variable="Y",
            variable_card=2,
            values=[[0.7, 0.7, 0.5, 0.1], [0.3, 0.3, 0.5, 0.9]],
            evidence=["W", "X"],
            evidence_card=[2, 2],
        )

        bn.add_cpds(cpd_w, cpd_x, cpd_y)
        causal_infer = CausalInference(bn)
        query = causal_infer.query(["Y"], do={"X": 1}, evidence={"W": 1})
        np_test.assert_array_almost_equal(query.values, np.array([0.1, 0.9]))

        # A slight modified version of the above model where only some of the adjustment
        # set variables are in evidence.
        bn = BayesianNetwork(
            [("X", "Y"), ("W1", "X"), ("W1", "Y"), ("W2", "X"), ("W2", "Y")]
        )
        cpd_w1 = TabularCPD(variable="W1", variable_card=2, values=[[0.7], [0.3]])
        cpd_w2 = TabularCPD(variable="W2", variable_card=2, values=[[0.3], [0.7]])
        cpd_x = TabularCPD(
            variable="X",
            variable_card=2,
            values=[[0.7, 0.4, 0.3, 0.8], [0.3, 0.6, 0.7, 0.2]],
            evidence=["W1", "W2"],
            evidence_card=[2, 2],
        )
        cpd_y = TabularCPD(
            variable="Y",
            variable_card=2,
            values=[
                [0.7, 0.7, 0.5, 0.1, 0.9, 0.2, 0.4, 0.6],
                [0.3, 0.3, 0.5, 0.9, 0.1, 0.8, 0.6, 0.4],
            ],
            evidence=["W1", "W2", "X"],
            evidence_card=[2, 2, 2],
        )
        bn.add_cpds(cpd_w1, cpd_w2, cpd_x, cpd_y)
        causal_infer = CausalInference(bn)
        query = causal_infer.query(["Y"], do={"X": 1}, evidence={"W1": 1})
        np_test.assert_array_almost_equal(query.values, np.array([0.48, 0.52]))

    def test_query_error(self):
        self.assertRaises(ValueError, self.simp_infer.query, variables="C", do={"T": 1})
        self.assertRaises(
            ValueError, self.simp_infer.query, variables=["E"], do={"T": 1}
        )
        self.assertRaises(ValueError, self.simp_infer.query, variables=["C"], do="T")
        self.assertRaises(
            ValueError,
            self.simp_infer.query,
            variables=["C"],
            do={"T": 1},
            evidence="S",
        )
        self.assertRaises(
            ValueError,
            self.simp_infer.query,
            variables=["C"],
            do={"T": 1},
            inference_algo="random",
        )


class TestEstimator(unittest.TestCase):
    def test_create_estimator(self):
        game1 = BayesianNetwork([("X", "A"), ("A", "Y"), ("A", "B")])
        data = pd.DataFrame(
            np.random.randint(2, size=(1000, 4)), columns=["X", "A", "B", "Y"]
        )
        inference = CausalInference(model=game1)
        ate = inference.estimate_ate("X", "Y", data=data, estimator_type="linear")
        self.assertAlmostEqual(ate, 0, places=1)

    def test_estimate_frontdoor(self):
        model = BayesianNetwork(
            [("X", "Z"), ("Z", "Y"), ("U", "X"), ("U", "Y")], latents=["U"]
        )
        U = np.random.randn(10000)
        X = 0.3 * U + np.random.randn(10000)
        Z = 0.8 * X + 0.3 * np.random.randn(10000)
        Y = 0.5 * U + 0.9 * Z + 0.4 * np.random.randn(10000)
        data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        infer = CausalInference(model=model)
        ate = infer.estimate_ate("X", "Y", data=data, estimator_type="linear")
        self.assertAlmostEqual(ate, 0.8 * 0.9, places=1)

    def test_estimate_fail_no_adjustment(self):
        model = BayesianNetwork([("X", "Y"), ("U", "X"), ("U", "Y")], latents=["U"])

        U = np.random.randn(10000)
        X = 0.3 * U + np.random.randn(10000)
        Z = 0.8 * X + 0.3 * np.random.randn(10000)
        Y = 0.5 * U + 0.9 * Z + 0.4 * np.random.randn(10000)
        data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        infer = CausalInference(model=model)
        self.assertRaises(ValueError, infer.estimate_ate, "X", "Y", data)

    def test_estimate_multiple_paths(self):
        model = BayesianNetwork(
            [("X", "Z"), ("U", "X"), ("U", "Y"), ("Z", "Y"), ("X", "P1"), ("P1", "Y")],
            latents=["U"],
        )

        U = np.random.randn(10000)
        X = 0.3 * U + np.random.randn(10000)
        P1 = 0.9 * X + np.random.randn(10000)
        Z = 0.8 * X + 0.3 * np.random.randn(10000)
        Y = 0.5 * U + 0.9 * Z + 0.1 * P1 + 0.4 * np.random.randn(10000)
        data = pd.DataFrame({"X": X, "Y": Y, "Z": Z, "P1": P1})

        infer = CausalInference(model=model)
        self.assertAlmostEqual(
            infer.estimate_ate("X", "Y", data), ((0.8 * 0.9) + (0.9 * 0.1)), places=1
        )
