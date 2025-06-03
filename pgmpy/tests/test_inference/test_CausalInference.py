import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd

from pgmpy.base import DAG
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference.CausalInference import CausalInference
from pgmpy.models import DiscreteBayesianNetwork, SEMGraph

np.random.seed(42)


class TestCausalGraphMethods(unittest.TestCase):
    def setUp(self):
        self.game = DiscreteBayesianNetwork(
            [("A", "X"), ("A", "B"), ("C", "B"), ("C", "Y"), ("X", "Y"), ("B", "X")]
        )
        self.inference = CausalInference(self.game)

        self.dag_bd1 = DiscreteBayesianNetwork([("X", "Y"), ("Z1", "X"), ("Z1", "Y")])
        self.inference_bd = CausalInference(self.dag_bd1)

        self.dag_bd2 = DiscreteBayesianNetwork(
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


class TestCausalInferenceInit(unittest.TestCase):
    def test_integer_variable_name(self):
        df = pd.DataFrame([[0, 1], [0, 0]])
        self.model = DiscreteBayesianNetwork(df)
        self.assertRaises(NotImplementedError, CausalInference, self.model)


class TestAdjustmentSet(unittest.TestCase):
    def setUp(self):
        # Model example taken from Constructing Separators and Adjustment Sets
        # in Ancestral Graphs UAI 2014.
        self.model_dag = DAG(
            [("x1", "y1"), ("x1", "z1"), ("z1", "z2"), ("z2", "x2"), ("y2", "z2")]
        )
        self.infer_dag = CausalInference(self.model_dag)

        self.model_sem = SEMGraph(
            [("x1", "y1"), ("x1", "z1"), ("z1", "z2"), ("z2", "x2"), ("y2", "z2")]
        )
        self.infer_sem = CausalInference(self.model_sem)

    def test_proper_backdoor_graph_error(self):
        # DAG
        self.assertRaises(
            ValueError,
            self.infer_dag.get_proper_backdoor_graph,
            X=["x3"],
            Y=["y1", "y2"],
        )
        self.assertRaises(
            ValueError,
            self.infer_dag.get_proper_backdoor_graph,
            X=["x2"],
            Y=["y1", "y3"],
        )
        self.assertRaises(
            ValueError,
            self.infer_dag.get_proper_backdoor_graph,
            X=["x3", "x2"],
            Y=["y1", "y3"],
        )

        # SEMGraph
        self.assertRaises(
            ValueError,
            self.infer_sem.get_proper_backdoor_graph,
            X=["x3"],
            Y=["y1", "y2"],
        )
        self.assertRaises(
            ValueError,
            self.infer_sem.get_proper_backdoor_graph,
            X=["x2"],
            Y=["y1", "y3"],
        )
        self.assertRaises(
            ValueError,
            self.infer_sem.get_proper_backdoor_graph,
            X=["x3", "x2"],
            Y=["y1", "y3"],
        )

    def test_proper_backdoor_graph(self):
        # DAG
        bd_graph = self.infer_dag.get_proper_backdoor_graph(
            X=["x1", "x2"], Y=["y1", "y2"]
        )
        self.assertTrue(("x1", "y1") not in bd_graph.edges())
        self.assertEqual(len(bd_graph.edges()), 4)
        self.assertTrue(
            set(bd_graph.edges()),
            set([("x1", "z1"), ("z1", "z2"), ("z2", "x2"), ("y2", "z2")]),
        )

        # SEMGraph
        bd_graph = self.infer_sem.get_proper_backdoor_graph(
            X=["x1", "x2"], Y=["y1", "y2"]
        )
        self.assertTrue(("x1", "y1") not in bd_graph.edges())
        self.assertEqual(len(bd_graph.edges()), 10)
        self.assertTrue(
            set(bd_graph.edges()),
            set(
                [
                    ("x1", "z1"),
                    ("z1", "z2"),
                    ("z2", "x2"),
                    ("y2", "z2"),
                    (".x1", "x1"),
                    (".y1", "y1"),
                    (".z1", "z1"),
                    (".z2", "z2"),
                    (".x2", "x2"),
                    (".y2", "y2"),
                ]
            ),
        )

    def test_proper_backdoor_graph_not_list(self):
        # DAG
        bd_graph = self.infer_dag.get_proper_backdoor_graph(X="x1", Y="y1")
        self.assertTrue(("x1", "y1") not in bd_graph.edges())
        self.assertEqual(len(bd_graph.edges()), 4)
        self.assertTrue(
            set(bd_graph.edges()),
            set([("x1", "z1"), ("z1", "z2"), ("z2", "x2"), ("y2", "z2")]),
        )

        # SEMGraph
        bd_graph = self.infer_sem.get_proper_backdoor_graph(X="x1", Y="y1")
        self.assertTrue(("x1", "y1") not in bd_graph.edges())
        self.assertEqual(len(bd_graph.edges()), 10)
        self.assertTrue(
            set(bd_graph.edges()),
            set(
                [
                    ("x1", "z1"),
                    ("z1", "z2"),
                    ("z2", "x2"),
                    ("y2", "z2"),
                    (".x1", "x1"),
                    (".y1", "y1"),
                    (".z1", "z1"),
                    (".z2", "z2"),
                    (".x2", "x2"),
                    (".y2", "y2"),
                ]
            ),
        )

    def test_is_valid_adjustment_set(self):
        # DAG
        self.assertTrue(
            self.infer_dag.is_valid_adjustment_set(
                X=["x1", "x2"], Y=["y1", "y2"], adjustment_set=["z1", "z2"]
            )
        )

        self.assertTrue(
            self.infer_dag.is_valid_adjustment_set(
                X="x1", Y="y1", adjustment_set=["z1", "z2"]
            )
        )

        self.assertFalse(
            self.infer_dag.is_valid_adjustment_set(
                X=["x1", "x2"], Y=["y1", "y2"], adjustment_set=["z1"]
            )
        )

        self.assertTrue(
            self.infer_dag.is_valid_adjustment_set(
                X=["x1", "x2"], Y=["y1", "y2"], adjustment_set=["z2"]
            )
        )

        # SEMGraph
        self.assertTrue(
            self.infer_sem.is_valid_adjustment_set(
                X=["x1", "x2"], Y=["y1", "y2"], adjustment_set=["z1", "z2"]
            )
        )

        self.assertTrue(
            self.infer_sem.is_valid_adjustment_set(
                X="x1", Y="y1", adjustment_set=["z1", "z2"]
            )
        )

        self.assertFalse(
            self.infer_sem.is_valid_adjustment_set(
                X=["x1", "x2"], Y=["y1", "y2"], adjustment_set=["z1"]
            )
        )

        self.assertTrue(
            self.infer_sem.is_valid_adjustment_set(
                X=["x1", "x2"], Y=["y1", "y2"], adjustment_set=["z2"]
            )
        )

    def test_get_minimal_adjustment_set(self):
        # Without latent variables
        dag1 = DAG([("X", "Y"), ("Z", "X"), ("Z", "Y")])
        infer = CausalInference(dag1)
        adj_set = infer.get_minimal_adjustment_set(X="X", Y="Y")
        self.assertEqual(adj_set, {"Z"})

        self.assertRaises(ValueError, infer.get_minimal_adjustment_set, X="W", Y="Y")

        # M graph
        dag2 = DAG([("X", "Y"), ("Z1", "X"), ("Z1", "Z3"), ("Z2", "Z3"), ("Z2", "Y")])
        infer = CausalInference(dag2)
        adj_set = infer.get_minimal_adjustment_set(X="X", Y="Y")
        self.assertEqual(adj_set, set())

        # With latents
        dag_lat1 = DAG([("X", "Y"), ("Z", "X"), ("Z", "Y")], latents={"Z"})
        infer = CausalInference(dag_lat1)
        adj_set = infer.get_minimal_adjustment_set(X="X", Y="Y")
        self.assertIsNone(adj_set)

        # Pearl's Simpson machine
        dag_lat2 = DAG(
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

    def test_get_minimal_adjustment_set_sem(self):
        # Without latent variables
        dag1 = SEMGraph([("X", "Y"), ("Z", "X"), ("Z", "Y")])
        infer = CausalInference(dag1)
        adj_set = infer.get_minimal_adjustment_set(X="X", Y="Y")
        self.assertEqual(adj_set, {"Z"})

        self.assertRaises(ValueError, infer.get_minimal_adjustment_set, X="W", Y="Y")

        # M graph
        dag2 = SEMGraph(
            [("X", "Y"), ("Z1", "X"), ("Z1", "Z3"), ("Z2", "Z3"), ("Z2", "Y")]
        )
        infer = CausalInference(dag2)
        adj_set = infer.get_minimal_adjustment_set(X="X", Y="Y")
        self.assertEqual(adj_set, set())

        # With latents
        dag_lat1 = SEMGraph([("X", "Y"), ("Z", "X"), ("Z", "Y")], latents={"Z"})
        infer = CausalInference(dag_lat1)
        adj_set = infer.get_minimal_adjustment_set(X="X", Y="Y")
        self.assertIsNone(adj_set)

        # Pearl's Simpson machine
        dag_lat2 = SEMGraph(
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
        # DAG
        dag = DAG([("X_1", "X_2"), ("Z", "X_1"), ("Z", "X_2")])
        infer = CausalInference(dag)
        adj_set = infer.get_minimal_adjustment_set("X_1", "X_2")

        self.assertEqual(adj_set, {"Z"})
        self.assertRaises(ValueError, infer.get_minimal_adjustment_set, X="X_3", Y="Y")

        # SEM
        sem = SEMGraph([("X_1", "X_2"), ("Z", "X_1"), ("Z", "X_2")])
        infer = CausalInference(sem)
        adj_set = infer.get_minimal_adjustment_set("X_1", "X_2")

        self.assertEqual(adj_set, {"Z"})
        self.assertRaises(ValueError, infer.get_minimal_adjustment_set, X="X_3", Y="Y")


class TestBackdoorPaths(unittest.TestCase):
    """
    These tests are drawn from games presented in The Book of Why by Judea Pearl. See the Jupyter Notebook called
    Causal Games in the examples folder for further explanation about each of these.
    """

    def test_game1_bn(self):
        game1 = DiscreteBayesianNetwork([("X", "A"), ("A", "Y"), ("A", "B")])
        inference = CausalInference(game1)
        self.assertTrue(inference.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(deconfounders, frozenset())

    def test_game1_sem(self):
        game1 = SEMGraph(ebunch=[("X", "A"), ("A", "Y"), ("A", "B")])
        inference = CausalInference(game1)
        self.assertTrue(inference.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(deconfounders, frozenset())

    def test_game2_bn(self):
        game2 = DiscreteBayesianNetwork(
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

    def test_game2_sem(self):
        game2 = SEMGraph(
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

    def test_game3_bn(self):
        game3 = DiscreteBayesianNetwork(
            [("X", "Y"), ("X", "A"), ("B", "A"), ("B", "Y"), ("B", "X")]
        )
        inference = CausalInference(game3)
        self.assertFalse(inference.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(deconfounders, frozenset({frozenset({"B"})}))

    def test_game3_sem(self):
        game3 = SEMGraph([("X", "Y"), ("X", "A"), ("B", "A"), ("B", "Y"), ("B", "X")])
        inference = CausalInference(game3)
        self.assertFalse(inference.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(deconfounders, frozenset({frozenset({"B"})}))

    def test_game4_bn(self):
        game4 = DiscreteBayesianNetwork(
            [("A", "X"), ("A", "B"), ("C", "B"), ("C", "Y")]
        )
        inference = CausalInference(game4)
        self.assertTrue(inference.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(deconfounders, frozenset())

    def test_game4_sem(self):
        game4 = SEMGraph([("A", "X"), ("A", "B"), ("C", "B"), ("C", "Y")])
        inference = CausalInference(game4)
        self.assertTrue(inference.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(deconfounders, frozenset())

    def test_game5_bn(self):
        game5 = DiscreteBayesianNetwork(
            [("A", "X"), ("A", "B"), ("C", "B"), ("C", "Y"), ("X", "Y"), ("B", "X")]
        )
        inference = CausalInference(game5)
        self.assertFalse(inference.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(
            deconfounders, frozenset({frozenset({"C"}), frozenset({"A", "B"})})
        )

    def test_game5_sem(self):
        game5 = SEMGraph(
            [("A", "X"), ("A", "B"), ("C", "B"), ("C", "Y"), ("X", "Y"), ("B", "X")]
        )
        inference = CausalInference(game5)
        self.assertFalse(inference.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(
            deconfounders, frozenset({frozenset({"C"}), frozenset({"A", "B"})})
        )

    def test_game6_bn(self):
        game6 = DiscreteBayesianNetwork(
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

    def test_game6_sem(self):
        game6 = SEMGraph(
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


class TestSEMIdentification(unittest.TestCase):
    def setUp(self):
        demo = SEMGraph(
            ebunch=[
                ("xi1", "x1"),
                ("xi1", "x2"),
                ("xi1", "x3"),
                ("xi1", "eta1"),
                ("eta1", "y1"),
                ("eta1", "y2"),
                ("eta1", "y3"),
                ("eta1", "y4"),
                ("eta1", "eta2"),
                ("xi1", "eta2"),
                ("eta2", "y5"),
                ("eta2", "y6"),
                ("eta2", "y7"),
                ("eta2", "y8"),
            ],
            latents=["xi1", "eta1", "eta2"],
            err_corr=[
                ("y1", "y5"),
                ("y2", "y6"),
                ("y2", "y4"),
                ("y3", "y7"),
                ("y4", "y8"),
                ("y6", "y8"),
            ],
        )

        union = SEMGraph(
            ebunch=[
                ("yrsmill", "unionsen"),
                ("age", "laboract"),
                ("age", "deferenc"),
                ("deferenc", "laboract"),
                ("deferenc", "unionsen"),
                ("laboract", "unionsen"),
            ],
            latents=[],
            err_corr=[("yrsmill", "age")],
        )

        demo_params = SEMGraph(
            ebunch=[
                ("xi1", "x1", 0.4),
                ("xi1", "x2", 0.5),
                ("xi1", "x3", 0.6),
                ("xi1", "eta1", 0.3),
                ("eta1", "y1", 1.1),
                ("eta1", "y2", 1.2),
                ("eta1", "y3", 1.3),
                ("eta1", "y4", 1.4),
                ("eta1", "eta2", 0.1),
                ("xi1", "eta2", 0.2),
                ("eta2", "y5", 0.7),
                ("eta2", "y6", 0.8),
                ("eta2", "y7", 0.9),
                ("eta2", "y8", 1.0),
            ],
            latents=["xi1", "eta1", "eta2"],
            err_corr=[
                ("y1", "y5", 1.5),
                ("y2", "y6", 1.6),
                ("y2", "y4", 1.9),
                ("y3", "y7", 1.7),
                ("y4", "y8", 1.8),
                ("y6", "y8", 2.0),
            ],
            err_var={
                "y1": 2.1,
                "y2": 2.2,
                "y3": 2.3,
                "y4": 2.4,
                "y5": 2.5,
                "y6": 2.6,
                "y7": 2.7,
                "y8": 2.8,
                "x1": 3.1,
                "x2": 3.2,
                "x3": 3.3,
                "eta1": 2.9,
                "eta2": 3.0,
                "xi1": 3.4,
            },
        )

        custom = SEMGraph(
            ebunch=[
                ("xi1", "eta1"),
                ("xi1", "y1"),
                ("xi1", "y4"),
                ("xi1", "x1"),
                ("xi1", "x2"),
                ("y4", "y1"),
                ("y1", "eta2"),
                ("eta2", "y5"),
                ("y1", "eta1"),
                ("eta1", "y2"),
                ("eta1", "y3"),
            ],
            latents=["xi1", "eta1", "eta2"],
            err_corr=[("y1", "y2"), ("y2", "y3")],
            err_var={},
        )

        self.demo = CausalInference(demo)
        self.union = CausalInference(union)
        self.demo_params = CausalInference(demo_params)
        self.custom = CausalInference(custom)

    def test_get_scaling_indicators(self):
        demo_scaling_indicators = self.demo.get_scaling_indicators()
        self.assertTrue(demo_scaling_indicators["eta1"] in ["y1", "y2", "y3", "y4"])
        self.assertTrue(demo_scaling_indicators["eta2"] in ["y5", "y6", "y7", "y8"])
        self.assertTrue(demo_scaling_indicators["xi1"] in ["x1", "x2", "x3"])

        union_scaling_indicators = self.union.get_scaling_indicators()
        self.assertDictEqual(union_scaling_indicators, dict())

        custom_scaling_indicators = self.custom.get_scaling_indicators()
        self.assertTrue(custom_scaling_indicators["xi1"] in ["x1", "x2", "y1", "y4"])
        self.assertTrue(custom_scaling_indicators["eta1"] in ["y2", "y3"])
        self.assertTrue(custom_scaling_indicators["eta2"] in ["y5"])

    def test_iv_transformations_demo(self):
        scale = {"eta1": "y1", "eta2": "y5", "xi1": "x1"}

        self.assertRaises(ValueError, self.demo._iv_transformations, "x1", "y1", scale)

        for y in ["y2", "y3", "y4"]:
            full_graph, dependent_var = self.demo._iv_transformations(
                X="eta1", Y=y, scaling_indicators=scale
            )
            self.assertEqual(dependent_var, y)
            self.assertTrue((".y1", y) in full_graph.edges)
            self.assertFalse(("eta1", y) in full_graph.edges)

        for y in ["y6", "y7", "y8"]:
            full_graph, dependent_var = self.demo._iv_transformations(
                X="eta2", Y=y, scaling_indicators=scale
            )
            self.assertEqual(dependent_var, y)
            self.assertTrue((".y5", y) in full_graph.edges)
            self.assertFalse(("eta2", y) in full_graph.edges)

        full_graph, dependent_var = self.demo._iv_transformations(
            X="xi1", Y="eta1", scaling_indicators=scale
        )
        self.assertEqual(dependent_var, "y1")
        self.assertTrue((".eta1", "y1") in full_graph.edges())
        self.assertTrue((".x1", "y1") in full_graph.edges())
        self.assertFalse(("xi1", "eta1") in full_graph.edges())

        full_graph, dependent_var = self.demo._iv_transformations(
            X="xi1", Y="eta2", scaling_indicators=scale
        )
        self.assertEqual(dependent_var, "y5")
        self.assertTrue((".y1", "y5") in full_graph.edges())
        self.assertTrue((".eta2", "y5") in full_graph.edges())
        self.assertTrue((".x1", "y5") in full_graph.edges())
        self.assertFalse(("eta1", "eta2") in full_graph.edges())
        self.assertFalse(("xi1", "eta2") in full_graph.edges())

        full_graph, dependent_var = self.demo._iv_transformations(
            X="eta1", Y="eta2", scaling_indicators=scale
        )
        self.assertEqual(dependent_var, "y5")
        self.assertTrue((".y1", "y5") in full_graph.edges())
        self.assertTrue((".eta2", "y5") in full_graph.edges())
        self.assertTrue((".x1", "y5") in full_graph.edges())
        self.assertFalse(("eta1", "eta2") in full_graph.edges())
        self.assertFalse(("xi1", "eta2") in full_graph.edges())

    def test_iv_transformations_union(self):
        scale = {}
        for u, v in [
            ("yrsmill", "unionsen"),
            ("age", "laboract"),
            ("age", "deferenc"),
            ("deferenc", "laboract"),
            ("deferenc", "unionsen"),
            ("laboract", "unionsen"),
        ]:
            full_graph, dependent_var = self.union._iv_transformations(
                u, v, scaling_indicators=scale
            )
            self.assertFalse((u, v) in full_graph.edges())
            self.assertEqual(dependent_var, v)

    def test_get_ivs_demo(self):
        scale = {"eta1": "y1", "eta2": "y5", "xi1": "x1"}

        self.assertSetEqual(
            self.demo.get_ivs("eta1", "y2", scaling_indicators=scale),
            {"x1", "x2", "x3", "y3", "y7", "y8"},
        )
        self.assertSetEqual(
            self.demo.get_ivs("eta1", "y3", scaling_indicators=scale),
            {"x1", "x2", "x3", "y2", "y4", "y6", "y8"},
        )
        self.assertSetEqual(
            self.demo.get_ivs("eta1", "y4", scaling_indicators=scale),
            {"x1", "x2", "x3", "y3", "y6", "y7"},
        )

        self.assertSetEqual(
            self.demo.get_ivs("eta2", "y6", scaling_indicators=scale),
            {"x1", "x2", "x3", "y3", "y4", "y7"},
        )
        self.assertSetEqual(
            self.demo.get_ivs("eta2", "y7", scaling_indicators=scale),
            {"x1", "x2", "x3", "y2", "y4", "y6", "y8"},
        )
        self.assertSetEqual(
            self.demo.get_ivs("eta2", "y8", scaling_indicators=scale),
            {"x1", "x2", "x3", "y2", "y3", "y7"},
        )

        self.assertSetEqual(
            self.demo.get_ivs("xi1", "x2", scaling_indicators=scale),
            {"x3", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8"},
        )
        self.assertSetEqual(
            self.demo.get_ivs("xi1", "x3", scaling_indicators=scale),
            {"x2", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8"},
        )

        self.assertSetEqual(
            self.demo.get_ivs("xi1", "eta1", scaling_indicators=scale), {"x2", "x3"}
        )
        self.assertSetEqual(
            self.demo.get_ivs("xi1", "eta2", scaling_indicators=scale),
            {"x2", "x3", "y2", "y3", "y4"},
        )
        self.assertSetEqual(
            self.demo.get_ivs("eta1", "eta2", scaling_indicators=scale),
            {"x2", "x3", "y2", "y3", "y4"},
        )

    def test_get_conditional_ivs_demo(self):
        scale = {"eta1": "y1", "eta2": "y5", "xi1": "x1"}

        self.assertEqual(
            self.demo.get_conditional_ivs("eta1", "y2", scaling_indicators=scale), []
        )
        self.assertEqual(
            self.demo.get_conditional_ivs("eta1", "y3", scaling_indicators=scale), []
        )
        self.assertEqual(
            self.demo.get_conditional_ivs("eta1", "y4", scaling_indicators=scale), []
        )

        self.assertEqual(
            self.demo.get_conditional_ivs("eta2", "y6", scaling_indicators=scale), []
        )
        self.assertEqual(
            self.demo.get_conditional_ivs("eta2", "y7", scaling_indicators=scale), []
        )
        self.assertEqual(
            self.demo.get_conditional_ivs("eta2", "y8", scaling_indicators=scale), []
        )

        self.assertEqual(
            self.demo.get_conditional_ivs("xi1", "x2", scaling_indicators=scale), []
        )
        self.assertEqual(
            self.demo.get_conditional_ivs("xi1", "x3", scaling_indicators=scale), []
        )

        self.assertEqual(
            self.demo.get_conditional_ivs("xi1", "eta1", scaling_indicators=scale), []
        )
        self.assertEqual(
            self.demo.get_conditional_ivs("xi1", "eta2", scaling_indicators=scale), []
        )
        self.assertEqual(
            self.demo.get_conditional_ivs("eta1", "eta2", scaling_indicators=scale), []
        )

    def test_get_ivs_union(self):
        scale = {}
        self.assertSetEqual(
            self.union.get_ivs("yrsmill", "unionsen", scaling_indicators=scale), set()
        )
        self.assertSetEqual(
            self.union.get_ivs("deferenc", "unionsen", scaling_indicators=scale), set()
        )
        self.assertSetEqual(
            self.union.get_ivs("laboract", "unionsen", scaling_indicators=scale), set()
        )
        self.assertSetEqual(
            self.union.get_ivs("deferenc", "laboract", scaling_indicators=scale), set()
        )
        self.assertSetEqual(
            self.union.get_ivs("age", "laboract", scaling_indicators=scale), {"yrsmill"}
        )
        self.assertSetEqual(
            self.union.get_ivs("age", "deferenc", scaling_indicators=scale), {"yrsmill"}
        )

    def test_get_conditional_ivs_union(self):
        self.assertEqual(
            self.union.get_conditional_ivs("yrsmill", "unionsen"),
            [("age", {"laboract", "deferenc"})],
        )
        # This case wouldn't have conditonal IV if the Total effect between `deferenc` and
        # `unionsen` needs to be computed because one of the conditional variable lies on the
        # effect path.
        self.assertEqual(
            self.union.get_conditional_ivs("deferenc", "unionsen"),
            [("age", {"yrsmill", "laboract"})],
        )
        self.assertEqual(
            self.union.get_conditional_ivs("laboract", "unionsen"),
            [("age", {"yrsmill", "deferenc"})],
        )
        self.assertEqual(self.union.get_conditional_ivs("deferenc", "laboract"), [])

        self.assertEqual(
            self.union.get_conditional_ivs("age", "laboract"),
            [("yrsmill", {"deferenc"})],
        )
        self.assertEqual(self.union.get_conditional_ivs("age", "deferenc"), [])

    def test_total_conditional_ivs_union(self):
        self.assertEqual(
            self.union.get_total_conditional_ivs("deferenc", "unionsen"),
            [],
        )

    def test_iv_transformations_custom(self):
        scale_custom = {"eta1": "y2", "eta2": "y5", "xi1": "x1"}

        full_graph, var = self.custom._iv_transformations(
            "xi1", "x2", scaling_indicators=scale_custom
        )
        self.assertEqual(var, "x2")
        self.assertTrue((".x1", "x2") in full_graph.edges())
        self.assertFalse(("xi1", "x2") in full_graph.edges())

        full_graph, var = self.custom._iv_transformations(
            "xi1", "y4", scaling_indicators=scale_custom
        )
        self.assertEqual(var, "y4")
        self.assertTrue((".x1", "y4") in full_graph.edges())
        self.assertFalse(("xi1", "y4") in full_graph.edges())

        full_graph, var = self.custom._iv_transformations(
            "xi1", "y1", scaling_indicators=scale_custom
        )
        self.assertEqual(var, "y1")
        self.assertTrue((".x1", "y1") in full_graph.edges())
        self.assertFalse(("xi1", "y1") in full_graph.edges())
        self.assertFalse(("y4", "y1") in full_graph.edges())

        full_graph, var = self.custom._iv_transformations(
            "xi1", "eta1", scaling_indicators=scale_custom
        )
        self.assertEqual(var, "y2")
        self.assertTrue((".eta1", "y2") in full_graph.edges())
        self.assertTrue((".x1", "y2") in full_graph.edges())
        self.assertFalse(("y1", "eta1") in full_graph.edges())
        self.assertFalse(("xi1", "eta1") in full_graph.edges())

        full_graph, var = self.custom._iv_transformations(
            "y1", "eta1", scaling_indicators=scale_custom
        )
        self.assertEqual(var, "y2")
        self.assertTrue((".eta1", "y2") in full_graph.edges())
        self.assertTrue((".x1", "y2") in full_graph.edges())
        self.assertFalse(("y1", "eta1") in full_graph.edges())
        self.assertFalse(("xi1", "eta1") in full_graph.edges())

        full_graph, var = self.custom._iv_transformations(
            "y1", "eta2", scaling_indicators=scale_custom
        )
        self.assertEqual(var, "y5")
        self.assertTrue((".eta2", "y5") in full_graph.edges())
        self.assertFalse(("y1", "eta2") in full_graph.edges())

        full_graph, var = self.custom._iv_transformations(
            "y4", "y1", scaling_indicators=scale_custom
        )
        self.assertEqual(var, "y1")
        self.assertFalse(("y4", "y1") in full_graph.edges())

        full_graph, var = self.custom._iv_transformations(
            "eta1", "y3", scaling_indicators=scale_custom
        )
        self.assertEqual(var, "y3")
        self.assertTrue((".y2", "y3") in full_graph.edges())
        self.assertFalse(("eta1", "y3") in full_graph.edges())

    def test_get_ivs_custom(self):
        scale_custom = {"eta1": "y2", "eta2": "y5", "xi1": "x1"}

        self.assertSetEqual(
            self.custom.get_ivs("xi1", "x2", scaling_indicators=scale_custom),
            {"y1", "y2", "y3", "y4", "y5"},
        )
        self.assertSetEqual(
            self.custom.get_ivs("xi1", "y4", scaling_indicators=scale_custom), {"x2"}
        )
        self.assertSetEqual(
            self.custom.get_ivs("xi1", "y1", scaling_indicators=scale_custom),
            {"x2", "y4"},
        )
        self.assertSetEqual(
            self.custom.get_ivs("xi1", "eta1", scaling_indicators=scale_custom),
            {"x2", "y4"},
        )
        # TODO: Test this and fix.
        self.assertSetEqual(
            self.custom.get_ivs("y1", "eta1", scaling_indicators=scale_custom),
            {"x2", "y4", "y5"},
        )
        self.assertSetEqual(
            self.custom.get_ivs("y1", "eta2", scaling_indicators=scale_custom),
            {"x1", "x2", "y2", "y3", "y4"},
        )
        self.assertSetEqual(
            self.custom.get_ivs("y4", "y1", scaling_indicators=scale_custom), set()
        )
        self.assertSetEqual(
            self.custom.get_ivs("eta1", "y3", scaling_indicators=scale_custom),
            {"x1", "x2", "y4"},
        )

    def test_small_model_ivs(self):
        model1 = SEMGraph(
            ebunch=[("X", "Y"), ("I", "X"), ("W", "I")],
            latents=[],
            err_corr=[("W", "Y")],
            err_var={},
        )
        inference1 = CausalInference(model1)
        self.assertEqual(inference1.get_conditional_ivs("X", "Y"), [("I", {"W"})])

        model2 = SEMGraph(
            ebunch=[
                ("x", "y"),
                ("z", "x"),
                ("w", "z"),
                ("w", "u"),
                ("u", "x"),
                ("u", "y"),
            ],
            latents=["u"],
        )
        inference2 = CausalInference(model2)
        self.assertEqual(inference2.get_conditional_ivs("x", "y"), [("z", {"w"})])

        model3 = SEMGraph(
            ebunch=[("x", "y"), ("u", "x"), ("u", "y"), ("z", "x")], latents=["u"]
        )
        inference3 = CausalInference(model3)
        self.assertEqual(inference3.get_ivs("x", "y"), {"z"})

        model4 = SEMGraph(ebunch=[("x", "y"), ("z", "x"), ("u", "x"), ("u", "y")])
        inference4 = CausalInference(model4)
        self.assertEqual(inference4.get_conditional_ivs("x", "y"), [("z", {"u"})])


class TestBayesianIV(unittest.TestCase):
    def setUp(self):
        self.model = DiscreteBayesianNetwork(
            ebunch=[("Z", "X"), ("X", "Y"), ("U", "Y"), ("U", "X")], latents=["U"]
        )

        self.causal_inf = CausalInference(self.model)

    def test_get_ivs(self):
        ivs = self.causal_inf.get_ivs("X", "Y")
        self.assertIn("Z", ivs)

    def test_get_conditional_ivs(self):
        self.model.add_edge("I", "X")
        self.model.add_edge("W", "I")
        self.model.add_edge("W", "Y")
        self.causal_inf = CausalInference(self.model)
        cond_ivs = self.causal_inf.get_conditional_ivs("X", "Y")
        self.assertIn(("I", {"W"}), cond_ivs)

    def test_identification_method(self):
        backdoor_model = DiscreteBayesianNetwork(
            ebunch=[("X", "Y"), ("M", "Y"), ("M", "X")]
        )
        causal_inf = CausalInference(backdoor_model)
        methods = causal_inf.identification_method("X", "Y")
        expected_backdoor = {"backdoor set": {frozenset({"M"})}}
        self.assertEqual(methods, expected_backdoor)

        frontdoor_model = DiscreteBayesianNetwork(ebunch=[("X", "M"), ("M", "Y")])
        causal_inf = CausalInference(frontdoor_model)
        methods = causal_inf.identification_method("X", "Y")
        expected_frontdoor = {"frontdoor set": {frozenset({"M"})}}
        self.assertEqual(methods, expected_frontdoor)

        iv_model = DiscreteBayesianNetwork(
            ebunch=[("Z", "X"), ("X", "Y"), ("U", "Y"), ("U", "X")], latents=["U"]
        )
        causal_inf = CausalInference(iv_model)
        methods = causal_inf.identification_method("X", "Y")
        expected_iv = {"instrumental variables": {"Z"}}
        self.assertEqual(methods, expected_iv)


class TestDoQuery(unittest.TestCase):
    def setUp(self):
        self.simpson_model = self.get_simpson_model()
        self.simp_infer = CausalInference(self.simpson_model)

        self.example_model = self.get_example_model()
        self.example_infer = CausalInference(self.example_model)

        self.iv_model = self.get_iv_model()
        self.iv_infer = CausalInference(self.iv_model)

    def get_simpson_model(self):
        simpson_model = DiscreteBayesianNetwork([("S", "T"), ("T", "C"), ("S", "C")])
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
        example_model = DiscreteBayesianNetwork(
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
        example_model = DiscreteBayesianNetwork(
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
        bn = DiscreteBayesianNetwork([("X", "Y"), ("W", "X"), ("W", "Y")])
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
        bn = DiscreteBayesianNetwork(
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

    def test_invalid_causal_query_direct_descendant_intervention(self):
        # Model: R -> S -> W & R -> W. We intervene on S and query R.
        model = DiscreteBayesianNetwork([("R", "W"), ("S", "W"), ("R", "S")])
        cpd_rain = TabularCPD(
            variable="R",
            variable_card=2,
            values=[[0.6], [0.4]],
            state_names={"R": ["True", "False"]},
        )
        cpd_sprinkler = TabularCPD(
            variable="S",
            variable_card=2,
            values=[[0.1, 0.5], [0.9, 0.5]],
            evidence=["R"],
            evidence_card=[2],
            state_names={"S": ["True", "False"], "R": ["True", "False"]},
        )
        cpd_wet_grass = TabularCPD(
            variable="W",
            variable_card=2,
            values=[[0.99, 0.9, 0.9, 0.01], [0.01, 0.1, 0.1, 0.99]],
            evidence=["R", "S"],
            evidence_card=[2, 2],
            state_names={
                "W": ["True", "False"],
                "R": ["True", "False"],
                "S": ["True", "False"],
            },
        )
        model.add_cpds(cpd_rain, cpd_sprinkler, cpd_wet_grass)
        causal_inference = CausalInference(model)

        evidence = {"W": "True"}
        counterfactual_intervention = {"S": "False"}
        with self.assertRaises(ValueError) as cm:
            causal_inference.query(
                variables=["R"], evidence=evidence, do=counterfactual_intervention
            )
        self.assertIn(
            "Invalid causal query: There is a direct edge from the query"
            " variable 'R' to the intervention variable 'S'.",
            str(cm.exception),
        )


class TestEstimator(unittest.TestCase):
    def test_create_estimator(self):
        game1 = DiscreteBayesianNetwork([("X", "A"), ("A", "Y"), ("A", "B")])
        data = pd.DataFrame(
            np.random.randint(2, size=(1000, 4)), columns=["X", "A", "B", "Y"]
        )
        inference = CausalInference(model=game1)
        ate = inference.estimate_ate("X", "Y", data=data, estimator_type="linear")
        self.assertAlmostEqual(ate, 0, places=1)

    def test_estimate_frontdoor(self):
        model = DiscreteBayesianNetwork(
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
        model = DiscreteBayesianNetwork(
            [("X", "Y"), ("U", "X"), ("U", "Y")], latents=["U"]
        )

        U = np.random.randn(10000)
        X = 0.3 * U + np.random.randn(10000)
        Z = 0.8 * X + 0.3 * np.random.randn(10000)
        Y = 0.5 * U + 0.9 * Z + 0.4 * np.random.randn(10000)
        data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        infer = CausalInference(model=model)
        self.assertRaises(ValueError, infer.estimate_ate, "X", "Y", data)

    def test_estimate_multiple_paths(self):
        model = DiscreteBayesianNetwork(
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
