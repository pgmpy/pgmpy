import unittest

import networkx as nx
import pandas as pd
import numpy as np

from pgmpy.models import NaiveBayes
from pgmpy.independencies import Independencies


class TestBaseModelCreation(unittest.TestCase):
    def setUp(self):
        self.G = NaiveBayes()

    def test_class_init_without_data(self):
        self.assertIsInstance(self.G, nx.DiGraph)

    def test_class_init_with_data_string(self):
        self.g = NaiveBayes(feature_vars=["b", "c"], dependent_var="a")
        self.assertCountEqual(list(self.g.nodes()), ["a", "b", "c"])
        self.assertCountEqual(list(self.g.edges()), [("a", "b"), ("a", "c")])
        self.assertEqual(self.g.dependent, "a")
        self.assertSetEqual(self.g.features, {"b", "c"})

    def test_class_init_with_data_nonstring(self):
        self.g = NaiveBayes(feature_vars=[2, 3], dependent_var=1)
        self.assertCountEqual(list(self.g.nodes()), [1, 2, 3])
        self.assertCountEqual(list(self.g.edges()), [(1, 2), (1, 3)])
        self.assertEqual(self.g.dependent, 1)
        self.assertSetEqual(self.g.features, {2, 3})

    def test_add_node_string(self):
        self.G.add_node("a")
        self.assertListEqual(list(self.G.nodes()), ["a"])

    def test_add_node_nonstring(self):
        self.G.add_node(1)
        self.assertListEqual(list(self.G.nodes()), [1])

    def test_add_nodes_from_string(self):
        self.G.add_nodes_from(["a", "b", "c", "d"])
        self.assertCountEqual(list(self.G.nodes()), ["a", "b", "c", "d"])

    def test_add_nodes_from_non_string(self):
        self.G.add_nodes_from([1, 2, 3, 4])
        self.assertCountEqual(list(self.G.nodes()), [1, 2, 3, 4])

    def test_add_edge_string(self):
        self.G.add_edge("a", "b")
        self.assertCountEqual(list(self.G.nodes()), ["a", "b"])
        self.assertListEqual(list(self.G.edges()), [("a", "b")])
        self.assertEqual(self.G.dependent, "a")
        self.assertSetEqual(self.G.features, {"b"})

        self.G.add_nodes_from(["c", "d"])
        self.G.add_edge("a", "c")
        self.G.add_edge("a", "d")
        self.assertCountEqual(list(self.G.nodes()), ["a", "b", "c", "d"])
        self.assertCountEqual(
            list(self.G.edges()), [("a", "b"), ("a", "c"), ("a", "d")]
        )
        self.assertEqual(self.G.dependent, "a")
        self.assertSetEqual(self.G.features, {"b", "c", "d"})

        self.assertRaises(ValueError, self.G.add_edge, "b", "c")
        self.assertRaises(ValueError, self.G.add_edge, "d", "f")
        self.assertRaises(ValueError, self.G.add_edge, "e", "f")
        self.assertRaises(ValueError, self.G.add_edges_from, [("a", "e"), ("b", "f")])
        self.assertRaises(ValueError, self.G.add_edges_from, [("b", "f")])

    def test_add_edge_nonstring(self):
        self.G.add_edge(1, 2)
        self.assertCountEqual(list(self.G.nodes()), [1, 2])
        self.assertListEqual(list(self.G.edges()), [(1, 2)])
        self.assertEqual(self.G.dependent, 1)
        self.assertSetEqual(self.G.features, {2})

        self.G.add_nodes_from([3, 4])
        self.G.add_edge(1, 3)
        self.G.add_edge(1, 4)
        self.assertCountEqual(list(self.G.nodes()), [1, 2, 3, 4])
        self.assertCountEqual(list(self.G.edges()), [(1, 2), (1, 3), (1, 4)])
        self.assertEqual(self.G.dependent, 1)
        self.assertSetEqual(self.G.features, {2, 3, 4})

        self.assertRaises(ValueError, self.G.add_edge, 2, 3)
        self.assertRaises(ValueError, self.G.add_edge, 3, 6)
        self.assertRaises(ValueError, self.G.add_edge, 5, 6)
        self.assertRaises(ValueError, self.G.add_edges_from, [(1, 5), (2, 6)])
        self.assertRaises(ValueError, self.G.add_edges_from, [(2, 6)])

    def test_add_edge_selfloop(self):
        self.assertRaises(ValueError, self.G.add_edge, "a", "a")
        self.assertRaises(ValueError, self.G.add_edge, 1, 1)

    def test_add_edges_from_self_loop(self):
        self.assertRaises(ValueError, self.G.add_edges_from, [("a", "a")])

    def test_update_node_parents_bm_constructor(self):
        self.g = NaiveBayes(feature_vars=["b", "c"], dependent_var="a")
        self.assertListEqual(list(self.g.predecessors("a")), [])
        self.assertListEqual(list(self.g.predecessors("b")), ["a"])
        self.assertListEqual(list(self.g.predecessors("c")), ["a"])

    def test_update_node_parents(self):
        self.G.add_nodes_from(["a", "b", "c"])
        self.G.add_edges_from([("a", "b"), ("a", "c")])
        self.assertListEqual(list(self.G.predecessors("a")), [])
        self.assertListEqual(list(self.G.predecessors("b")), ["a"])
        self.assertListEqual(list(self.G.predecessors("c")), ["a"])

    def tearDown(self):
        del self.G


class TestNaiveBayesMethods(unittest.TestCase):
    def setUp(self):
        self.G1 = NaiveBayes(feature_vars=["b", "c", "d", "e"], dependent_var="a")
        self.G2 = NaiveBayes(feature_vars=["g", "l", "s"], dependent_var="d")

    def test_local_independencies(self):
        self.assertEqual(self.G1.local_independencies("a"), Independencies())
        self.assertEqual(
            self.G1.local_independencies("b"),
            Independencies(["b", ["e", "c", "d"], "a"]),
        )
        self.assertEqual(
            self.G1.local_independencies("c"),
            Independencies(["c", ["e", "b", "d"], "a"]),
        )
        self.assertEqual(
            self.G1.local_independencies("d"),
            Independencies(["d", ["b", "c", "e"], "a"]),
        )

    def test_active_trail_nodes(self):
        self.assertListEqual(
            sorted(self.G2.active_trail_nodes("d")), ["d", "g", "l", "s"]
        )
        self.assertListEqual(
            sorted(self.G2.active_trail_nodes("g")), ["d", "g", "l", "s"]
        )
        self.assertListEqual(
            sorted(self.G2.active_trail_nodes("l")), ["d", "g", "l", "s"]
        )
        self.assertListEqual(
            sorted(self.G2.active_trail_nodes("s")), ["d", "g", "l", "s"]
        )

    def test_active_trail_nodes_args(self):
        self.assertListEqual(
            sorted(self.G2.active_trail_nodes("d", observed="g")), ["d", "l", "s"]
        )
        self.assertListEqual(
            sorted(self.G2.active_trail_nodes("l", observed="g")), ["d", "l", "s"]
        )
        self.assertListEqual(
            sorted(self.G2.active_trail_nodes("s", observed=["g", "l"])), ["d", "s"]
        )
        self.assertListEqual(
            sorted(self.G2.active_trail_nodes("s", observed=["d", "l"])), ["s"]
        )

    def test_get_ancestors_of(self):
        self.assertListEqual(sorted(self.G1._get_ancestors_of("b")), ["a", "b"])
        self.assertListEqual(sorted(self.G1._get_ancestors_of("e")), ["a", "e"])
        self.assertListEqual(sorted(self.G1._get_ancestors_of("a")), ["a"])
        self.assertListEqual(
            sorted(self.G1._get_ancestors_of(["b", "e"])), ["a", "b", "e"]
        )

    def tearDown(self):
        del self.G1
        del self.G2


class TestNaiveBayesFit(unittest.TestCase):
    def setUp(self):
        self.model1 = NaiveBayes()
        self.model2 = NaiveBayes(feature_vars=["B"], dependent_var="A")

    def test_fit_model_creation(self):
        values = pd.DataFrame(
            np.random.randint(low=0, high=2, size=(1000, 5)),
            columns=["A", "B", "C", "D", "E"],
        )

        self.model1.fit(values, "A")
        self.assertCountEqual(self.model1.nodes(), ["A", "B", "C", "D", "E"])
        self.assertCountEqual(
            self.model1.edges(), [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E")]
        )
        self.assertEqual(self.model1.dependent, "A")
        self.assertSetEqual(self.model1.features, {"B", "C", "D", "E"})

        self.model2.fit(values)
        self.assertCountEqual(self.model1.nodes(), ["A", "B", "C", "D", "E"])
        self.assertCountEqual(
            self.model1.edges(), [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E")]
        )
        self.assertEqual(self.model2.dependent, "A")
        self.assertSetEqual(self.model2.features, {"B", "C", "D", "E"})

    def test_fit_model_creation_exception(self):
        values = pd.DataFrame(
            np.random.randint(low=0, high=2, size=(1000, 5)),
            columns=["A", "B", "C", "D", "E"],
        )
        values2 = pd.DataFrame(
            np.random.randint(low=0, high=2, size=(1000, 3)), columns=["C", "D", "E"]
        )

        self.assertRaises(ValueError, self.model1.fit, values)
        self.assertRaises(ValueError, self.model1.fit, values2)
        self.assertRaises(ValueError, self.model2.fit, values2, "A")

    def tearDown(self):
        del self.model1
        del self.model2
