import unittest

import numpy as np

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import ClusterGraph
from pgmpy.tests import help_functions as hf


class TestClusterGraphCreation(unittest.TestCase):
    def setUp(self):
        self.graph = ClusterGraph()

    def test_add_single_node(self):
        self.graph.add_node(("a", "b"))
        self.assertListEqual(list(self.graph.nodes()), [("a", "b")])

    def test_add_single_node_raises_error(self):
        self.assertRaises(TypeError, self.graph.add_node, "a")

    def test_add_multiple_nodes(self):
        self.graph.add_nodes_from([("a", "b"), ("b", "c")])
        self.assertListEqual(
            hf.recursive_sorted(self.graph.nodes()), [["a", "b"], ["b", "c"]]
        )

    def test_add_single_edge(self):
        self.graph.add_edge(("a", "b"), ("b", "c"))
        self.assertListEqual(
            hf.recursive_sorted(self.graph.nodes()), [["a", "b"], ["b", "c"]]
        )
        self.assertListEqual(
            sorted([node for edge in self.graph.edges() for node in edge]),
            [("a", "b"), ("b", "c")],
        )

    def test_add_single_edge_raises_error(self):
        self.assertRaises(ValueError, self.graph.add_edge, ("a", "b"), ("c", "d"))

    def tearDown(self):
        del self.graph


class TestClusterGraphFactorOperations(unittest.TestCase):
    def setUp(self):
        self.graph = ClusterGraph()

    def test_add_single_factor(self):
        self.graph.add_node(("a", "b"))
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1)
        self.assertCountEqual(self.graph.factors, [phi1])

    def test_add_single_factor_raises_error(self):
        self.graph.add_node(("a", "b"))
        phi1 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.assertRaises(ValueError, self.graph.add_factors, phi1)

    def test_add_multiple_factors(self):
        self.graph.add_edges_from([[("a", "b"), ("b", "c")]])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1, phi2)
        self.assertCountEqual(self.graph.factors, [phi1, phi2])

    def test_get_factors(self):
        self.graph.add_edges_from([[("a", "b"), ("b", "c")]])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.assertCountEqual(self.graph.get_factors(), [])
        self.graph.add_factors(phi1, phi2)
        self.assertEqual(self.graph.get_factors(node=("b", "a")), phi1)
        self.assertEqual(self.graph.get_factors(node=("b", "c")), phi2)
        self.assertCountEqual(self.graph.get_factors(), [phi1, phi2])

    def test_remove_factors(self):
        self.graph.add_edges_from([[("a", "b"), ("b", "c")]])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1, phi2)
        self.graph.remove_factors(phi1)
        self.assertCountEqual(self.graph.factors, [phi2])

    def test_get_partition_function(self):
        self.graph.add_edges_from([[("a", "b"), ("b", "c")]])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], range(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], range(4))
        self.graph.add_factors(phi1, phi2)
        self.assertEqual(self.graph.get_partition_function(), 22.0)

    def tearDown(self):
        del self.graph


class TestClusterGraphMethods(unittest.TestCase):
    def setUp(self):
        self.graph = ClusterGraph()

    def test_get_cardinality(self):
        self.graph.add_edges_from(
            [(("a", "b", "c"), ("a", "b")), (("a", "b", "c"), ("a", "c"))]
        )

        self.assertDictEqual(self.graph.get_cardinality(), {})

        phi1 = DiscreteFactor(["a", "b", "c"], [1, 2, 2], np.random.rand(4))
        self.graph.add_factors(phi1)
        self.assertDictEqual(self.graph.get_cardinality(), {"a": 1, "b": 2, "c": 2})
        self.graph.remove_factors(phi1)
        self.assertDictEqual(self.graph.get_cardinality(), {})

        phi1 = DiscreteFactor(["a", "b"], [1, 2], np.random.rand(2))
        phi2 = DiscreteFactor(["a", "c"], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi1, phi2)
        self.assertDictEqual(self.graph.get_cardinality(), {"a": 1, "b": 2, "c": 2})

        phi3 = DiscreteFactor(["a", "c"], [1, 1], np.random.rand(1))
        self.graph.add_factors(phi3)
        self.assertDictEqual(self.graph.get_cardinality(), {"c": 1, "b": 2, "a": 1})

        self.graph.remove_factors(phi1, phi2, phi3)
        self.assertDictEqual(self.graph.get_cardinality(), {})

    def test_get_cardinality_with_node(self):
        self.graph.add_edges_from([(("a", "b"), ("a", "c"))])
        phi1 = DiscreteFactor(["a", "b"], [1, 2], np.random.rand(2))
        phi2 = DiscreteFactor(["a", "c"], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi1, phi2)
        self.assertEqual(self.graph.get_cardinality("a"), 1)
        self.assertEqual(self.graph.get_cardinality("b"), 2)
        self.assertEqual(self.graph.get_cardinality("c"), 2)

    def test_check_model(self):
        self.graph.add_edges_from([(("a", "b"), ("a", "c"))])
        phi1 = DiscreteFactor(["a", "b"], [1, 2], np.random.rand(2))
        phi2 = DiscreteFactor(["a", "c"], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi1, phi2)
        self.assertTrue(self.graph.check_model())

        self.graph.remove_factors(phi2)
        phi2 = DiscreteFactor(["a", "c"], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi2)
        self.assertTrue(self.graph.check_model())

    def test_check_model1(self):
        self.graph.add_edges_from([(("a", "b"), ("a", "c")), (("a", "c"), ("a", "d"))])
        phi1 = DiscreteFactor(["a", "b"], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi1)
        self.assertRaises(ValueError, self.graph.check_model)
        phi2 = DiscreteFactor(["a", "c"], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi2)
        self.assertRaises(ValueError, self.graph.check_model)

    def test_check_model2(self):
        self.graph.add_edges_from([(("a", "b"), ("a", "c")), (("a", "c"), ("a", "d"))])

        phi1 = DiscreteFactor(["a", "b"], [1, 2], np.random.rand(2))
        phi2 = DiscreteFactor(["a", "c"], [3, 3], np.random.rand(9))
        phi3 = DiscreteFactor(["a", "d"], [4, 4], np.random.rand(16))
        self.graph.add_factors(phi1, phi2, phi3)
        self.assertRaises(ValueError, self.graph.check_model)
        self.graph.remove_factors(phi2)
        phi2 = DiscreteFactor(["a", "c"], [1, 3], np.random.rand(3))
        self.graph.add_factors(phi2)
        self.assertRaises(ValueError, self.graph.check_model)
        self.graph.remove_factors(phi3)

        phi3 = DiscreteFactor(["a", "d"], [1, 4], np.random.rand(4))
        self.graph.add_factors(phi3)
        self.assertTrue(self.graph.check_model())

    def test_copy_with_factors(self):
        self.graph.add_edges_from([[("a", "b"), ("b", "c")]])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1, phi2)
        graph_copy = self.graph.copy()
        self.assertIsInstance(graph_copy, ClusterGraph)
        self.assertEqual(
            hf.recursive_sorted(self.graph.nodes()),
            hf.recursive_sorted(graph_copy.nodes()),
        )
        self.assertEqual(
            hf.recursive_sorted(self.graph.edges()),
            hf.recursive_sorted(graph_copy.edges()),
        )
        self.assertTrue(graph_copy.check_model())
        self.assertEqual(self.graph.get_factors(), graph_copy.get_factors())
        self.graph.remove_factors(phi1, phi2)
        self.assertTrue(
            phi1 not in self.graph.factors and phi2 not in self.graph.factors
        )
        self.assertTrue(phi1 in graph_copy.factors and phi2 in graph_copy.factors)
        self.graph.add_factors(phi1, phi2)
        self.graph.factors[0] = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        self.assertNotEqual(self.graph.get_factors()[0], graph_copy.get_factors()[0])
        self.assertNotEqual(self.graph.factors, graph_copy.factors)

    def test_copy_without_factors(self):
        self.graph.add_nodes_from([("a", "b", "c"), ("a", "b"), ("a", "c")])
        self.graph.add_edges_from(
            [(("a", "b", "c"), ("a", "b")), (("a", "b", "c"), ("a", "c"))]
        )
        graph_copy = self.graph.copy()
        self.graph.remove_edge(("a", "b", "c"), ("a", "c"))
        self.assertFalse(self.graph.has_edge(("a", "b", "c"), ("a", "c")))
        self.assertTrue(graph_copy.has_edge(("a", "b", "c"), ("a", "c")))
        self.graph.remove_node(("a", "c"))
        self.assertFalse(self.graph.has_node(("a", "c")))
        self.assertTrue(graph_copy.has_node(("a", "c")))
        self.graph.add_node(("c", "d"))
        self.assertTrue(self.graph.has_node(("c", "d")))
        self.assertFalse(graph_copy.has_node(("c", "d")))

    def tearDown(self):
        del self.graph
