__author__ = 'navin'

import unittest
from pgmpy import MarkovModel as mm


class TestBaseModelCreation(unittest.TestCase):
    def setUp(self):
        pass

    def test_is_triangulated(self):
        graph = mm.UndirectedGraph()
        graph.read_simple_format("test_graphs/graph_triangulated")
        self.assertTrue(graph.is_triangulated())
        graph.read_simple_format("test_graphs/graph_not_triangulated")
        self.assertFalse(graph.is_triangulated())

    def test_triangulation_all_heuristics(self):
        i = 2
        while True:
            graph = mm.UndirectedGraph()
            graph.read_simple_format("test_graphs/graph")
            ret = graph.jt_techniques(i, False, True)
            if not ret:
                break
            self.assertTrue(graph.is_triangulated())
            i += 1

    def test_jt_from_chordal_graph(self):
        graph = mm.UndirectedGraph()
        graph.read_simple_format("test_graphs/small_triangulated_graph")
        ret = graph.jt_techniques(0, True, True)
        self.assertTrue(ret.is_triangulated())

    def test_jt_tree_width(self):
        graph = mm.UndirectedGraph()
        graph.read_simple_format("test_graphs/small_triangulated_graph")
        ret = graph.jt_tree_width(0)
        self.assertEqual(ret, 2)
        res = [4, 4, 7, 5]
        i = 2
        while True:
            graph = mm.UndirectedGraph()
            graph.read_simple_format("test_graphs/graph")
            ret = graph.jt_tree_width(i)
            if not ret:
                break
            #print("heu num "+str(i))
            #print(ret)
            self.assertEqual(ret, res[i-2])
            i += 1

    def test_check_clique(self):
        graph = mm.UndirectedGraph()
        graph.read_simple_format("test_graphs/clique_graph")
        ret = graph.check_clique(graph.nodes())

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
