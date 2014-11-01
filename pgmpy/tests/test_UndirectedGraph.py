import unittest
from pgmpy.base import UndirectedGraph


class TestBaseModelCreation(unittest.TestCase):
    def setUp(self):
        pass

    def test_is_triangulated(self):
        #triangulated graph
        graph = UndirectedGraph([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5),
                                    (3, 5), (3, 7), (6, 7), (6, 9), (9, 8), (7, 8), (6, 8)])
        self.assertTrue(graph.is_triangulated())
        #graph_not_triangulated
        graph = UndirectedGraph([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5),
                                    (3, 5), (3, 7), (6, 7), (6, 9), (9, 8), (7, 8),
                                    (6, 8), (1, 6)])
        self.assertFalse(graph.is_triangulated())

    def test_triangulation_all_heuristics(self):
        i = 2
        while True:
            graph = UndirectedGraph([(0, 1), (0, 3), (0, 8), (1, 2), (1, 4), (1, 8),
                                        (2, 4), (2, 6), (2, 7), (3, 8), (3, 9), (4, 7),
                                        (4, 8), (5, 8), (5, 9), (5, 10), (6, 7), (7, 10),
                                        (8, 10)])
            #graph.read_simple_format("test_graphs/graph")
            #print(i)
            ret = graph.jt_techniques(i, False, True)
            if not ret:
                break
            self.assertTrue(graph.is_triangulated())
            i += 1

    def test_jt_from_chordal_graph(self):
        #small triangulated graph
        graph = UndirectedGraph([(0, 1), (1, 2), (2, 0), (3, 4),
                                    (4, 5), (5, 3), (0, 3)])
        ret = graph.jt_techniques(0, True, True)
        self.assertTrue(ret.is_triangulated())

    def test_jt_tree_width(self):
        #small triangulated graph
        graph = UndirectedGraph([(0, 1), (1, 2), (2, 0), (3, 4),
                                    (4, 5), (5, 3), (0, 3)])
        ret = graph.jt_tree_width(0)
        self.assertEqual(ret, 2)
        res = [4, 4, 7, 5]
        i = 2
        while True:
            graph = UndirectedGraph([(0, 1), (0, 3), (0, 8), (1, 2), (1, 4), (1, 8),
                                        (2, 4), (2, 6), (2, 7), (3, 8), (3, 9), (4, 7),
                                        (4, 8), (5, 8), (5, 9), (5, 10), (6, 7), (7, 10),
                                        (8, 10)])
            #graph.read_simple_format("test_graphs/graph")
            ret = graph.jt_tree_width(i)
            if not ret:
                break
            #print("heu num "+str(i))
            #print(ret)
            self.assertEqual(ret, res[i - 2])
            i += 1

    def test_check_clique(self):
        #clique graph
        graph = UndirectedGraph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        ret = graph.check_clique(graph.nodes())
        self.assertTrue(ret)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
