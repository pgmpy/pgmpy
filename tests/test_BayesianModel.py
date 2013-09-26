import unittest
import BayesianModel as bm
import networkx as nx


class TestModel(unittest.TestCase):

    def setUp(self):
        self.G = bm.BayesianModel()

    def test_class_init(self):
        self.assertIsInstance(self.G, nx.DiGraph)

    def test_add_nodes_string(self):
        self.G.add_nodes('a', 'b', 'c', 'd')
        self.assertItemsEqual(self.G.nodes(), ['a', 'b', 'c', 'd'])

    def test_add_nodes_non_string(self):
        self.assertRaises(TypeError, self.G.add_nodes, [1, 2, 3, 4])

    def test_add_edges_both_tuples(self):
        self.G.add_nodes('a', 'b', 'c', 'd')
        self.G.add_edges(('a', 'b'), ('c', 'd'))
        self.assertItemsEqual(self.G.edges(), [('a', 'c'), ('a', 'd'), 
            ('b', 'c'), ('b', 'd')])

    def test_add_edges_tail_string(self):
        self.G.add_nodes('a', 'b', 'c', 'd')
        self.G.add_edges('a', ('b', 'c'))
        self.assertItemsEqual(self.G.edges(), [('a', 'b'), ('a', 'c')])

    def test_add_edges_head_string(self):
        self.G.add_nodes('a', 'b', 'c', 'd')
        self.G.add_edges(('a', 'b'), 'c')
        self.assertItemsEqual(self.G.edges(), [('a', 'c'), ('b', 'c')])

    def tearDown(self):
        del self.G

if __name__ == '__main__':
        unittest.main()
