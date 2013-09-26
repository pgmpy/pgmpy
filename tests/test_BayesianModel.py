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
    
    def test_add_edges_both_string(self):
        self.G.add_nodes('a', 'b')
        self.G.add_edges('a', 'b')
        self.assertItemsEqual(self.G.edges(), [('a', 'b')])

    def test_add_edges_multiple_times(self):
        self.G.add_nodes('a', 'b', 'c', 'd')
        self.G.add_edges('a', ('c', 'd'))
        self.G.add_edges('b', ('c', 'd'))
        self.assertItemsEqual(self.G.edges(), [('a', 'c'), ('a', 'd'),
            ('b', 'c'), ('b', 'd')])

    def tearDown(self):
        del self.G


class TestNodeProperties(unittest.TestCase):

    def setUp(self):
        G = bm.BayesianModel()
        G.add_nodes('a', 'b', 'c', 'd', 'e')
        G.add_edges(('a', 'b'), ('c', 'd'))

    def test_parents(self):
        self.assertItemsEqual(self.G.node['c']['_parents'], ['a', 'b'])
        self.assertItemsEqual(self.G.node['d']['_parents'], ['a', 'b'])
        self.assertItemsEqual(self.G.node['a']['_parents'], [])
        self.assertItemsEqal(self.G.nodes['b']['_paresnts'], [])
            
    def tearDown(self):
        del self.G

if __name__ == '__main__':
        unittest.main()
