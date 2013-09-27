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
        self.assertSameElements(self.G.nodes(), ['a', 'b', 'c', 'd'])

    def test_add_nodes_non_string(self):
        self.assertSameElements(TypeError, self.G.add_nodes, [1, 2, 3, 4])

    def test_add_edges_both_tuples(self):
        self.G.add_nodes('a', 'b', 'c', 'd')
        self.G.add_edges(('a', 'b'), ('c', 'd'))
        self.assertSameElements(self.G.edges(), [('a', 'c'), ('a', 'd'), 
            ('b', 'c'), ('b', 'd')])

    def test_add_edges_tail_string(self):
        self.G.add_nodes('a', 'b', 'c', 'd')
        self.G.add_edges('a', ('b', 'c'))
        self.assertSameElements(self.G.edges(), [('a', 'b'), ('a', 'c')])

    def test_add_edges_head_string(self):
        self.G.add_nodes('a', 'b', 'c', 'd')
        self.G.add_edges(('a', 'b'), 'c')
        self.assertSameElements(self.G.edges(), [('a', 'c'), ('b', 'c')])
    
    def test_add_edges_both_string(self):
        self.G.add_nodes('a', 'b')
        self.G.add_edges('a', 'b')
        self.assertSameElements(self.G.edges(), [('a', 'b')])

    def test_add_edges_multiple_times(self):
        self.G.add_nodes('a', 'b', 'c', 'd')
        self.G.add_edges('a', ('c', 'd'))
        self.G.add_edges('b', ('c', 'd'))
        self.assertSameElements(self.G.edges(), [('a', 'c'), ('a', 'd'),
            ('b', 'c'), ('b', 'd')])

    def tearDown(self):
        del self.G


class TestNodeProperties(unittest.TestCase):

    def setUp(self):
        self.G = bm.BayesianModel()
        self.G.add_nodes('a', 'b', 'c', 'd', 'e')
        self.G.add_edges(('a', 'b'), ('c', 'd'))

    def test_parents(self):
        self.assertSameElements(self.G.node['c']['_parents'], ['a', 'b'])
        self.assertSameElements(self.G.node['d']['_parents'], ['a', 'b'])
#        self.assertRaises(KeyError, self.G.node['a']['_parents'])
#        self.assertRaises(KeyError, self.G.node['b']['_parents'])
            
    def tearDown(self):
        del self.G

if __name__ == '__main__':
        unittest.main()
