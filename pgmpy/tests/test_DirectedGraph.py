import unittest
from pgmpy import MarkovModel as mm


class TestBaseModelCreation(unittest.TestCase):
    def setUp(self):
        self.g = mm.DirectedGraph()
        self.g.add_nodes_from(['a','b'])

    def test_get_node_name_with_suffix(self):
        node_name = self.g.get_node_name_with_suffix("c")
        self.assertEqual(node_name, "c")
        node_name = self.g.get_node_name_with_suffix("a")
        self.assertEqual(node_name, "a0")

    def test_get_flow_capacity(self):
        self.g.add_edge('a','b',capacity = 5)
        cp = self.g.get_flow_capacity('a','b')
        self.assertEqual(cp, 5)

    def test_add_to_flow_edge_capacity(self):
        self.g.add_edge('a','b',capacity = 5)
        self.g.add_to_flow_edge_capacity('a','b',5)
        cp = self.g.get_flow_capacity('a','b')
        self.assertEqual(cp, 10)

    def test_flow_dfs(self):
        self.g.add_edge('a','b',capacity = 5, flow=1)
        dfs_set = set()
        self.g.flow_dfs('a',dfs_set)
        self.assertListEqual(list(dfs_set),['a','b'])

    def tearDown(self):
        del self.g

class TestMaxFlow(unittest.TestCase):
    def setUp(self):
        self.graph = mm.DirectedGraph()
        for node in "sopqrt":
            self.graph.add_node(node)
        self.graph.add_to_flow_edge_capacity('s','o',3)
        self.graph.add_to_flow_edge_capacity('s','p',3)
        self.graph.add_to_flow_edge_capacity('o','p',2)
        self.graph.add_to_flow_edge_capacity('o','q',3)
        self.graph.add_to_flow_edge_capacity('p','r',2)
        self.graph.add_to_flow_edge_capacity('r','t',3)
        self.graph.add_to_flow_edge_capacity('q','r',4)
        self.graph.add_to_flow_edge_capacity('q','t',2)

    def test_max_flow(self):
        v = self.graph.max_flow_ford_fulkerson('s','t')
        self.assertEquals(v, 5)

    def tearDown(self):
        del self.graph

if __name__ == '__main__':
    unittest.main()
