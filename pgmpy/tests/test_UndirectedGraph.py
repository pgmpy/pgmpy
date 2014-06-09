__author__ = 'navin'

import unittest
from pgmpy import MarkovModel as mm

class TestBaseModelCreation(unittest.TestCase):

    def setUp(self):
        pass


    def test_is_triangulated(self):
        G= mm.UndirectedGraph()
        G.read_simple_format("graph_triangulated")
        self.assertTrue(G.is_triangulated())
        G.read_simple_format("graph_not_triangulated")
        self.assertFalse(G.is_triangulated())


    def test_triangulation_all_techniques(self):
        i=2
        while True:
            G = mm.UndirectedGraph()
            G.read_simple_format("graph")
            ret = G.jt_techniques(i,False,True)
            if not ret:
                break
            self.assertTrue(G.is_triangulated())
            i+=1

    def test_check_clique(self):
        G=mm.UndirectedGraph()
        G.read_simple_format("clique_graph")
        ret= G.check_clique()

    def test_spannningTree(self):
        pass

    def tearDown(self):
        pass


if __name__ == '__main__':
        unittest.main()
