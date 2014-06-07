__author__ = 'navin'

import unittest
from pgmpy import MarkovModel as mm
from pgmpy.MarkovModel.UndirectedGraph import UndirectedGraph

class TestBaseModelCreation(unittest.TestCase):

    def setUp(self):
        pass


    def test_is_triangulated(self):
        G= UndirectedGraph()
        G.readSimpleFormatGraph("graph_triangulated")
        self.assertTrue(G.isTriangulated())
        G.readSimpleFormatGraph("graph_not_triangulated")
        self.assertFalse(G.isTriangulated())

    def test_triangulation_all_techniques(self):
        i=2
        while True:
            G = UndirectedGraph()
            G.readSimpleFormatGraph("graph")
            ret = G.junctionTreeTechniques(i,False,True)
            if not ret:
                break
            self.assertTrue(G.isTriangulated())
            i+=1

    def test_spannningTree(self):
        pass

    def tearDown(self):
        pass


if __name__ == '__main__':
        unittest.main()
