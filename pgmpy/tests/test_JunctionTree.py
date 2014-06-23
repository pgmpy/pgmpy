import unittest
from pgmpy import MarkovModel as mm


class test_normalization(unittest.TestCase):
    def setUp(self):
        self.graph = mm.MarkovModel([('d', 'g'), ('i', 'g')])
        self.graph.add_states(
            {'d': ['easy', 'hard'], 'g': ['A', 'B', 'C'], 'i': ['dumb', 'smart']})

    def test_temp(self):
        from pgmpy.Factor.Factor import Factor
        f1 = Factor(['d', 'g'], [2,3], [1,2,3,4,5,6])
        f2 = Factor(['i', 'g'], [2,3],[1,2,3,4,5,6])
        print("Multiplying the two factors ")
        print(f1)
        print(f2)
        print("f1 * f2")
        print(f1.product(f2))

    def test_normalization_pos_dist(self):
        f1 = self.graph.add_factor(['d', 'g'], [1,2,3,4,5,6])
        f2 = self.graph.add_factor(['i', 'g'], [1,2,3,4,5,6])
        print("Multiplying the two factors ")
        print(f1)
        print(f2)
        print("f1 * f2")
        print(f1.product(f2))
        #print(f1)
        #print(f2)
        jt = self.graph.make_jt(2)
        jt.print_graph("printing the junction tree")
        val = jt.normalization_constant()
        print("Value using junction tree "+str(val))
        print("value using basic brute force "+str(self.graph.normalization_constant_brute_force()))


    def tearDown(self):
        del self.graph


if __name__ == '__main__':
    unittest.main()
