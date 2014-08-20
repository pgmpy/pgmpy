import unittest
from pgmpy import MarkovModel as mm
from pgmpy import Factor


class TestOperations(unittest.TestCase):
    def setUp(self):
        self.graph = mm.MarkovModel([('d', 'g'), ('i', 'g')])
        self.graph.add_states(
            {'d': ['easy', 'hard'], 'g': ['A', 'B', 'C'], 'i': ['dumb', 'smart']})
        self.graph2 = mm.MarkovModel([('a', 'b'), ('b', 'c'), ('a', 'c')])
        self.graph2.add_states({'a': ['0', '1'], 'b': ['0', '1'], 'c': ['0', '1']})

    # def test_temp(self):
    #     from pgmpy.Factor.Factor import Factor
    #     f1 = Factor(['d', 'g'], [2, 3], [1, 2, 3, 4, 5, 6])
    #     f2 = Factor([], [], [])
    #     print("Multiplying the two factors ")
    #     print(f1)
    #     print(f2)
    #     print("f1 * f2")
    #     print(f1.product(f2))

    def test_temp1(self):
        from pgmpy.Factor.Factor import Factor

        f1 = Factor(['d', 'g'], [2, 3], [1, 2, 3, 4, 5, 6])
        f2 = Factor(['i', 'g'], [2, 3], [1, 2, 3, 4, 5, 6])
        print("Multiplying the two factors ")
        print(f1)
        print(f2)
        print("f1 * f2")
        print(f1.product(f2))

    def test_normalization_pos_dist(self):
        self.graph.add_factor(['d', 'g'], [1, 2, 3, 4, 5, 6])
        self.graph.add_factor(['i', 'g'], [1, 2, 3, 4, 5, 6])
        jt = self.graph.make_jt(2)
        #jt.print_graph("printing the junction tree")
        val = jt.normalization_constant()
        self.assertAlmostEqual(val, 155.0)
        #print("Value using junction tree " + str(val))
        #print("value using basic brute force " + str(self.graph.normalization_constant_brute_force()))

    def test_marginal_prob(self):
        self.graph.add_factor(['d', 'g'], [1, 2, 3, 4, 5, 6])
        self.graph.add_factor(['i', 'g'], [1, 2, 3, 4, 5, 6])
        jt = self.graph.make_jt(2)
        #print("norm "+str(jt.normalization_constant()))
        res_factor = jt.marginal_prob('d')
        ex_factor = Factor.Factor(['d'], [2], [46.0, 109.0])
        self.assertEqual(res_factor, ex_factor)

    def test_map(self):
        self.graph2.add_factor(['a', 'b'], [5, 1, 1, 2])
        self.graph2.add_factor(['b', 'c'], [1, 1, 1, 5])
        self.graph2.add_factor(['a', 'c'], [1, 1, 1, 5])
        jt = self.graph2.make_jt(2)
        res_factor = jt.map()
        self.assertEqual(res_factor, [('a', 1), ('c', 1), ('b', 1)])

    def test_map_2(self):
        self.graph.add_factor(['d', 'g'], [1, 2, 3, 4, 5, 6])
        self.graph.add_factor(['i', 'g'], [1, 2, 3, 4, 5, 6])
        jt = self.graph.make_jt(2)
        res_factor = jt.map()
        self.assertEqual(res_factor, [('i', 1), ('d', 1), ('g', 2)])

    def tearDown(self):
        del self.graph


if __name__ == '__main__':
    unittest.main()
