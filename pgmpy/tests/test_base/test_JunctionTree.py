# import unittest
# from pgmpy import MarkovModel as mm
# from pgmpy import factors
#
#
# class TestOperations(unittest.TestCase):
#     def setUp(self):
#         self.graph = mm.MarkovModel([('d', 'g'), ('i', 'g')])
#         self.graph.add_states(
#             {'d': ['easy', 'hard'], 'g': ['A', 'B', 'C'], 'i': ['dumb', 'smart']})
#
#     # def test_temp(self):
#     #     from pgmpy.factors.factors import factors
#     #     f1 = factors(['d', 'g'], [2, 3], [1, 2, 3, 4, 5, 6])
#     #     f2 = factors([], [], [])
#     #     print("Multiplying the two factors ")
#     #     print(f1)
#     #     print(f2)
#     #     print("f1 * f2")
#     #     print(f1.product(f2))
#
#     def test_temp1(self):
#         from pgmpy.factors.factors import factors
#         f1 = factors(['d', 'g'], [2,3], [1,2,3,4,5,6])
#         f2 = factors(['i', 'g'], [2,3],[1,2,3,4,5,6])
#         print("Multiplying the two factors ")
#         print(f1)
#         print(f2)
#         print("f1 * f2")
#         print(f1.product(f2))
#
#     def test_normalization_pos_dist(self):
#         self.graph.add_factor(['d', 'g'], [1, 2, 3, 4, 5, 6])
#         self.graph.add_factor(['i', 'g'], [1, 2, 3, 4, 5, 6])
#         jt = self.graph.make_jt(2)
#         #jt.print_graph("printing the junction tree")
#         val = jt.normalization_constant()
#         self.assertAlmostEqual(val, 155.0)
#         #print("Value using junction tree " + str(val))
#         #print("value using basic brute force " + str(self.graph.normalization_constant_brute_force()))
#
#     def test_marginal_prob(self):
#         self.graph.add_factor(['d', 'g'], [1, 2, 3, 4, 5, 6])
#         self.graph.add_factor(['i', 'g'], [1, 2, 3, 4, 5, 6])
#         jt = self.graph.make_jt(2)
#         #print("norm "+str(jt.normalization_constant()))
#         res_factor = jt.marginal_prob('d')
#         ex_factor = factors.factors(['d'], [2], [71.0, 92.0])
#         self.assertEqual(res_factor, ex_factor)
#         #print(factor)
#
#     def test_marginalize(self):
#         from pgmpy.factors.factors import factors
#         fac = factors(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
#         print(fac)
#         fac.marginalize("x1")
#         print(fac)
#
#     def test_maximize(self):
#         from pgmpy.factors.factors import factors
#         from pgmpy.factors.factors import factors
#         fac = factors(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
#         print(fac)
#         fac.maximize_on_variables("x1")
#         print(fac)
#
#     def tearDown(self):
#         del self.graph
#
# if __name__ == '__main__':
#     unittest.main()
