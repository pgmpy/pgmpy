import unittest
import itertools
from collections import OrderedDict, namedtuple

import numpy as np
import numpy.testing as np_test

from pgmpy.factors import Factor
from pgmpy.factors import factor_product
from pgmpy.factors import factor_divide
from pgmpy.factors.CPD import TabularCPD
from pgmpy import exceptions


State = namedtuple('State', ['var', 'state'])


class TestFactorInit(unittest.TestCase):
    def test_class_init(self):
        phi = Factor(['x1', 'x2', 'x3'], [2, 2, 2], np.ones(8))
        self.assertEqual(phi.variables, ['x1', 'x2', 'x3'])
        np_test.assert_array_equal(phi.cardinality, np.array([2, 2, 2]))
        np_test.assert_array_equal(phi.values, np.ones(8).reshape(2, 2, 2))

    def test_class_init1(self):
        phi = Factor([1, 2, 3], [2, 3, 2], np.arange(12))
        self.assertEqual(phi.variables, [1, 2, 3])
        np_test.assert_array_equal(phi.cardinality, np.array([2, 3, 2]))
        np_test.assert_array_equal(phi.values, np.arange(12).reshape(2, 3, 2))

    def test_class_init_sizeerror(self):
        self.assertRaises(ValueError, Factor, ['x1', 'x2', 'x3'], [2, 2, 2], np.ones(9))

    def test_init_size_var_card_not_equal(self):
        self.assertRaises(ValueError, Factor, ['x1', 'x2'], [2], np.ones(2))


class TestFactorMethods(unittest.TestCase):
    def setUp(self):
        self.phi = Factor(['x1', 'x2', 'x3'], [2, 2, 2], np.random.uniform(5, 10, size=8))
        self.phi1 = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        self.phi2 = Factor([('x1', 0), ('x2', 0), ('x3', 0)], [2, 3, 2], range(12))

    def test_scope(self):
        self.assertListEqual(self.phi.scope(), ['x1', 'x2', 'x3'])
        self.assertListEqual(self.phi1.scope(), ['x1', 'x2', 'x3'])

#    def test_assignment(self):
#        self.assertListEqual(self.phi.assignment([0]), [[State('x1', 0), State('x2', 0), State('x3', 0)]])
#        self.assertListEqual(self.phi.assignment([4, 5, 6]), [[State('x1', 1), State('x2', 0), State('x3', 0)],
#                                                              [State('x1', 1), State('x2', 0), State('x3', 1)],
#                                                              [State('x1', 1), State('x2', 1), State('x3', 0)]])

#        self.assertListEqual(self.phi1.assignment(np.array([4, 5, 6])),
#                             [[State('x1', 0), State('x2', 2), State('x3', 0)],
#                              [State('x1', 0), State('x2', 2), State('x3', 1)],
#                              [State('x1', 1), State('x2', 0), State('x3', 0)]])

#    def test_assignment_indexerror(self):
#        self.assertRaises(IndexError, self.phi.assignment, [10])
#        self.assertRaises(IndexError, self.phi.assignment, [1, 3, 10, 5])
#        self.assertRaises(IndexError, self.phi.assignment, np.array([1, 3, 10, 5]))

    def test_get_cardinality(self):
        self.assertEqual(self.phi.get_cardinality(['x1']), {'x1': 2})
        self.assertEqual(self.phi.get_cardinality(['x2']), {'x2': 2})
        self.assertEqual(self.phi.get_cardinality(['x3']), {'x3': 2})
        self.assertEqual(self.phi.get_cardinality(['x1', 'x2']), {'x1': 2, 'x2': 2})
        self.assertEqual(self.phi.get_cardinality(['x1', 'x3']), {'x1': 2, 'x3': 2})
        self.assertEqual(self.phi.get_cardinality(['x1', 'x2', 'x3']), {'x1': 2, 'x2': 2, 'x3': 2})

    def test_get_cardinality_scopeerror(self):
        self.assertRaises(ValueError, self.phi.get_cardinality, 'x4')

    def test_marginalize(self):
        self.phi1.marginalize(['x1'])
        np_test.assert_array_equal(self.phi1.values, np.array([[6, 8],
                                                               [10, 12],
                                                               [14, 16]]))
        self.phi1.marginalize(['x2'])
        np_test.assert_array_equal(self.phi1.values, np.array([30, 36]))
        self.phi1.marginalize(['x3'])
        np_test.assert_array_equal(self.phi1.values, np.array(66))

    def test_marginalize_scopeerror(self):
        self.assertRaises(ValueError, self.phi.marginalize, ['x4'])
        self.assertRaises(ValueError, self.phi.marginalize, ['x4'])

        self.phi.marginalize(['x1'])
        self.assertRaises(ValueError, self.phi.marginalize, 'x1')

    def test_normalize(self):
        self.phi1.normalize()
        np_test.assert_almost_equal(self.phi1.values,
                                    np.array([[[0, 0.01515152],
                                               [0.03030303, 0.04545455],
                                               [0.06060606, 0.07575758]],
                                              [[0.09090909, 0.10606061],
                                               [0.12121212, 0.13636364],
                                               [0.15151515, 0.16666667]]]))

    def test_reduce(self):
        self.phi1.reduce([('x1', 0), ('x2', 0)])
        np_test.assert_array_equal(self.phi1.values, np.array([0, 1]))

    def test_reduce1(self):
        self.phi1.reduce([('x2', 0), ('x1', 0)])
        np_test.assert_array_equal(self.phi1.values, np.array([0, 1]))

    def test_reduce2(self):
        self.phi2.reduce([(('x2', 0), 0), (('x1', 0), 0)])
        np_test.assert_array_equal(self.phi2.values, np.array([0, 1]))

    @unittest.skip
    def test_complete_reduce(self):
        self.phi1.reduce([('x1', 0), ('x2', 0), ('x3', 1)])
        np_test.assert_array_equal(self.phi1.values, np.array([1]))
        np_test.assert_array_equal(self.phi1.cardinality, np.array([]))
        np_test.assert_array_equal(self.phi1.variables, OrderedDict())

    def test_reduce_typeerror(self):
        self.assertRaises(ValueError, self.phi1.reduce, 'x10')
        self.assertRaises(ValueError, self.phi1.reduce, ['x10'])

    def test_reduce_scopeerror(self):
        self.assertRaises(ValueError, self.phi1.reduce, [('x4', 1)])

    def test_reduce_sizeerror(self):
        self.assertRaises(IndexError, self.phi1.reduce, [('x3', 5)])

    def test_identity_factor(self):
        identity_factor = self.phi.identity_factor()
        self.assertEquals(list(identity_factor.variables), ['x1', 'x2', 'x3'])
        np_test.assert_array_equal(identity_factor.cardinality, [2, 2, 2])
        np_test.assert_array_equal(identity_factor.values, np.ones(8).reshape(2, 2, 2))

    def test_factor_product(self):
        phi = Factor(['x1', 'x2'], [2, 2], range(4))
        phi1 = Factor(['x3', 'x4'], [2, 2], range(4))
        prod = factor_product(phi, phi1)
        expected_factor = Factor(['x1', 'x2', 'x3', 'x4'], [2, 2, 2, 2], [0, 0, 0, 0, 0, 1,
                                                                          2, 3, 0, 2, 4, 6,
                                                                          0, 3, 6, 9])
        self.assertEqual(prod, expected_factor)
        self.assertEqual(sorted(prod.variables), ['x1', 'x2', 'x3', 'x4'])

        phi = Factor(['x1', 'x2'], [3, 2], range(6))
        phi1 = Factor(['x2', 'x3'], [2, 2], range(4))
        prod = factor_product(phi, phi1)
        expected_factor = Factor(['x1', 'x2', 'x3'], [3, 2, 2], [0, 0, 2, 3, 0, 2,
                                                                 6, 9, 0, 4, 10, 15])
        np_test.assert_almost_equal(prod.values,
                                   np.array([0, 0, 2, 3, 0, 2,
                                             6, 9, 0, 4, 10, 15]).reshape(3, 2, 2))
        self.assertEqual(sorted(prod.variables), ['x1', 'x2', 'x3'])

    def test_factor_product2(self):
        from pgmpy import factors
        phi = factors.Factor(['x1', 'x2'], [2, 2], range(4))
        phi1 = factors.Factor(['x3', 'x4'], [2, 2], range(4))
        prod = phi.product(phi1, inplace=False)
        expected_factor = Factor(['x1', 'x2', 'x3', 'x4'], [2, 2, 2, 2],
                                 [0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 4, 6, 0, 3, 6, 9])
        self.assertEqual(prod, expected_factor)
        self.assertEqual(sorted(prod.variables), ['x1', 'x2', 'x3', 'x4'])

        phi = Factor(['x1', 'x2'], [3, 2], range(6))
        phi1 = Factor(['x2', 'x3'], [2, 2], range(4))
        prod = phi.product(phi1, inplace=False)
        expected_factor = Factor(['x1', 'x2', 'x3'], [3, 2, 2],
                                 [0, 0, 2, 3, 0, 2, 6, 9, 0, 4, 10, 15])
        self.assertEqual(prod, expected_factor)
        self.assertEqual(sorted(prod.variables), ['x1', 'x2', 'x3'])

    def test_factor_product_non_factor_arg(self):
        self.assertRaises(TypeError, factor_product, 1, 2)

    def test_factor_mul(self):
        phi = Factor(['x1', 'x2'], [2, 2], range(4))
        phi1 = Factor(['x3', 'x4'], [2, 2], range(4))
        prod = phi * phi1

        sorted_vars = ['x1', 'x2', 'x3', 'x4']
        for axis in range(prod.values.ndim):
            exchange_index = prod.variables.index(sorted_vars[axis])
            prod.variables[axis], prod.variables[exchange_index] = prod.variables[exchange_index], prod.variables[axis]
            prod.values = prod.values.swapaxes(axis, exchange_index)

        np_test.assert_almost_equal(prod.values.ravel(),
                                    np.array([0, 0, 0, 0, 0, 1,
                                              2, 3, 0, 2, 4, 6,
                                              0, 3, 6, 9]))

        self.assertEqual(prod.variables, ['x1', 'x2', 'x3', 'x4'])

    def test_factor_divide(self):
        phi1 = Factor(['x1', 'x2'], [2, 2], [1, 2, 2, 4])
        phi2 = Factor(['x1'], [2], [1, 2])
        div = phi1.divide(phi2, inplace=False)
        phi3 = Factor(['x1', 'x2'], [2, 2], [1, 2, 1, 2])
        self.assertEqual(phi3, div)

    def test_factor_divide_truediv(self):
        phi1 = Factor(['x1', 'x2'], [2, 2], [1, 2, 2, 4])
        phi2 = Factor(['x1'], [2], [1, 2])
        div = phi1 / phi2
        phi3 = Factor(['x1', 'x2'], [2, 2], [1, 2, 1, 2])
        self.assertEqual(phi3, div)

    def test_factor_divide_invalid(self):
        phi1 = Factor(['x1', 'x2'], [2, 2], [1, 2, 3, 4])
        phi2 = Factor(['x1'], [2], [0, 2])
        div = phi1.divide(phi2, inplace=False)
        np_test.assert_array_equal(div.values.ravel(), np.array([np.inf, np.inf, 1.5, 2]))

    def test_factor_divide_no_common_scope(self):
        phi1 = Factor(['x1', 'x2'], [2, 2], [1, 2, 3, 4])
        phi2 = Factor(['x3'], [2], [0, 2])
        self.assertRaises(ValueError, factor_divide, phi1, phi2)

    def test_factor_divide_non_factor_arg(self):
        self.assertRaises(TypeError, factor_divide, 1, 1)

    def test_eq(self):
        self.assertFalse(self.phi == self.phi1)
        self.assertTrue(self.phi == self.phi)
        self.assertTrue(self.phi1 == self.phi1)

    def test_eq1(self):
        phi1 = Factor(['x1', 'x2', 'x3'], [2, 4, 3], range(24))
        phi2 = Factor(['x2', 'x1', 'x3'], [4, 2, 3], [0, 1, 2, 12, 13, 14, 3,
                                                      4, 5, 15, 16, 17, 6, 7,
                                                      8, 18, 19, 20, 9, 10, 11,
                                                      21, 22, 23])
        self.assertTrue(phi1, phi2)

    def test_maximize1(self):
        self.phi1.maximize(['x1'])
        self.assertEqual(self.phi1, Factor(['x2', 'x3'], [3, 2], [6, 7, 8, 9, 10, 11]))
        self.phi1.maximize(['x2'])
        self.assertEqual(self.phi1, Factor(['x3'], [2], [10, 11]))

    def test_maximize2(self):
        self.phi1.maximize(['x1', 'x2'])
        self.assertEqual(self.phi1, Factor(['x3'], [2], [10, 11]))

    def test_maximize3(self):
        self.phi2 = Factor(['x1', 'x2', 'x3'], [3, 2, 2], [0.25, 0.35, 0.08, 0.16, 0.05, 0.07,
                                                           0.00, 0.00, 0.15, 0.21, 0.08, 0.18])
        self.phi2.maximize(['x2'])
        self.assertEqual(self.phi2, Factor(['x1', 'x3'], [3, 2], [0.25, 0.35, 0.05,
                                                                  0.07, 0.15, 0.21]))

    def tearDown(self):
        del self.phi
        del self.phi1


class TestTabularCPDInit(unittest.TestCase):
    def test_cpd_init(self):
        cpd = TabularCPD('grade', 3, [[0.1, 0.1, 0.1]])
        self.assertEqual(cpd.variable, 'grade')
        self.assertEqual(cpd.variable_card, 3)
        self.assertEqual(list(cpd.variables), ['grade'])
        np_test.assert_array_equal(cpd.cardinality, np.array([3]))
        np_test.assert_array_almost_equal(cpd.values, np.array([0.1, 0.1, 0.1]))

        cpd = TabularCPD('grade', 3, [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                      [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
                         evidence=['intel', 'diff'], evidence_card=[3, 2])
        self.assertEqual(cpd.variable, 'grade')
        self.assertEqual(cpd.variable_card, 3)
        np_test.assert_array_equal(cpd.cardinality, np.array([3, 2, 3]))
        self.assertListEqual(list(cpd.variables), ['grade', 'diff', 'intel'])
        np_test.assert_array_equal(cpd.values, np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                                         0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                                         0.8, 0.8, 0.8, 0.8, 0.8, 0.8]).reshape(3, 2, 3))

        cpd = TabularCPD('grade', 3, [[0.1, 0.1],
                                      [0.1, 0.1],
                                      [0.8, 0.8]],
                         evidence='evi1', evidence_card=2)
        self.assertEqual(cpd.variable, 'grade')
        self.assertEqual(cpd.variable_card, 3)
        np_test.assert_array_equal(cpd.cardinality, np.array([3, 2]))
        self.assertListEqual(list(cpd.variables), ['grade', 'evi1'])
        np_test.assert_array_equal(cpd.values, np.array([0.1, 0.1,
                                                         0.1, 0.1,
                                                         0.8, 0.8]).reshape(3, 2))

    def test_cpd_init_event_card_not_int(self):
        self.assertRaises(TypeError, TabularCPD, 'event', '2', [[0.1, 0.9]])

    def test_cpd_init_cardinality_not_specified(self):
        self.assertRaises(exceptions.CardinalityError, TabularCPD, 'event', 3, [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
                          ['evi1', 'evi2'], [5])
        self.assertRaises(exceptions.CardinalityError, TabularCPD, 'event', 3, [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
                          ['evi1', 'evi2'], 5)
        self.assertRaises(exceptions.CardinalityError, TabularCPD, 'event', 3, [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
                          ['evi1'], [5, 6])
        self.assertRaises(exceptions.CardinalityError, TabularCPD, 'event', 3, [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
                          'evi1', [5, 6])

    def test_cpd_init_value_not_2d(self):
        self.assertRaises(TypeError, TabularCPD, 'event', 3, [[[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                               [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                               [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]]],
                          ['evi1', 'evi2'], [5, 6])


class TestTabularCPDMethods(unittest.TestCase):
    def setUp(self):
        self.cpd = TabularCPD('grade', 3, [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                           [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                           [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
                              evidence=['intel', 'diff'], evidence_card=[3, 2])

    def test_marginalize_1(self):
        self.cpd.marginalize(['diff'])
        self.assertEqual(self.cpd.variable, 'grade')
        self.assertEqual(self.cpd.variable_card, 3)
        self.assertListEqual(list(self.cpd.variables), ['grade', 'intel'])
        np_test.assert_array_equal(self.cpd.cardinality, np.array([3, 3]))
        np_test.assert_array_equal(self.cpd.values.ravel(), np.array([0.1, 0.1, 0.1,
                                                                      0.1, 0.1, 0.1,
                                                                      0.8, 0.8, 0.8]))

    def test_marginalize_2(self):
        self.cpd.marginalize(['grade'])
        self.assertEqual(self.cpd.variable, 'grade')
        self.assertListEqual(list(self.cpd.variables), ['diff', 'intel'])
        np_test.assert_array_equal(self.cpd.cardinality, np.array([2, 3]))
        np_test.assert_array_equal(self.cpd.values.ravel(), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

    def test_normalize(self):
        cpd_un_normalized = TabularCPD('grade', 2, [[0.7, 0.2, 0.6, 0.2], [0.4, 0.4, 0.4, 0.8]],
                                       ['intel', 'diff'], [2, 2])
        cpd_un_normalized.normalize()
        np_test.assert_array_almost_equal(cpd_un_normalized.values.ravel(), np.array([0.63636364, 0.33333333, 0.6, 0.2,
                                                                                      0.36363636, 0.66666667, 0.4, 0.8]))

    def test__repr__(self):
        grade_cpd = TabularCPD('grade', 3, [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                            [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
                               evidence=['intel', 'diff'], evidence_card=[3, 2])
        intel_cpd = TabularCPD('intel', 3, [[0.5], [0.3], [0.2]])
        diff_cpd = TabularCPD('grade', 3, [[0.1, 0.1], [0.1, 0.1],  [0.8, 0.8]], evidence=['diff'], evidence_card=[2])
        self.assertEqual(repr(grade_cpd), '<TabularCPD representing P(grade:3 | intel:3, diff:2) at {address}>'
                         .format(address=hex(id(grade_cpd))))
        self.assertEqual(repr(intel_cpd), '<TabularCPD representing P(intel:3) at {address}>'
                         .format(address=hex(id(intel_cpd))))
        self.assertEqual(repr(diff_cpd), '<TabularCPD representing P(grade:3 | diff:2) at {address}>'
                         .format(address=hex(id(diff_cpd))))

    def test_reduce_1(self):
        self.cpd.reduce([('diff', 0)])
        np_test.assert_array_equal(self.cpd.get_cpd(), np.array([[0.1, 0.1, 0.1],
                                                                 [0.1, 0.1, 0.1],
                                                                 [0.8, 0.8, 0.8]]))

    def test_reduce_2(self):
        self.cpd.reduce([('intel', 0)])
        np_test.assert_array_equal(self.cpd.get_cpd(), np.array([[0.1, 0.1],
                                                                 [0.1, 0.1],
                                                                 [0.8, 0.8]]))

    def test_reduce_3(self):
        self.cpd.reduce([('intel', 0), ('diff', 0)])
        np_test.assert_array_equal(self.cpd.get_cpd(), np.array([[0.1],
                                                                 [0.1],
                                                                 [0.8]]))

    def test_reduce_4(self):
        self.cpd.reduce([('grade', 0)])
        np_test.assert_array_equal(self.cpd.get_cpd(), np.array([[1, 1, 1, 1, 1, 1]]))

    def test_get_cpd(self):
        np_test.assert_array_equal(self.cpd.get_cpd(), np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                 [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                 [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]]))

    def tearDown(self):
        del self.cpd


# class TestJointProbabilityDistributionInit(unittest.TestCase):
#     def test_jpd_init(self):
#         jpd = JPD(['x1', 'x2', 'x3'], [2, 3, 2], np.ones(12) / 12)
#         np_test.assert_array_equal(jpd.cardinality, np.array([2, 3, 2]))
#         np_test.assert_array_equal(jpd.values, np.ones(12) / 12)
#         dic = {'x1': [('x1', 0), ('x1', 1)], 'x2': [('x2', 0), ('x2', 1), ('x2', 2)], 'x3': [('x3', 0), ('x3', 1)]}
#         self.assertEqual(jpd.variables, OrderedDict(sorted(dic.items(), key=lambda t: t[1])))
#
#     def test_jpd_init_exception(self):
#         self.assertRaises(ValueError, JPD, ['x1', 'x2', 'x3'], [2, 2, 2], np.ones(8))
#
#
# class TestJointProbabilityDistributionMethods(unittest.TestCase):
#     def setUp(self):
#         self.jpd = JPD(['x1', 'x2', 'x3'], [2, 3, 2], values=np.ones(12) / 12)
#
#     def test_jpd_marginal_distribution_list(self):
#         self.jpd.marginal_distribution(['x1', 'x2'])
#         np_test.assert_array_almost_equal(self.jpd.values, np.array([0.16666667, 0.16666667, 0.16666667,
#                                                                      0.16666667, 0.16666667, 0.16666667]))
#         np_test.assert_array_equal(self.jpd.cardinality, np.array([2, 3]))
#         dic = {'x1': [('x1', 0), ('x1', 1)], 'x2': [('x2', 0), ('x2', 1), ('x2', 2)]}
#         self.assertEqual(self.jpd.variables, OrderedDict(sorted(dic.items(), key=lambda t: t[1])))
#         np_test.assert_almost_equal(np.sum(self.jpd.values), 1)
#
#     def test_marginal_distribution_str(self):
#         self.jpd.marginal_distribution('x1')
#         np_test.assert_array_almost_equal(self.jpd.values, np.array([0.5, 0.5]))
#         np_test.assert_array_equal(self.jpd.cardinality, np.array([2]))
#         dic = {'x1': [('x1', 0), ('x1', 1)]}
#         self.assertEqual(self.jpd.variables, OrderedDict(sorted(dic.items(), key=lambda t: t[1])))
#         np_test.assert_almost_equal(np.sum(self.jpd.values), 1)
#
#     def test_conditional_distribution_list(self):
#         self.jpd.conditional_distribution([('x1', 1), ('x2', 0)])
#         np_test.assert_array_almost_equal(self.jpd.values, np.array([0.5, 0.5]))
#         np_test.assert_array_equal(self.jpd.cardinality, np.array([2]))
#         dic = {'x3': [('x3', 0), ('x3', 1)]}
#         self.assertEqual(self.jpd.variables, OrderedDict(sorted(dic.items(), key=lambda t: t[1])))
#         np_test.assert_almost_equal(np.sum(self.jpd.values), 1)
#
#     def test_conditional_distribution_str(self):
#         self.jpd.conditional_distribution(('x1', 1))
#         np_test.assert_array_almost_equal(self.jpd.values, np.array([0.16666667, 0.16666667,
#                                                                      0.16666667, 0.16666667,
#                                                                      0.16666667, 0.16666667]))
#         np_test.assert_array_equal(self.jpd.cardinality, np.array([3, 2]))
#         dic = {'x2': [('x2', 0), ('x2', 1), ('x2', 2)], 'x3': [('x3', 0), ('x3', 1)]}
#         self.assertEqual(self.jpd.variables, OrderedDict(sorted(dic.items(), key=lambda t: t[1])))
#         np_test.assert_almost_equal(np.sum(self.jpd.values), 1)
#
#     def tearDown(self):
#         del self.jpd

#
# class TestTreeCPDInit(unittest.TestCase):
#     def test_init_single_variable_nodes(self):
#         tree = TreeCPD([('B', Factor(['A'], [2], [0.8, 0.2]), 0),
#                         ('B', 'C', 1),
#                         ('C', Factor(['A'], [2], [0.1, 0.9]), 0),
#                         ('C', 'D', 1),
#                         ('D', Factor(['A'], [2], [0.9, 0.1]), 0),
#                         ('D', Factor(['A'], [2], [0.4, 0.6]), 1)])
#
#         self.assertTrue('B' in tree.nodes())
#         self.assertTrue('C' in tree.nodes())
#         self.assertTrue('D' in tree.nodes())
#         self.assertTrue(Factor(['A'], [2], [0.8, 0.2]) in tree.nodes())
#         self.assertTrue(Factor(['A'], [2], [0.1, 0.9]) in tree.nodes())
#         self.assertTrue(Factor(['A'], [2], [0.9, 0.1]) in tree.nodes())
#         self.assertTrue(Factor(['A'], [2], [0.4, 0.6]) in tree.nodes())
#
#         self.assertTrue(('B', Factor(['A'], [2], [0.8, 0.2]) in tree.edges()))
#         self.assertTrue(('B', Factor(['A'], [2], [0.1, 0.9]) in tree.edges()))
#         self.assertTrue(('B', Factor(['A'], [2], [0.9, 0.1]) in tree.edges()))
#         self.assertTrue(('B', Factor(['A'], [2], [0.4, 0.6]) in tree.edges()))
#         self.assertTrue(('C', 'D') in tree.edges())
#         self.assertTrue(('B', 'C') in tree.edges())
#
#         self.assertEqual(tree['B'][Factor(['A'], [2], [0.8, 0.2])]['label'], 0)
#         self.assertEqual(tree['B']['C']['label'], 1)
#         self.assertEqual(tree['C'][Factor(['A'], [2], [0.1, 0.9])]['label'], 0)
#         self.assertEqual(tree['C']['D']['label'], 1)
#         self.assertEqual(tree['D'][Factor(['A'], [2], [0.9, 0.1])]['label'], 0)
#         self.assertEqual(tree['D'][Factor(['A'], [2], [0.4, 0.6])]['label'], 1)
#
#         self.assertRaises(ValueError, tree.add_edges_from, [('F', 'G')])
#
#     def test_init_self_loop(self):
#         self.assertRaises(ValueError, TreeCPD, [('B', 'B', 0)])
#
#     def test_init_cycle(self):
#         self.assertRaises(ValueError, TreeCPD, [('A', 'B', 0), ('B', 'C', 1), ('C', 'A', 0)])
#
#     def test_init_multi_variable_nodes(self):
#         tree = TreeCPD([(('B', 'C'), Factor(['A'], [2], [0.8, 0.2]), (0, 0)),
#                         (('B', 'C'), 'D', (0, 1)),
#                         (('B', 'C'), Factor(['A'], [2], [0.1, 0.9]), (1, 0)),
#                         (('B', 'C'), 'E', (1, 1)),
#                         ('D', Factor(['A'], [2], [0.9, 0.1]), 0),
#                         ('D', Factor(['A'], [2], [0.4, 0.6]), 1),
#                         ('E', Factor(['A'], [2], [0.3, 0.7]), 0),
#                         ('E', Factor(['A'], [2], [0.8, 0.2]), 1)
#                         ])
#
#         self.assertTrue(('B', 'C') in tree.nodes())
#         self.assertTrue('D' in tree.nodes())
#         self.assertTrue('E' in tree.nodes())
#         self.assertTrue(Factor(['A'], [2], [0.8, 0.2]) in tree.nodes())
#         self.assertTrue(Factor(['A'], [2], [0.9, 0.1]) in tree.nodes())
#
#         self.assertTrue((('B', 'C'), Factor(['A'], [2], [0.8, 0.2]) in tree.edges()))
#         self.assertTrue((('B', 'C'), 'E') in tree.edges())
#         self.assertTrue(('D', Factor(['A'], [2], [0.4, 0.6])) in tree.edges())
#         self.assertTrue(('E', Factor(['A'], [2], [0.8, 0.2])) in tree.edges())
#
#         self.assertEqual(tree[('B', 'C')][Factor(['A'], [2], [0.8, 0.2])]['label'], (0, 0))
#         self.assertEqual(tree[('B', 'C')]['D']['label'], (0, 1))
#         self.assertEqual(tree['D'][Factor(['A'], [2], [0.9, 0.1])]['label'], 0)
#         self.assertEqual(tree['E'][Factor(['A'], [2], [0.3, 0.7])]['label'], 0)
#
#
# class TestTreeCPD(unittest.TestCase):
#     def setUp(self):
#         self.tree1 = TreeCPD([('B', Factor(['A'], [2], [0.8, 0.2]), '0'),
#                               ('B', 'C', '1'),
#                               ('C', Factor(['A'], [2], [0.1, 0.9]), '0'),
#                               ('C', 'D', '1'),
#                               ('D', Factor(['A'], [2], [0.9, 0.1]), '0'),
#                               ('D', Factor(['A'], [2], [0.4, 0.6]), '1')])
#
#         self.tree2 = TreeCPD([('C','A','0'),('C','B','1'),
#                               ('A', Factor(['J'], [2], [0.9, 0.1]), '0'),
#                               ('A', Factor(['J'], [2], [0.3, 0.7]), '1'),
#                               ('B', Factor(['J'], [2], [0.8, 0.2]), '0'),
#                               ('B', Factor(['J'], [2], [0.4, 0.6]), '1')])
#
#     def test_add_edge(self):
#         self.tree1.add_edge('yolo', 'yo', 0)
#         self.assertTrue('yolo' in self.tree1.nodes() and 'yo' in self.tree1.nodes())
#         self.assertTrue(('yolo', 'yo') in self.tree1.edges())
#         self.assertEqual(self.tree1['yolo']['yo']['label'], 0)
#
#     def test_add_edges_from(self):
#         self.tree1.add_edges_from([('yolo', 'yo', 0), ('hello', 'world', 1)])
#         self.assertTrue('yolo' in self.tree1.nodes() and 'yo' in self.tree1.nodes() and
#                         'hello' in self.tree1.nodes() and 'world' in self.tree1.nodes())
#         self.assertTrue(('yolo', 'yo') in self.tree1.edges())
#         self.assertTrue(('hello', 'world') in self.tree1.edges())
#         self.assertEqual(self.tree1['yolo']['yo']['label'], 0)
#         self.assertEqual(self.tree1['hello']['world']['label'], 1)
#
#     def test_to_tabular_cpd(self):
#         tabular_cpd = self.tree1.to_tabular_cpd()
#         self.assertEqual(tabular_cpd.evidence, ['D', 'C', 'B'])
#         self.assertEqual(tabular_cpd.evidence_card, [2, 2, 2])
#         self.assertEqual(list(tabular_cpd.variables), ['A', 'B', 'C', 'D'])
#         np_test.assert_array_equal(tabular_cpd.values,
#                                    np.array([0.8, 0.8, 0.8, 0.8, 0.1, 0.1, 0.9, 0.4,
#                                              0.2, 0.2, 0.2, 0.2, 0.9, 0.9, 0.1, 0.6]))
#
#         tabular_cpd = self.tree2.to_tabular_cpd()
#         self.assertEqual(tabular_cpd.evidence, ['A', 'B', 'C'])
#         self.assertEqual(tabular_cpd.evidence_card, [2, 2, 2])
#         self.assertEqual(list(tabular_cpd.variables), ['J', 'C', 'B', 'A'])
#         np_test.assert_array_equal(tabular_cpd.values,
#                                   np.array([ 0.9,  0.3,  0.9,  0.3,  0.8,  0.8,  0.4,  0.4,
#                                              0.1,  0.7,  0.1,  0.7,  0.2,  0.2,  0.6,  0.6]))
#
#     @unittest.skip('Not implemented yet')
#     def test_to_tabular_cpd_parent_order(self):
#         tabular_cpd = self.tree1.to_tabular_cpd('A', parents_order=['D', 'C', 'B'])
#         self.assertEqual(tabular_cpd.evidence, ['D', 'C', 'B'])
#         self.assertEqual(tabular_cpd.evidence_card, [2, 2, 2])
#         self.assertEqual(list(tabular_cpd.variables), ['A', 'D', 'C', 'B'])
#         np_test.assert_array_equal(tabular_cpd.values,
#                                    np.array([0.8, 0.1, 0.8, 0.9, 0.8, 0.1, 0.8, 0.4,
#                                              0.2, 0.9, 0.2, 0.1, 0.2, 0.9, 0.2, 0.6]))
#
#         tabular_cpd = self.tree2.to_tabular_cpd('A', parents_order=['E', 'D', 'C', 'B'])
#
#     @unittest.skip('Not implemented yet')
#     def test_to_rule_cpd(self):
#         rule_cpd = self.tree1.to_rule_cpd()
#         self.assertEqual(rule_cpd.cardinality(), {'A': 2, 'B': 2, 'C': 2, 'D': 2})
#         self.assertEqual(rule_cpd.scope(), {'A', 'B', 'C', 'D'})
#         self.assertEqual(rule_cpd.variable, 'A')
#         self.assertEqual(rule_cpd.rules, {('A_0', 'B_0'): 0.8,
#                                           ('A_1', 'B_0'): 0.2,
#                                           ('A_0', 'B_1', 'C_0'): 0.1,
#                                           ('A_0', 'B_1', 'C_1', 'D_0'): 0.9,
#                                           ('A_1', 'B_1', 'C_1', 'D_0'): 0.1,
#                                           ('A_0', 'B_1', 'C_1', 'D_1'): 0.4,
#                                           ('A_1', 'B_!', 'C_1', 'D_1'): 0.6})
#
#         rule_cpd = self.tree2.to_rule_cpd()
#         self.assertEqual(rule_cpd.cardinality(), {'A': 2, 'B': 2, 'C': 2, 'D': 2, 'E': 2})
#         self.assertEqual(rule_cpd.scope(), {'A', 'B', 'C', 'D', 'E'})
#         self.assertEqual(rule_cpd.variable, 'A')
#         self.assertEqual(rule_cpd.rules, {('A_0', 'B_0', 'C_0'): 0.8,
#                                           ('A_1', 'B_0', 'C_0'): 0.2,
#                                           ('A_0', 'B_0', 'C_1', 'D_0'): 0.9,
#                                           ('A_1', 'B_0', 'C_1', 'D_0'): 0.1,
#                                           ('A_0', 'B_0', 'C_1', 'D_1'): 0.4,
#                                           ('A_1', 'B_0', 'C_1', 'D_1'): 0.6,
#                                           ('A_0', 'B_1', 'C_0'): 0.1,
#                                           ('A_1', 'B_1', 'C_0'): 0.9,
#                                           ('A_0', 'B_1', 'C_1', 'E_0'): 0.3,
#                                           ('A_1', 'B_1', 'C_1', 'E_0'): 0.7,
#                                           ('A_0', 'B_1', 'C_1', 'E_1'): 0.8,
#                                           ('A_1', 'B_1', 'C_1', 'E_1'): 0.2})
#
#
# class TestRuleCPDInit(unittest.TestCase):
#     def test_init_without_errors_rules_none(self):
#         rule_cpd = RuleCPD('A')
#         self.assertEqual(rule_cpd.variable, 'A')
#
#     def test_init_without_errors_rules_not_none(self):
#         rule_cpd = RuleCPD('A', {('A_0', 'B_0'): 0.8,
#                                  ('A_1', 'B_0'): 0.2,
#                                  ('A_0', 'B_1', 'C_0'): 0.4,
#                                  ('A_1', 'B_1', 'C_0'): 0.6,
#                                  ('A_0', 'B_1', 'C_1'): 0.9,
#                                  ('A_1', 'B_1', 'C_1'): 0.1})
#         self.assertEqual(rule_cpd.variable, 'A')
#         self.assertEqual(rule_cpd.rules, {('A_0', 'B_0'): 0.8,
#                                           ('A_1', 'B_0'): 0.2,
#                                           ('A_0', 'B_1', 'C_0'): 0.4,
#                                           ('A_1', 'B_1', 'C_0'): 0.6,
#                                           ('A_0', 'B_1', 'C_1'): 0.9,
#                                           ('A_1', 'B_1', 'C_1'): 0.1})
#
#     def test_init_with_errors(self):
#         self.assertRaises(ValueError, RuleCPD, 'A', {('A_0',): 0.5,
#                                                      ('A_0', 'B_0'): 0.8,
#                                                      ('A_1', 'B_0'): 0.2,
#                                                      ('A_0', 'B_1', 'C_0'): 0.4,
#                                                      ('A_1', 'B_1', 'C_0'): 0.6,
#                                                      ('A_0', 'B_1', 'C_1'): 0.9,
#                                                      ('A_1', 'B_1', 'C_1'): 0.1})
#
#
# class TestRuleCPDMethods(unittest.TestCase):
#     def setUp(self):
#         self.rule_cpd_with_rules = RuleCPD('A', {('A_0', 'B_0'): 0.8,
#                                                  ('A_1', 'B_0'): 0.2,
#                                                  ('A_0', 'B_1', 'C_0'): 0.4,
#                                                  ('A_1', 'B_1', 'C_0'): 0.6})
#         self.rule_cpd_without_rules = RuleCPD('A')
#
#     def test_add_rules_single(self):
#         self.rule_cpd_with_rules.add_rules({('A_0', 'B_1', 'C_1'): 0.9})
#         self.assertEqual(self.rule_cpd_with_rules.rules, {('A_0', 'B_0'): 0.8,
#                                                           ('A_1', 'B_0'): 0.2,
#                                                           ('A_0', 'B_1', 'C_0'): 0.4,
#                                                           ('A_1', 'B_1', 'C_0'): 0.6,
#                                                           ('A_0', 'B_1', 'C_1'): 0.9})
#         self.assertEqual(self.rule_cpd_with_rules.variable, 'A')
#         self.rule_cpd_without_rules.add_rules({('A_0', 'B_1', 'C_1'): 0.9})
#         self.assertEqual(self.rule_cpd_without_rules.rules, {('A_0', 'B_1', 'C_1'): 0.9})
#         self.assertEqual(self.rule_cpd_without_rules.variable, 'A')
#
#     def test_add_rules_multiple(self):
#         self.rule_cpd_with_rules.add_rules({('A_0', 'B_1', 'C_1'): 0.9,
#                                             ('A_1', 'B_1', 'C_1'): 0.1})
#         self.assertEqual(self.rule_cpd_with_rules.rules, {('A_0', 'B_0'): 0.8,
#                                                           ('A_1', 'B_0'): 0.2,
#                                                           ('A_0', 'B_1', 'C_0'): 0.4,
#                                                           ('A_1', 'B_1', 'C_0'): 0.6,
#                                                           ('A_0', 'B_1', 'C_1'): 0.9,
#                                                           ('A_1', 'B_1', 'C_1'): 0.1})
#         self.assertEqual(self.rule_cpd_with_rules.variable, 'A')
#         self.rule_cpd_without_rules.add_rules({('A_0', 'B_1', 'C_1'): 0.9,
#                                                ('A_1', 'B_1', 'C_1'): 0.1})
#         self.assertEqual(self.rule_cpd_without_rules.rules, {('A_0', 'B_1', 'C_1'): 0.9,
#                                                              ('A_1', 'B_1', 'C_1'): 0.1})
#         self.assertEqual(self.rule_cpd_without_rules.variable, 'A')
#
#     def test_add_rules_error(self):
#         self.assertRaises(ValueError, self.rule_cpd_with_rules.add_rules, {('A_0',): 0.8})
#
#     def test_scope(self):
#         self.assertEqual(self.rule_cpd_with_rules.scope(), {'A', 'B', 'C'})
#         self.assertEqual(self.rule_cpd_without_rules.scope(), set())
#
#     def test_cardinality(self):
#         self.assertEqual(self.rule_cpd_with_rules.cardinality(), {'A': 2, 'B': 2, 'C': 1})
#         self.assertEqual(self.rule_cpd_without_rules.cardinality(), {})
#
#     def tearDown(self):
#         del self.rule_cpd_without_rules
#
