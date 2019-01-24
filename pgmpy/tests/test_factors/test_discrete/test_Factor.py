import unittest
import warnings
from collections import OrderedDict

import numpy as np
import numpy.testing as np_test
from pgmpy.extern.six.moves import range

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors import factor_divide
from pgmpy.factors import factor_product
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.models import BayesianModel
from pgmpy.models import MarkovModel


class TestFactorInit(unittest.TestCase):

    def test_class_init(self):
        phi = DiscreteFactor(['x1', 'x2', 'x3'], [2, 2, 2], np.ones(8))
        self.assertEqual(phi.variables, ['x1', 'x2', 'x3'])
        np_test.assert_array_equal(phi.cardinality, np.array([2, 2, 2]))
        np_test.assert_array_equal(phi.values, np.ones(8).reshape(2, 2, 2))

    def test_class_init1(self):
        phi = DiscreteFactor([1, 2, 3], [2, 3, 2], np.arange(12))
        self.assertEqual(phi.variables, [1, 2, 3])
        np_test.assert_array_equal(phi.cardinality, np.array([2, 3, 2]))
        np_test.assert_array_equal(phi.values, np.arange(12).reshape(2, 3, 2))

    def test_class_init_sizeerror(self):
        self.assertRaises(ValueError, DiscreteFactor, ['x1', 'x2', 'x3'], [2, 2, 2], np.ones(9))

    def test_class_init_typeerror(self):
        self.assertRaises(TypeError, DiscreteFactor, 'x1', [3], [1, 2, 3])
        self.assertRaises(ValueError, DiscreteFactor, ['x1', 'x1', 'x3'], [2, 3, 2], range(12))

    def test_init_size_var_card_not_equal(self):
        self.assertRaises(ValueError, DiscreteFactor, ['x1', 'x2'], [2], np.ones(2))


class TestFactorMethods(unittest.TestCase):

    def setUp(self):
        self.phi = DiscreteFactor(['x1', 'x2', 'x3'], [2, 2, 2], np.random.uniform(5, 10, size=8))
        self.phi1 = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        self.phi2 = DiscreteFactor([('x1', 0), ('x2', 0), ('x3', 0)], [2, 3, 2], range(12))
        # This larger factor (phi3) caused a bug in reduce
        card3 = [3, 3, 3, 2, 2, 2, 2, 2, 2]
        self.phi3 = DiscreteFactor(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
                                   card3, np.arange(np.prod(card3), dtype=np.float))

        self.tup1 = ('x1', 'x2')
        self.tup2 = ('x2', 'x3')
        self.tup3 = ('x3', (1, 'x4'))
        self.phi4 = DiscreteFactor([self.tup1, self.tup2, self.tup3], [2, 3, 4], np.random.uniform(3, 10, size=24))
        self.phi5 = DiscreteFactor([self.tup1, self.tup2, self.tup3], [2, 3, 4], range(24))

        self.card6 = [4, 2, 1, 3, 5, 6]
        self.phi6 = DiscreteFactor([self.tup1, self.tup2, self.tup3, self.tup1 + self.tup2,
                                    self.tup2 + self.tup3, self.tup3 + self.tup1], self.card6,
                                   np.arange(np.prod(self.card6), dtype=np.float))

        self.var1 = 'x1'
        self.var2 = ('x2', 1)
        self.var3 = frozenset(['x1', 'x2'])
        self.phi7 = DiscreteFactor([self.var1, self.var2], [3, 2], [3, 2, 4, 5, 9, 8])
        self.phi8 = DiscreteFactor([self.var2, self.var3], [2, 2], [2, 1, 5, 6])
        self.phi9 = DiscreteFactor([self.var1, self.var3], [3, 2], [3, 2, 4, 5, 9, 8])
        self.phi10 = DiscreteFactor([self.var3], [2], [3, 6])

    def test_scope(self):
        self.assertListEqual(self.phi.scope(), ['x1', 'x2', 'x3'])
        self.assertListEqual(self.phi1.scope(), ['x1', 'x2', 'x3'])

        self.assertListEqual(self.phi4.scope(), [self.tup1, self.tup2, self.tup3])

    def test_assignment(self):
        self.assertListEqual(self.phi.assignment([0]), [[('x1', 0), ('x2', 0), ('x3', 0)]])
        self.assertListEqual(self.phi.assignment([4, 5, 6]), [[('x1', 1), ('x2', 0), ('x3', 0)],
                                                              [('x1', 1), ('x2', 0), ('x3', 1)],
                                                              [('x1', 1), ('x2', 1), ('x3', 0)]])

        self.assertListEqual(self.phi1.assignment(np.array([4, 5, 6])), [[('x1', 0), ('x2', 2), ('x3', 0)],
                                                                         [('x1', 0), ('x2', 2), ('x3', 1)],
                                                                         [('x1', 1), ('x2', 0), ('x3', 0)]])
        self.assertListEqual(self.phi4.assignment(np.array([11, 12, 23])),
                             [[(self.tup1, 0), (self.tup2, 2), (self.tup3, 3)],
                              [(self.tup1, 1), (self.tup2, 0), (self.tup3, 0)],
                              [(self.tup1, 1), (self.tup2, 2), (self.tup3, 3)]])

    def test_assignment_indexerror(self):
        self.assertRaises(IndexError, self.phi.assignment, [10])
        self.assertRaises(IndexError, self.phi.assignment, [1, 3, 10, 5])
        self.assertRaises(IndexError, self.phi.assignment, np.array([1, 3, 10, 5]))

        self.assertRaises(IndexError, self.phi4.assignment, [2, 24])
        self.assertRaises(IndexError, self.phi4.assignment, np.array([24, 2, 4, 30]))

    def test_get_cardinality(self):
        self.assertEqual(self.phi.get_cardinality(['x1']), {'x1': 2})
        self.assertEqual(self.phi.get_cardinality(['x2']), {'x2': 2})
        self.assertEqual(self.phi.get_cardinality(['x3']), {'x3': 2})
        self.assertEqual(self.phi.get_cardinality(['x1', 'x2']), {'x1': 2, 'x2': 2})
        self.assertEqual(self.phi.get_cardinality(['x1', 'x3']), {'x1': 2, 'x3': 2})
        self.assertEqual(self.phi.get_cardinality(['x1', 'x2', 'x3']), {'x1': 2, 'x2': 2, 'x3': 2})

        self.assertEqual(self.phi4.get_cardinality([self.tup1, self.tup3]),
                         {self.tup1: 2, self.tup3: 4})

    def test_get_cardinality_scopeerror(self):
        self.assertRaises(ValueError, self.phi.get_cardinality, ['x4'])
        self.assertRaises(ValueError, self.phi4.get_cardinality, [('x1', 'x4')])

        self.assertRaises(ValueError, self.phi4.get_cardinality, [('x3', (2, 'x4'))])

    def test_get_cardinality_typeerror(self):
        self.assertRaises(TypeError, self.phi.get_cardinality, 'x1')

    def test_marginalize(self):
        self.phi1.marginalize(['x1'])
        np_test.assert_array_equal(self.phi1.values, np.array([[6, 8],
                                                               [10, 12],
                                                               [14, 16]]))
        self.phi1.marginalize(['x2'])
        np_test.assert_array_equal(self.phi1.values, np.array([30, 36]))
        self.phi1.marginalize(['x3'])
        np_test.assert_array_equal(self.phi1.values, np.array(66))

        self.phi5.marginalize([self.tup1])
        np_test.assert_array_equal(self.phi5.values, np.array([[12, 14, 16, 18],
                                                               [20, 22, 24, 26],
                                                               [28, 30, 32, 34]]))
        self.phi5.marginalize([self.tup2])
        np_test.assert_array_equal(self.phi5.values, np.array([60, 66, 72, 78]))

        self.phi5.marginalize([self.tup3])
        np_test.assert_array_equal(self.phi5.values, np.array([276]))

    def test_marginalize_scopeerror(self):
        self.assertRaises(ValueError, self.phi.marginalize, ['x4'])
        self.phi.marginalize(['x1'])
        self.assertRaises(ValueError, self.phi.marginalize, ['x1'])

        self.assertRaises(ValueError, self.phi4.marginalize, [('x1', 'x3')])
        self.phi4.marginalize([self.tup2])
        self.assertRaises(ValueError, self.phi4.marginalize, [self.tup2])

    def test_marginalize_typeerror(self):
        self.assertRaises(TypeError, self.phi.marginalize, 'x1')

    def test_marginalize_shape(self):
        values = ['A', 'D', 'F', 'H']
        phi3_mar = self.phi3.marginalize(values, inplace=False)
        # Previously a sorting error caused these to be different
        np_test.assert_array_equal(phi3_mar.values.shape, phi3_mar.cardinality)

        phi6_mar = self.phi6.marginalize([self.tup1, self.tup2], inplace=False)
        np_test.assert_array_equal(phi6_mar.values.shape, phi6_mar.cardinality)

        self.phi6.marginalize([self.tup1, self.tup3 + self.tup1], inplace=True)
        np_test.assert_array_equal(self.phi6.values.shape, self.phi6.cardinality)

    def test_normalize(self):
        self.phi1.normalize()
        np_test.assert_almost_equal(self.phi1.values,
                                    np.array([[[0, 0.01515152],
                                               [0.03030303, 0.04545455],
                                               [0.06060606, 0.07575758]],
                                              [[0.09090909, 0.10606061],
                                               [0.12121212, 0.13636364],
                                               [0.15151515, 0.16666667]]]))
        self.phi5.normalize()
        np_test.assert_almost_equal(self.phi5.values,
                                    [[[0., 0.00362319, 0.00724638, 0.01086957],
                                      [0.01449275, 0.01811594, 0.02173913, 0.02536232],
                                      [0.02898551, 0.0326087,  0.03623188, 0.03985507]],
                                     [[0.04347826, 0.04710145, 0.05072464, 0.05434783],
                                      [0.05797101, 0.0615942,  0.06521739, 0.06884058],
                                      [0.07246377, 0.07608696, 0.07971014, 0.08333333]]])

    def test_reduce(self):
        self.phi1.reduce([('x1', 0), ('x2', 0)])
        np_test.assert_array_equal(self.phi1.values, np.array([0, 1]))

        self.phi5.reduce([(self.tup1, 0), (self.tup3, 1)])
        np_test.assert_array_equal(self.phi5.values, np.array([1, 5, 9]))

    def test_reduce1(self):
        self.phi1.reduce([('x2', 0), ('x1', 0)])
        np_test.assert_array_equal(self.phi1.values, np.array([0, 1]))

        self.phi5.reduce([(self.tup3, 1), (self.tup1, 0)])
        np_test.assert_array_equal(self.phi5.values, np.array([1, 5, 9]))

    def test_reduce_shape(self):
        values = [('A', 0), ('D', 0), ('F', 0), ('H', 1)]
        phi3_reduced = self.phi3.reduce(values, inplace=False)
        # Previously a sorting error caused these to be different
        np_test.assert_array_equal(phi3_reduced.values.shape, phi3_reduced.cardinality)

        values = [(self.tup1, 2), (self.tup3, 0)]
        phi6_reduced = self.phi6.reduce(values, inplace=False)
        np_test.assert_array_equal(phi6_reduced.values.shape, phi6_reduced.cardinality)

        self.phi6.reduce(values, inplace=True)
        np_test.assert_array_equal(self.phi6.values.shape, self.phi6.cardinality)

    def test_complete_reduce(self):
        self.phi1.reduce([('x1', 0), ('x2', 0), ('x3', 1)])
        np_test.assert_array_equal(self.phi1.values, np.array([1]))
        np_test.assert_array_equal(self.phi1.cardinality, np.array([]))
        np_test.assert_array_equal(self.phi1.variables, OrderedDict())

        self.phi5.reduce([(('x1', 'x2'), 1), (('x2', 'x3'), 0), (('x3', (1, 'x4')), 3)])
        np_test.assert_array_equal(self.phi5.values, np.array([15]))
        np_test.assert_array_equal(self.phi5.cardinality, np.array([]))
        np_test.assert_array_equal(self.phi5.variables, OrderedDict())

    def test_reduce_typeerror(self):
        self.assertRaises(TypeError, self.phi1.reduce, 'x10')
        self.assertRaises(TypeError, self.phi1.reduce, ['x10'])
        self.assertRaises(TypeError, self.phi1.reduce, [('x1', 'x2')])
        self.assertRaises(TypeError, self.phi1.reduce, [(0, 'x1')])
        self.assertRaises(TypeError, self.phi1.reduce, [(0.1, 'x1')])
        self.assertRaises(TypeError, self.phi1.reduce, [(0.1, 0.1)])
        self.assertRaises(TypeError, self.phi1.reduce, [('x1', 0.1)])

        self.assertRaises(TypeError, self.phi5.reduce, [(('x1', 'x2'), 0), (('x2', 'x3'), 0.2)])

    def test_reduce_scopeerror(self):
        self.assertRaises(ValueError, self.phi1.reduce, [('x4', 1)])
        self.assertRaises(ValueError, self.phi5.reduce, [((('x1', 0.1), 0))])

    def test_reduce_sizeerror(self):
        self.assertRaises(IndexError, self.phi1.reduce, [('x3', 5)])
        self.assertRaises(IndexError, self.phi5.reduce, [(('x2', 'x3'), 3)])

    def test_identity_factor(self):
        identity_factor = self.phi.identity_factor()
        self.assertEqual(list(identity_factor.variables), ['x1', 'x2', 'x3'])
        np_test.assert_array_equal(identity_factor.cardinality, [2, 2, 2])
        np_test.assert_array_equal(identity_factor.values, np.ones(8).reshape(2, 2, 2))

        identity_factor1 = self.phi5.identity_factor()
        self.assertEqual(list(identity_factor1.variables), [self.tup1, self.tup2, self.tup3])
        np_test.assert_array_equal(identity_factor1.cardinality, [2, 3, 4])
        np_test.assert_array_equal(identity_factor1.values, np.ones(24).reshape(2, 3, 4))

    def test_factor_product(self):
        phi = DiscreteFactor(['x1', 'x2'], [2, 2], range(4))
        phi1 = DiscreteFactor(['x3', 'x4'], [2, 2], range(4))
        prod = factor_product(phi, phi1)
        expected_factor = DiscreteFactor(['x1', 'x2', 'x3', 'x4'], [2, 2, 2, 2],
                                         [0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 4, 6, 0, 3, 6, 9])
        self.assertEqual(prod, expected_factor)
        self.assertEqual(sorted(prod.variables), ['x1', 'x2', 'x3', 'x4'])

        phi = DiscreteFactor(['x1', 'x2'], [3, 2], range(6))
        phi1 = DiscreteFactor(['x2', 'x3'], [2, 2], range(4))
        prod = factor_product(phi, phi1)
        expected_factor = DiscreteFactor(['x1', 'x2', 'x3'], [3, 2, 2],
                                         [0, 0, 2, 3, 0, 2, 6, 9, 0, 4, 10, 15])
        self.assertEqual(prod, expected_factor)
        self.assertEqual(prod.variables, expected_factor.variables)

        prod = factor_product(self.phi7, self.phi8)
        expected_factor = DiscreteFactor([self.var1, self.var2, self.var3], [3, 2, 2],
                                         [6, 3, 10, 12, 8, 4, 25, 30, 18, 9, 40, 48])
        self.assertEqual(prod, expected_factor)
        self.assertEqual(prod.variables, expected_factor.variables)

    def test_sum(self):
        phi = DiscreteFactor(['x1', 'x2'], [2, 2], range(4))
        phi1 = DiscreteFactor(['x3', 'x4'], [2, 2], range(4))
        summation = phi.sum(phi1, inplace=False)
        expected_factor = DiscreteFactor(['x1', 'x2', 'x4', 'x3'], [2, 2, 2, 2],
                                         [0, 2, 1, 3, 1, 3, 2, 4, 2, 4, 3, 5, 3, 5, 4, 6])
        self.assertEqual(summation, expected_factor)
        self.assertEqual(sorted(summation.variables), ['x1', 'x2', 'x3', 'x4'])

        phi = DiscreteFactor(['x1', 'x2'], [3, 2], range(6))
        phi1 = DiscreteFactor(['x2', 'x3'], [2, 2], range(4))
        summation = phi.sum(phi1, inplace=False)
        expected_factor = DiscreteFactor(['x1', 'x2', 'x3'], [3, 2, 2],
                                         [0, 1, 3, 4, 2, 3, 5, 6, 4, 5, 7, 8])
        self.assertEqual(summation, expected_factor)
        self.assertEqual(sorted(summation.variables), ['x1', 'x2', 'x3'])

        phi7_copy = self.phi7
        phi7_copy.sum(self.phi8, inplace=True)
        expected_factor = DiscreteFactor([self.var1, self.var2, self.var3], [3, 2, 2],
                                         [5, 4, 7, 8, 6, 5, 10, 11, 11, 10, 13, 14])
        self.assertEqual(expected_factor, phi7_copy)
        self.assertEqual(phi7_copy.variables, [self.var1, self.var2, self.var3])

    def test_factor_add(self):
        phi = DiscreteFactor(['x1', 'x2'], [2, 2], range(4))
        phi1 = DiscreteFactor(['x3', 'x4'], [2, 2], range(4))
        summation = phi + phi1
        phi3 = DiscreteFactor(['x1','x2','x4','x3'], [2, 2, 2, 2],
                              [0, 2, 1, 3, 1, 3, 2, 4, 2, 4, 3, 5, 3, 5, 4, 6])
        self.assertEqual(phi3, summation)
        self.assertEqual(sorted(summation.variables), ['x1', 'x2', 'x3', 'x4'])

        self.phi9 = self.phi9 + self.phi10
        expected_factor = DiscreteFactor([self.var1, self.var3], [3, 2],
                                         [6, 8, 7, 11, 12, 14])
        self.assertEqual(self.phi9, expected_factor)
        self.assertEqual(self.phi9.variables, [self.var1, self.var3])

    def test_product(self):
        phi = DiscreteFactor(['x1', 'x2'], [2, 2], range(4))
        phi1 = DiscreteFactor(['x3', 'x4'], [2, 2], range(4))
        prod = phi.product(phi1, inplace=False)
        expected_factor = DiscreteFactor(['x1', 'x2', 'x3', 'x4'], [2, 2, 2, 2],
                                         [0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 4, 6, 0, 3, 6, 9])
        self.assertEqual(prod, expected_factor)
        self.assertEqual(sorted(prod.variables), ['x1', 'x2', 'x3', 'x4'])

        phi = DiscreteFactor(['x1', 'x2'], [3, 2], range(6))
        phi1 = DiscreteFactor(['x2', 'x3'], [2, 2], range(4))
        prod = phi.product(phi1, inplace=False)
        expected_factor = DiscreteFactor(['x1', 'x2', 'x3'], [3, 2, 2],
                                         [0, 0, 2, 3, 0, 2, 6, 9, 0, 4, 10, 15])
        self.assertEqual(prod, expected_factor)
        self.assertEqual(sorted(prod.variables), ['x1', 'x2', 'x3'])

        phi7_copy = self.phi7
        phi7_copy.product(self.phi8, inplace=True)
        expected_factor = DiscreteFactor([self.var1, self.var2, self.var3], [3, 2, 2],
                                         [6, 3, 10, 12, 8, 4, 25, 30, 18, 9, 40, 48])
        self.assertEqual(expected_factor, phi7_copy)
        self.assertEqual(phi7_copy.variables, [self.var1, self.var2, self.var3])

    def test_factor_product_non_factor_arg(self):
        self.assertRaises(TypeError, factor_product, 1, 2)

    def test_factor_mul(self):
        phi = DiscreteFactor(['x1', 'x2'], [2, 2], range(4))
        phi1 = DiscreteFactor(['x3', 'x4'], [2, 2], range(4))
        prod = phi * phi1

        sorted_vars = ['x1', 'x2', 'x3', 'x4']
        for axis in range(prod.values.ndim):
            exchange_index = prod.variables.index(sorted_vars[axis])
            prod.variables[axis], prod.variables[exchange_index] = prod.variables[exchange_index], prod.variables[axis]
            prod.values = prod.values.swapaxes(axis, exchange_index)

        np_test.assert_almost_equal(prod.values.ravel(),
                                    np.array([0, 0, 0, 0, 0, 1, 2, 3,
                                              0, 2, 4, 6, 0, 3, 6, 9]))

        self.assertEqual(prod.variables, ['x1', 'x2', 'x3', 'x4'])

    def test_factor_divide(self):
        phi1 = DiscreteFactor(['x1', 'x2'], [2, 2], [1, 2, 2, 4])
        phi2 = DiscreteFactor(['x1'], [2], [1, 2])
        expected_factor = phi1.divide(phi2, inplace=False)
        phi3 = DiscreteFactor(['x1', 'x2'], [2, 2], [1, 2, 1, 2])
        self.assertEqual(phi3, expected_factor)

        self.phi9.divide(self.phi10, inplace=True)
        np_test.assert_array_almost_equal(self.phi9.values, np.array([1.000000, 0.333333, 1.333333,
                                                                      0.833333, 3.000000, 1.333333]).reshape(3, 2))
        self.assertEqual(self.phi9.variables, [self.var1, self.var3])

    def test_factor_divide_truediv(self):
        phi1 = DiscreteFactor(['x1', 'x2'], [2, 2], [1, 2, 2, 4])
        phi2 = DiscreteFactor(['x1'], [2], [1, 2])
        div = phi1 / phi2
        phi3 = DiscreteFactor(['x1', 'x2'], [2, 2], [1, 2, 1, 2])
        self.assertEqual(phi3, div)

        self.phi9 = self.phi9 / self.phi10
        np_test.assert_array_almost_equal(self.phi9.values, np.array([1.000000, 0.333333, 1.333333,
                                                                      0.833333, 3.000000, 1.333333]).reshape(3, 2))
        self.assertEqual(self.phi9.variables, [self.var1, self.var3])

    def test_factor_divide_invalid(self):
        phi1 = DiscreteFactor(['x1', 'x2'], [2, 2], [1, 2, 3, 4])
        phi2 = DiscreteFactor(['x1'], [2], [0, 2])
        div = phi1.divide(phi2, inplace=False)
        np_test.assert_array_equal(div.values.ravel(), np.array([np.inf, np.inf, 1.5, 2]))

    def test_factor_divide_no_common_scope(self):
        phi1 = DiscreteFactor(['x1', 'x2'], [2, 2], [1, 2, 3, 4])
        phi2 = DiscreteFactor(['x3'], [2], [0, 2])
        self.assertRaises(ValueError, factor_divide, phi1, phi2)

        phi2 = DiscreteFactor([self.var3], [2], [2, 1])
        self.assertRaises(ValueError, factor_divide, self.phi7, phi2)

    def test_factor_divide_non_factor_arg(self):
        self.assertRaises(TypeError, factor_divide, 1, 1)

    def test_eq(self):
        self.assertFalse(self.phi == self.phi1)
        self.assertTrue(self.phi == self.phi)
        self.assertTrue(self.phi1 == self.phi1)

        self.assertTrue(self.phi5 == self.phi5)
        self.assertFalse(self.phi5 == self.phi6)
        self.assertTrue(self.phi6 == self.phi6)

    def test_eq1(self):
        phi1 = DiscreteFactor(['x1', 'x2', 'x3'], [2, 4, 3], range(24))
        phi2 = DiscreteFactor(['x2', 'x1', 'x3'], [4, 2, 3],
                              [0, 1, 2, 12, 13, 14, 3, 4, 5, 15, 16, 17, 6, 7,
                               8, 18, 19, 20, 9, 10, 11, 21, 22, 23])
        self.assertTrue(phi1 == phi2)
        self.assertEqual(phi2.variables, ['x2', 'x1', 'x3'])

        phi3 = DiscreteFactor([self.tup1, self.tup2, self.tup3], [2, 4, 3], range(24))
        phi4 = DiscreteFactor([self.tup2, self.tup1, self.tup3], [4, 2, 3],
                              [0, 1, 2, 12, 13, 14, 3, 4, 5, 15, 16, 17,
                               6, 7, 8, 18, 19, 20, 9, 10, 11, 21, 22, 23])
        self.assertTrue(phi3 == phi4)

    def test__repr__(self):
        phi = DiscreteFactor(['x1', 'x2'], [2, 2], [1, 2, 3, 4])
        self.assertEqual(repr(phi), "<DiscreteFactor representing phi(x1:2, x2:2) at {address}>"
                         .format(address=hex(id(phi))))

        self.assertEqual(repr(self.phi7), "<DiscreteFactor representing phi(x1:3, ('x2', 1):2) at {address}>"
                         .format(address=hex(id(self.phi7))))

    def test_hash(self):
        phi1 = DiscreteFactor(['x1', 'x2'], [2, 2], [1, 2, 3, 4])
        phi2 = DiscreteFactor(['x2', 'x1'], [2, 2], [1, 3, 2, 4])
        self.assertEqual(hash(phi1), hash(phi2))

        phi1 = DiscreteFactor(['x1', 'x2', 'x3'], [2, 2, 2], range(8))
        phi2 = DiscreteFactor(['x3', 'x1', 'x2'], [2, 2, 2], [0, 2, 4, 6, 1, 3, 5, 7])
        self.assertEqual(hash(phi1), hash(phi2))

        var1 = TestHash(1, 2)
        phi3 = DiscreteFactor([var1, self.var2, self.var3], [2, 4, 3], range(24))
        phi4 = DiscreteFactor([self.var2, var1, self.var3], [4, 2, 3],
                              [0, 1, 2, 12, 13, 14, 3, 4, 5, 15, 16, 17,
                               6, 7, 8, 18, 19, 20, 9, 10, 11, 21, 22, 23])
        self.assertEqual(hash(phi3), hash(phi4))

        var1 = TestHash(2, 3)
        var2 = TestHash('x2', 1)
        phi3 = DiscreteFactor([var1, var2, self.var3], [2, 2, 2], range(8))
        phi4 = DiscreteFactor([self.var3, var1, var2], [2, 2, 2], [0, 2, 4, 6, 1, 3, 5, 7])
        self.assertEqual(hash(phi3), hash(phi4))

    def test_maximize_single(self):
        self.phi1.maximize(['x1'])
        self.assertEqual(self.phi1, DiscreteFactor(['x2', 'x3'], [3, 2], [6, 7, 8, 9, 10, 11]))
        self.phi1.maximize(['x2'])
        self.assertEqual(self.phi1, DiscreteFactor(['x3'], [2], [10, 11]))
        self.phi2 = DiscreteFactor(['x1', 'x2', 'x3'], [3, 2, 2], [0.25, 0.35, 0.08, 0.16, 0.05, 0.07,
                                                                   0.00, 0.00, 0.15, 0.21, 0.08, 0.18])
        self.phi2.maximize(['x2'])
        self.assertEqual(self.phi2, DiscreteFactor(['x1', 'x3'], [3, 2], [0.25, 0.35, 0.05,
                                                                          0.07, 0.15, 0.21]))

        self.phi5.maximize([('x1', 'x2')])
        self.assertEqual(self.phi5, DiscreteFactor([('x2', 'x3'), ('x3', (1, 'x4'))], [3, 4],
                                                   [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]))
        self.phi5.maximize([('x2', 'x3')])
        self.assertEqual(self.phi5, DiscreteFactor([('x3', (1, 'x4'))], [4], [20, 21, 22, 23]))

    def test_maximize_list(self):
        self.phi1.maximize(['x1', 'x2'])
        self.assertEqual(self.phi1, DiscreteFactor(['x3'], [2], [10, 11]))

        self.phi5.maximize([('x1', 'x2'), ('x2', 'x3')])
        self.assertEqual(self.phi5, DiscreteFactor([('x3', (1, 'x4'))], [4], [20, 21, 22, 23]))

    def test_maximize_shape(self):
        values = ['A', 'D', 'F', 'H']
        phi3_max = self.phi3.maximize(values, inplace=False)
        # Previously a sorting error caused these to be different
        np_test.assert_array_equal(phi3_max.values.shape, phi3_max.cardinality)

        phi = DiscreteFactor([self.var1, self.var2, self.var3], [3, 2, 2], [3, 2, 4, 5, 9, 8, 3, 2, 4, 5, 9, 8])
        phi_max = phi.marginalize([self.var1, self.var2], inplace=False)
        np_test.assert_array_equal(phi_max.values.shape, phi_max.cardinality)

    def test_maximize_scopeerror(self):
        self.assertRaises(ValueError, self.phi.maximize, ['x10'])

    def test_maximize_typeerror(self):
        self.assertRaises(TypeError, self.phi.maximize, 'x1')

    def test_copy(self):
        phi_copy = self.phi.copy()
        self.assertListEqual(self.phi.variables, phi_copy.variables)
        np_test.assert_array_equal(self.phi.cardinality, phi_copy.cardinality)
        np_test.assert_array_equal(self.phi.values, phi_copy.values)

    def tearDown(self):
        del self.phi
        del self.phi1
        del self.phi2
        del self.phi3
        del self.phi4
        del self.phi5
        del self.phi6
        del self.phi7
        del self.phi8
        del self.phi9
        del self.phi10


class TestHash:
    # Used to check the hash function of DiscreteFactor class.

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __hash__(self):
        return hash(str(self.x) + str(self.y))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.x == other.x and self.y == other.y






#
# class TestTreeCPDInit(unittest.TestCase):
#     def test_init_single_variable_nodes(self):
#         tree = TreeCPD([('B', DiscreteFactor(['A'], [2], [0.8, 0.2]), 0),
#                         ('B', 'C', 1),
#                         ('C', DiscreteFactor(['A'], [2], [0.1, 0.9]), 0),
#                         ('C', 'D', 1),
#                         ('D', DiscreteFactor(['A'], [2], [0.9, 0.1]), 0),
#                         ('D', DiscreteFactor(['A'], [2], [0.4, 0.6]), 1)])
#
#         self.assertTrue('B' in tree.nodes())
#         self.assertTrue('C' in tree.nodes())
#         self.assertTrue('D' in tree.nodes())
#         self.assertTrue(DiscreteFactor(['A'], [2], [0.8, 0.2]) in tree.nodes())
#         self.assertTrue(DiscreteFactor(['A'], [2], [0.1, 0.9]) in tree.nodes())
#         self.assertTrue(DiscreteFactor(['A'], [2], [0.9, 0.1]) in tree.nodes())
#         self.assertTrue(DiscreteFactor(['A'], [2], [0.4, 0.6]) in tree.nodes())
#
#         self.assertTrue(('B', DiscreteFactor(['A'], [2], [0.8, 0.2]) in tree.edges()))
#         self.assertTrue(('B', DiscreteFactor(['A'], [2], [0.1, 0.9]) in tree.edges()))
#         self.assertTrue(('B', DiscreteFactor(['A'], [2], [0.9, 0.1]) in tree.edges()))
#         self.assertTrue(('B', DiscreteFactor(['A'], [2], [0.4, 0.6]) in tree.edges()))
#         self.assertTrue(('C', 'D') in tree.edges())
#         self.assertTrue(('B', 'C') in tree.edges())
#
#         self.assertEqual(tree['B'][DiscreteFactor(['A'], [2], [0.8, 0.2])]['label'], 0)
#         self.assertEqual(tree['B']['C']['label'], 1)
#         self.assertEqual(tree['C'][DiscreteFactor(['A'], [2], [0.1, 0.9])]['label'], 0)
#         self.assertEqual(tree['C']['D']['label'], 1)
#         self.assertEqual(tree['D'][DiscreteFactor(['A'], [2], [0.9, 0.1])]['label'], 0)
#         self.assertEqual(tree['D'][DiscreteFactor(['A'], [2], [0.4, 0.6])]['label'], 1)
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
#         tree = TreeCPD([(('B', 'C'), DiscreteFactor(['A'], [2], [0.8, 0.2]), (0, 0)),
#                         (('B', 'C'), 'D', (0, 1)),
#                         (('B', 'C'), DiscreteFactor(['A'], [2], [0.1, 0.9]), (1, 0)),
#                         (('B', 'C'), 'E', (1, 1)),
#                         ('D', DiscreteFactor(['A'], [2], [0.9, 0.1]), 0),
#                         ('D', DiscreteFactor(['A'], [2], [0.4, 0.6]), 1),
#                         ('E', DiscreteFactor(['A'], [2], [0.3, 0.7]), 0),
#                         ('E', DiscreteFactor(['A'], [2], [0.8, 0.2]), 1)
#                         ])
#
#         self.assertTrue(('B', 'C') in tree.nodes())
#         self.assertTrue('D' in tree.nodes())
#         self.assertTrue('E' in tree.nodes())
#         self.assertTrue(DiscreteFactor(['A'], [2], [0.8, 0.2]) in tree.nodes())
#         self.assertTrue(DiscreteFactor(['A'], [2], [0.9, 0.1]) in tree.nodes())
#
#         self.assertTrue((('B', 'C'), DiscreteFactor(['A'], [2], [0.8, 0.2]) in tree.edges()))
#         self.assertTrue((('B', 'C'), 'E') in tree.edges())
#         self.assertTrue(('D', DiscreteFactor(['A'], [2], [0.4, 0.6])) in tree.edges())
#         self.assertTrue(('E', DiscreteFactor(['A'], [2], [0.8, 0.2])) in tree.edges())
#
#         self.assertEqual(tree[('B', 'C')][DiscreteFactor(['A'], [2], [0.8, 0.2])]['label'], (0, 0))
#         self.assertEqual(tree[('B', 'C')]['D']['label'], (0, 1))
#         self.assertEqual(tree['D'][DiscreteFactor(['A'], [2], [0.9, 0.1])]['label'], 0)
#         self.assertEqual(tree['E'][DiscreteFactor(['A'], [2], [0.3, 0.7])]['label'], 0)
#
#
# class TestTreeCPD(unittest.TestCase):
#     def setUp(self):
#         self.tree1 = TreeCPD([('B', DiscreteFactor(['A'], [2], [0.8, 0.2]), '0'),
#                               ('B', 'C', '1'),
#                               ('C', DiscreteFactor(['A'], [2], [0.1, 0.9]), '0'),
#                               ('C', 'D', '1'),
#                               ('D', DiscreteFactor(['A'], [2], [0.9, 0.1]), '0'),
#                               ('D', DiscreteFactor(['A'], [2], [0.4, 0.6]), '1')])
#
#         self.tree2 = TreeCPD([('C','A','0'),('C','B','1'),
#                               ('A', DiscreteFactor(['J'], [2], [0.9, 0.1]), '0'),
#                               ('A', DiscreteFactor(['J'], [2], [0.3, 0.7]), '1'),
#                               ('B', DiscreteFactor(['J'], [2], [0.8, 0.2]), '0'),
#                               ('B', DiscreteFactor(['J'], [2], [0.4, 0.6]), '1')])
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
