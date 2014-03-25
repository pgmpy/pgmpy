import unittest
from pgmpy.Factor import Factor
from pgmpy.tests import help_functions as hf
from collections import OrderedDict
import numpy.testing as np_test
import numpy as np
from pgmpy import Exceptions


class TestFactorInit(unittest.TestCase):

    def test_class_init(self):
        phi = Factor(['x1', 'x2', 'x3'], [2, 2, 2], np.ones(8))
        dic = {'x1': ['x1_0', 'x1_1'], 'x2': ['x2_0', 'x2_1'], 'x3': ['x3_0', 'x3_1']}
        hf.assertOrderedDictEqual(phi.variables, OrderedDict(sorted(dic.items(), key=lambda t: t[1])))
        np_test.assert_array_equal(phi.cardinality, np.array([2, 2, 2]))
        np_test.assert_array_equal(phi.values, np.ones(8))

    def test_class_init_sizeerror(self):
        self.assertRaises(Exceptions.SizeError, Factor, ['x1', 'x2', 'x3'], [2, 2, 2], np.ones(9))


class TestFactorMethods(unittest.TestCase):

    def setUp(self):
        self.phi = Factor(['x1', 'x2', 'x3'], [2, 2, 2], np.random.uniform(5, 10, size=8))
        self.phi1 = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))

    def test_assignment(self):
        self.assertListEqual(self.phi.assignment([0]), [['x1_0', 'x2_0', 'x3_0']])
        self.assertListEqual(self.phi.assignment([4, 5, 6]), [['x1_0', 'x2_0', 'x3_1'],
                                                              ['x1_1', 'x2_0', 'x3_1'],
                                                              ['x1_0', 'x2_1', 'x3_1']])
        self.assertListEqual(self.phi.assignment(np.array([4, 5, 6])), [['x1_0', 'x2_0', 'x3_1'],
                                                              ['x1_1', 'x2_0', 'x3_1'],
                                                              ['x1_0', 'x2_1', 'x3_1']])

    def test_assignment_indexerror(self):
        self.assertRaises(IndexError, self.phi.assignment, [10])
        self.assertRaises(IndexError, self.phi.assignment, [1, 3, 10, 5])
        self.assertRaises(IndexError, self.phi.assignment, np.array([1, 3, 10, 5]))

    def test_get_cardinality(self):
        self.assertEqual(self.phi.get_cardinality('x1'), 2)
        self.assertEqual(self.phi.get_cardinality('x2'), 2)
        self.assertEqual(self.phi.get_cardinality('x3'), 2)

    def test_get_cardinality_scopeerror(self):
        self.assertRaises(Exceptions.ScopeError, self.phi.get_cardinality, 'x4')

    def test_marginalize(self):
        self.phi1.marginalize('x1')
        np_test.assert_array_equal(self.phi1.values, np.array([1, 5, 9, 13, 17, 21]))
        self.phi1.marginalize(['x2'])
        np_test.assert_array_equal(self.phi1.values, np.array([15, 51]))
        self.phi1.marginalize('x3')
        np_test.assert_array_equal(self.phi1.values, np.array([66]))

    def test_marginalize_scopeerror(self):
        self.assertRaises(Exceptions.ScopeError, self.phi.marginalize, 'x4')
        self.assertRaises(Exceptions.ScopeError, self.phi.marginalize, ['x4'])
        self.phi.marginalize('x1')
        self.assertRaises(Exceptions.ScopeError, self.phi.marginalize, 'x1')

    def test_normalize(self):
        self.phi1.normalize()
        np_test.assert_almost_equal(self.phi1.values, np.array(
            [0, 0.01515152, 0.03030303, 0.04545455, 0.06060606,
             0.07575758, 0.09090909, 0.10606061, 0.12121212,
             0.13636364, 0.15151515, 0.16666667]))

    def test_reduce(self):
        self.phi1.reduce(['x1_0', 'x2_0'])
        np_test.assert_array_equal(self.phi1.values, np.array([0, 6]))

    def test_reduce_typeerror(self):
        self.assertRaises(TypeError, self.phi1.reduce, 'x10')
        self.assertRaises(TypeError, self.phi1.reduce, ['x10'])

    def test_reduce_scopeerror(self):
        self.assertRaises(Exceptions.ScopeError, self.phi1.reduce, 'x4_1')

    def test_reduce_sizeerror(self):
        self.assertRaises(Exceptions.SizeError, self.phi1.reduce, 'x3_5')

    def factor_product(self):
        from pgmpy import Factor
        phi = Factor.Factor(['x1', 'x2'], [2, 2], range(4))
        phi1 = Factor.Factor(['x3', 'x4'], [2, 2], range(4))
        factor_product = Factor.factor_product(phi, phi1)
        np_test.assert_array_equal(factor_product.values,
                                   np.array([0, 0, 0, 0, 0, 1,
                                             2, 3, 0, 2, 4, 6,
                                             0, 3, 6, 9]))
        self.assertEqual(factor_product.variables, OrderedDict([
            ('x1', ['x1_0', 'x1_1']),
            ('x2', ['x2_0', 'x2_1']),
            ('x3', ['x3_0', 'x3_1']),
            ('x4', ['x4_0', 'x4_1'])]
        ))

        phi = Factor.Factor(['x1', 'x2'], [3, 2], range(6))
        phi1 = Factor.Factor(['x2', 'x3'], [2, 2], range(4))
        factor_product = Factor.factor_product(phi, phi1)
        np_test.assert_array_equal(factor_product.values,
                                   np.array([0, 1, 0, 3, 0, 5, 0, 3, 4, 9, 8, 15]))
        self.assertEqual(factor_product.variables, OrderedDict(
            [('x1', ['x1_0', 'x1_1', 'x1_2']),
             ('x2', ['x2_0', 'x2_1']),
             ('x3', ['x3_0', 'x3_1'])]))

    def tearDown(self):
        del self.phi
        del self.phi1