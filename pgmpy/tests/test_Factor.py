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



    def tearDown(self):
        del self.phi