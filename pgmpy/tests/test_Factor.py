import unittest
from pgmpy.Factor import Factor
from pgmpy.Factor.CPD import TabularCPD
import help_functions as hf
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

    def test_factor_product(self):
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

    def test_factor_product2(self):
        from pgmpy import Factor
        phi = Factor.Factor(['x1', 'x2'], [2, 2], range(4))
        phi1 = Factor.Factor(['x3', 'x4'], [2, 2], range(4))
        factor_product = phi.product(phi1)
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
        factor_product = phi.product(phi1)
        np_test.assert_array_equal(factor_product.values,
                                   np.array([0, 1, 0, 3, 0, 5, 0, 3, 4, 9, 8, 15]))
        self.assertEqual(factor_product.variables, OrderedDict(
            [('x1', ['x1_0', 'x1_1', 'x1_2']),
             ('x2', ['x2_0', 'x2_1']),
             ('x3', ['x3_0', 'x3_1'])]))

    def tearDown(self):
        del self.phi
        del self.phi1


class TestTabularCPDInit(unittest.TestCase):

    def test_cpd_init(self):
        cpd = TabularCPD('grade', 3,  [[0.1, 0.1, 0.1]])
        self.assertEqual(cpd.event, 'grade')
        self.assertEqual(cpd.event_card, 3)
        np_test.assert_array_equal(cpd.cpd, np.array([[0.1, 0.1, 0.1]]))
        self.assertEqual(cpd.evidence, None)
        self.assertEqual(cpd.evidence_card, None)

        cpd = TabularCPD('grade', 3, [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                      [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
                         evidence=['intel', 'diff'], evidence_card=[3, 2])
        self.assertListEqual(cpd.evidence_card, [3, 2])
        self.assertListEqual(cpd.evidence, ['intel', 'diff'])
        np_test.assert_array_equal(cpd.cpd, np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                      [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]]))

        cpd = TabularCPD('grade', 3, [[0.1, 0.1],
                                      [0.1, 0.1],
                                      [0.8, 0.8]],
                         evidence='evi1', evidence_card=2)
        self.assertListEqual(cpd.evidence_card, [2])
        self.assertListEqual(cpd.evidence, ['evi1'])
        np_test.assert_array_equal(cpd.cpd, np.array([[0.1, 0.1],
                                                      [0.1, 0.1],
                                                      [0.8, 0.8]]))

    def test_cpd_init_event_not_string(self):
        self.assertRaises(TypeError, TabularCPD, 1, 2, [[0.1, 0.1]])
        self.assertRaises(TypeError, TabularCPD, 1, 'event', "something undefined for this cardinality")

    def test_cpd_init_event_card_not_int(self):
        self.assertRaises(TypeError, TabularCPD, 'event', '2', "something undefined as cardinality is a string")

    def test_cpd_init_cardinality_not_specified(self):
        self.assertRaises(Exceptions.CardinalityError, TabularCPD, 'event', 3, [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
                          ['evi1', 'evi2'], [5])
        self.assertRaises(Exceptions.CardinalityError, TabularCPD, 'event', 3, [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
                          ['evi1', 'evi2'], 5)
        self.assertRaises(Exceptions.CardinalityError, TabularCPD, 'event', 3, [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
                          ['evi1'], [5, 6])
        self.assertRaises(Exceptions.CardinalityError, TabularCPD, 'event', 3, [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
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
        self.cpd.marginalize('diff')
        self.assertListEqual(self.cpd.evidence, ['intel'])
        self.assertListEqual(self.cpd.evidence_card, [3])
        np_test.assert_array_equal(self.cpd.cpd, np.array([[0.2, 0.2, 0.2],
                                                           [0.2, 0.2, 0.2],
                                                           [1.6, 1.6, 1.6]]))

    def test_marginalize_2(self):
        self.cpd.marginalize('grade')
        self.assertListEqual(self.cpd.evidence, ['intel', 'diff'])
        self.assertListEqual(self.cpd.evidence_card, [3, 2])
        np_test.assert_array_equal(self.cpd.cpd, np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]))

    def test_normalize(self):
        cpd_un_normalized = TabularCPD('grade', 2, [[0.7, 0.2, 0.6, 0.2], [0.4, 0.4, 0.4, 0.8]],
                                       ['intel', 'diff'], [2, 2])
        cpd_un_normalized.normalize()
        np_test.assert_array_almost_equal(cpd_un_normalized.cpd, np.array([[0.63636364, 0.33333333, 0.6, 0.2],
                                                                           [0.36363636, 0.66666667, 0.4, 0.8]]))

    def test_reduce_1(self):
        self.cpd.reduce('diff_0')
        np_test.assert_array_equal(self.cpd.cpd, np.array([[0.1, 0.1, 0.1],
                                                           [0.1, 0.1, 0.1],
                                                           [0.8, 0.8, 0.8]]))

    def test_reduce_2(self):
        self.cpd.reduce('intel_0')
        np_test.assert_array_equal(self.cpd.cpd, np.array([[0.1, 0.1],
                                                           [0.1, 0.1],
                                                           [0.8, 0.8]]))

    def test_reduce_3(self):
        self.cpd.reduce(['intel_0', 'diff_0'])
        np_test.assert_array_equal(self.cpd.cpd, np.array([[0.1],
                                                           [0.1],
                                                           [0.8]]))

    def test_reduce_4(self):
        self.cpd.reduce('grade_0')
        np_test.assert_array_equal(self.cpd.cpd, np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]))

    def test_get_cpd(self):
        np_test.assert_array_equal(self.cpd.get_cpd(), np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                 [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                 [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]]))

    def tearDown(self):
        del self.cpd
