import unittest
import warnings
from collections import OrderedDict

import numpy as np
import numpy.testing as np_test
from pgmpy.extern.six.moves import range

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.factors.discrete import JointProbabilityDistribution as JPD
from pgmpy.factors import factor_divide
from pgmpy.factors import factor_product
from pgmpy.independencies import Independencies
from pgmpy.models import BayesianModel
from pgmpy.models import MarkovModel


class TestJointProbabilityDistributionInit(unittest.TestCase):

    def test_jpd_init(self):
        jpd = JPD(['x1', 'x2', 'x3'], [2, 3, 2], np.ones(12) / 12)
        np_test.assert_array_equal(jpd.cardinality, np.array([2, 3, 2]))
        np_test.assert_array_equal(jpd.values, np.ones(12).reshape(2, 3, 2) / 12)
        self.assertEqual(jpd.get_cardinality(['x1', 'x2', 'x3']), {'x1': 2, 'x2': 3, 'x3': 2})

    def test_jpd_init_exception(self):
        self.assertRaises(ValueError, JPD, ['x1', 'x2', 'x3'], [2, 2, 2], np.ones(8))


class TestJointProbabilityDistributionMethods(unittest.TestCase):

    def setUp(self):
        self.jpd = JPD(['x1', 'x2', 'x3'], [2, 3, 2], values=np.ones(12) / 12)
        self.jpd1 = JPD(['x1', 'x2', 'x3'], [2, 3, 2], values=np.ones(12) / 12)
        self.jpd2 = JPD(['x1', 'x2', 'x3'], [2, 2, 3],
                        [0.126, 0.168, 0.126, 0.009, 0.045, 0.126, 0.252, 0.0224, 0.0056, 0.06, 0.036, 0.024])
        self.jpd3 = JPD(['x1', 'x2', 'x3'], [2, 2, 2],
                        [5.0e-04, 5.225e-04, 0.00, 8.9775e-03, 9.9e-03, 5.39055e-02, 0.00, 9.261945e-01])

    def test_jpd_marginal_distribution_list(self):
        self.jpd.marginal_distribution(['x1', 'x2'])
        np_test.assert_array_almost_equal(self.jpd.values,
                                          np.array([[0.16666667, 0.16666667, 0.16666667],
                                                    [0.16666667, 0.16666667, 0.16666667]]))
        np_test.assert_array_equal(self.jpd.cardinality, np.array([2, 3]))
        dic = {'x1': 2, 'x2': 3}
        self.assertEqual(self.jpd.get_cardinality(['x1', 'x2']), dic)
        self.assertEqual(self.jpd.scope(), ['x1', 'x2'])
        np_test.assert_almost_equal(np.sum(self.jpd.values), 1)
        new_jpd = self.jpd1.marginal_distribution(['x1', 'x2'], inplace=False)
        self.assertTrue(self.jpd1 != self.jpd)
        self.assertTrue(new_jpd == self.jpd)

    def test_marginal_distribution_str(self):
        self.jpd.marginal_distribution('x1')
        np_test.assert_array_almost_equal(self.jpd.values, np.array([0.5, 0.5]))
        np_test.assert_array_equal(self.jpd.cardinality, np.array([2]))
        self.assertEqual(self.jpd.scope(), ['x1'])
        np_test.assert_almost_equal(np.sum(self.jpd.values), 1)
        new_jpd = self.jpd1.marginal_distribution('x1', inplace=False)
        self.assertTrue(self.jpd1 != self.jpd)
        self.assertTrue(self.jpd == new_jpd)

    def test_conditional_distribution_list(self):
        self.jpd = self.jpd1.copy()
        self.jpd.conditional_distribution([('x1', 1), ('x2', 0)])
        np_test.assert_array_almost_equal(self.jpd.values, np.array([0.5, 0.5]))
        np_test.assert_array_equal(self.jpd.cardinality, np.array([2]))
        self.assertEqual(self.jpd.scope(), ['x3'])
        np_test.assert_almost_equal(np.sum(self.jpd.values), 1)
        new_jpd = self.jpd1.conditional_distribution([('x1', 1), ('x2', 0)], inplace=False)
        self.assertTrue(self.jpd1 != self.jpd)
        self.assertTrue(self.jpd == new_jpd)

    def test_check_independence(self):
        self.assertTrue(self.jpd2.check_independence(['x1'], ['x2']))
        self.assertRaises(TypeError, self.jpd2.check_independence, 'x1', ['x2'])
        self.assertRaises(TypeError, self.jpd2.check_independence, ['x1'], 'x2')
        self.assertRaises(TypeError, self.jpd2.check_independence, ['x1'], ['x2'], 'x3')
        self.assertFalse(self.jpd2.check_independence(['x1'], ['x2'], ('x3',), condition_random_variable=True))
        self.assertFalse(self.jpd2.check_independence(['x1'], ['x2'], [('x3', 0)]))
        self.assertTrue(self.jpd1.check_independence(['x1'], ['x2'], ('x3',), condition_random_variable=True))
        self.assertTrue(self.jpd1.check_independence(['x1'], ['x2'], [('x3', 1)]))
        self.assertTrue(self.jpd3.check_independence(['x1'], ['x2'], ('x3',), condition_random_variable=True))

    def test_get_independencies(self):
        independencies = Independencies(['x1', 'x2'], ['x2', 'x3'], ['x3', 'x1'])
        independencies1 = Independencies(['x1', 'x2'])
        self.assertEqual(self.jpd1.get_independencies(), independencies)
        self.assertEqual(self.jpd2.get_independencies(), independencies1)
        self.assertEqual(self.jpd1.get_independencies([('x3', 0)]), independencies1)
        self.assertEqual(self.jpd2.get_independencies([('x3', 0)]), Independencies())

    def test_minimal_imap(self):
        bm = self.jpd1.minimal_imap(order=['x1', 'x2', 'x3'])
        self.assertEqual(sorted(bm.edges()), sorted([('x1', 'x3'), ('x2', 'x3')]))
        bm = self.jpd1.minimal_imap(order=['x2', 'x3', 'x1'])
        self.assertEqual(sorted(bm.edges()), sorted([('x2', 'x1'), ('x3', 'x1')]))
        bm = self.jpd2.minimal_imap(order=['x1', 'x2', 'x3'])
        self.assertEqual(bm.edges(), [])
        bm = self.jpd2.minimal_imap(order=['x1', 'x2'])
        self.assertEqual(bm.edges(), [])

    def test_repr(self):
        self.assertEqual(repr(self.jpd1), '<Joint Distribution representing P(x1:2, x2:3, x3:2) at {address}>'.format(
            address=hex(id(self.jpd1))))

    def test_is_imap(self):
        G1 = BayesianModel([('diff', 'grade'), ('intel', 'grade')])
        diff_cpd = TabularCPD('diff', 2, [[0.2], [0.8]])
        intel_cpd = TabularCPD('intel', 3, [[0.5], [0.3], [0.2]])
        grade_cpd = TabularCPD('grade', 3,
                               [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
                               evidence=['diff', 'intel'],
                               evidence_card=[2, 3])
        G1.add_cpds(diff_cpd, intel_cpd, grade_cpd)
        val = [0.01, 0.01, 0.08, 0.006, 0.006, 0.048, 0.004, 0.004, 0.032,
               0.04, 0.04, 0.32, 0.024, 0.024, 0.192, 0.016, 0.016, 0.128]
        jpd = JPD(['diff', 'intel', 'grade'], [2, 3, 3], val)
        self.assertTrue(jpd.is_imap(G1))
        self.assertRaises(TypeError, jpd.is_imap, MarkovModel())

    def tearDown(self):
        del self.jpd
        del self.jpd1
        del self.jpd2
        del self.jpd3