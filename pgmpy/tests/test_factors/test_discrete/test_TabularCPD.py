import unittest
import warnings
from collections import OrderedDict

import numpy as np
import numpy.testing as np_test
from pgmpy.extern.six.moves import range

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.models import BayesianModel
from pgmpy.models import MarkovModel


class TestTabularCPDInit(unittest.TestCase):

    def test_cpd_init(self):
        cpd = TabularCPD('grade', 3, [[0.1, 0.1, 0.1]])
        self.assertEqual(cpd.variable, 'grade')
        self.assertEqual(cpd.variable_card, 3)
        self.assertEqual(list(cpd.variables), ['grade'])
        np_test.assert_array_equal(cpd.cardinality, np.array([3]))
        np_test.assert_array_almost_equal(cpd.values, np.array([0.1, 0.1, 0.1]))

        values = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                  [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                  [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]]

        evidence = ['intel', 'diff']
        evidence_card = [3, 2]

        valid_value_inputs = [values, np.asarray(values)]
        valid_evidence_inputs = [evidence, set(evidence), np.asarray(evidence)]
        valid_evidence_card_inputs = [evidence_card, np.asarray(evidence_card)]

        for value in valid_value_inputs:
            for evidence in valid_evidence_inputs:
                for evidence_card in valid_evidence_card_inputs:
                    cpd = TabularCPD('grade', 3, values, evidence=['intel', 'diff'], evidence_card=[3, 2])
                    self.assertEqual(cpd.variable, 'grade')
                    self.assertEqual(cpd.variable_card, 3)
                    np_test.assert_array_equal(cpd.cardinality, np.array([3, 3, 2]))
                    self.assertListEqual(list(cpd.variables), ['grade', 'intel', 'diff'])
                    np_test.assert_array_equal(cpd.values, np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                                                     0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                                                     0.8, 0.8, 0.8, 0.8, 0.8, 0.8]).reshape(3, 3, 2))

        cpd = TabularCPD('grade', 3, [[0.1, 0.1],
                                      [0.1, 0.1],
                                      [0.8, 0.8]],
                         evidence=['evi1'], evidence_card=[2.0])
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
        self.assertRaises(ValueError, TabularCPD, 'event', 3, [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                               [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                               [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
                          ['evi1', 'evi2'], [5])
        self.assertRaises(ValueError, TabularCPD, 'event', 3, [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                               [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                               [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
                          ['evi1', 'evi2'], [5.0])
        self.assertRaises(ValueError, TabularCPD, 'event', 3, [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                               [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                               [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
                          ['evi1'], [5, 6])
        self.assertRaises(TypeError, TabularCPD, 'event', 3, [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
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

        self.cpd2 = TabularCPD('J', 2, [[0.9, 0.3, 0.9, 0.3, 0.8, 0.8, 0.4, 0.4],
                                        [0.1, 0.7, 0.1, 0.7, 0.2, 0.2, 0.6, 0.6]],
                               evidence=['A', 'B', 'C'], evidence_card=[2, 2, 2])

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
        self.assertRaises(ValueError, self.cpd.marginalize, ['grade'])

    def test_marginalize_3(self):
        copy_cpd = self.cpd.copy()
        copy_cpd.marginalize(['intel', 'diff'])
        self.cpd.marginalize(['intel'])
        self.cpd.marginalize(['diff'])
        np_test.assert_array_almost_equal(self.cpd.values, copy_cpd.values)

    def test_normalize(self):
        cpd_un_normalized = TabularCPD('grade', 2, [[0.7, 0.2, 0.6, 0.2], [0.4, 0.4, 0.4, 0.8]],
                                       ['intel', 'diff'], [2, 2])
        cpd_un_normalized.normalize()
        np_test.assert_array_almost_equal(cpd_un_normalized.values, np.array([[[0.63636364, 0.33333333],
                                                                               [0.6, 0.2]],
                                                                              [[0.36363636, 0.66666667],
                                                                               [0.4, 0.8]]]))

    def test_normalize_not_in_place(self):
        cpd_un_normalized = TabularCPD('grade', 2, [[0.7, 0.2, 0.6, 0.2], [0.4, 0.4, 0.4, 0.8]],
                                       ['intel', 'diff'], [2, 2])
        np_test.assert_array_almost_equal(cpd_un_normalized.normalize(inplace=False).values,
                                          np.array([[[0.63636364, 0.33333333],
                                                     [0.6, 0.2]],
                                                    [[0.36363636, 0.66666667],
                                                     [0.4, 0.8]]]))

    def test_normalize_original_safe(self):
        cpd_un_normalized = TabularCPD('grade', 2, [[0.7, 0.2, 0.6, 0.2], [0.4, 0.4, 0.4, 0.8]],
                                       ['intel', 'diff'], [2, 2])
        cpd_un_normalized.normalize(inplace=False)
        np_test.assert_array_almost_equal(cpd_un_normalized.values, np.array([[[0.7, 0.2], [0.6, 0.2]],
                                                                              [[0.4, 0.4], [0.4, 0.8]]]))

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

    def test_copy(self):
        copy_cpd = self.cpd.copy()
        np_test.assert_array_equal(self.cpd.get_values(), copy_cpd.get_values())

    def test_copy_original_safe(self):
        copy_cpd = self.cpd.copy()
        copy_cpd.reorder_parents(['diff', 'intel'])
        np_test.assert_array_equal(self.cpd.get_values(),
                                   np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                             [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]]))

    def test_reduce_1(self):
        self.cpd.reduce([('diff', 0)])
        np_test.assert_array_equal(self.cpd.get_values(), np.array([[0.1, 0.1, 0.1],
                                                                    [0.1, 0.1, 0.1],
                                                                    [0.8, 0.8, 0.8]]))

    def test_reduce_2(self):
        self.cpd.reduce([('intel', 0)])
        np_test.assert_array_equal(self.cpd.get_values(), np.array([[0.1, 0.1],
                                                                    [0.1, 0.1],
                                                                    [0.8, 0.8]]))

    def test_reduce_3(self):
        self.cpd.reduce([('intel', 0), ('diff', 0)])
        np_test.assert_array_equal(self.cpd.get_values(), np.array([[0.1],
                                                                    [0.1],
                                                                    [0.8]]))

    def test_reduce_4(self):
        self.assertRaises(ValueError, self.cpd.reduce, [('grade', 0)])

    def test_reduce_5(self):
        copy_cpd = self.cpd.copy()
        copy_cpd.reduce([('intel', 2), ('diff', 1)])
        self.cpd.reduce([('intel', 2)])
        self.cpd.reduce([('diff', 1)])
        np_test.assert_array_almost_equal(self.cpd.values, copy_cpd.values)

    def test_to_factor(self):
        cpd = TabularCPD('grade', 3, [[0.1, 0.1],
                                      [0.1, 0.1],
                                      [0.8, 0.8]],evidence=['evi1'], evidence_card=[2])
        factor = cpd.to_factor()
        expected_factor = DiscreteFactor(['grade','evi1'], [3, 2], [[0.1, 0.1],
                                                                    [0.1, 0.1],
                                                                    [0.8, 0.8]])
        self.assertEqual(factor, expected_factor)

    def test_get_values(self):
        np_test.assert_array_equal(self.cpd.get_values(),
                                   np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                             [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]]))

    def test_reorder_parents_inplace(self):
        new_vals = self.cpd2.reorder_parents(['B', 'A', 'C'])
        np_test.assert_array_equal(new_vals, np.array([[0.9, 0.3, 0.8, 0.8, 0.9, 0.3, 0.4, 0.4],
                                                       [0.1, 0.7, 0.2, 0.2, 0.1, 0.7, 0.6, 0.6]]))
        np_test.assert_array_equal(self.cpd2.get_values(),
                                   np.array([[0.9, 0.3, 0.8, 0.8, 0.9, 0.3, 0.4, 0.4],
                                             [0.1, 0.7, 0.2, 0.2, 0.1, 0.7, 0.6, 0.6]]))

    def test_reorder_parents(self):
        new_vals = self.cpd2.reorder_parents(['B', 'A', 'C'])
        np_test.assert_array_equal(new_vals, np.array([[0.9, 0.3, 0.8, 0.8, 0.9, 0.3, 0.4, 0.4],
                                                       [0.1, 0.7, 0.2, 0.2, 0.1, 0.7, 0.6, 0.6]]))

    def test_reorder_parents_no_effect(self):
        self.cpd2.reorder_parents(['C', 'A', 'B'], inplace=False)
        np_test.assert_array_equal(self.cpd2.get_values(),
                                   np.array([[0.9, 0.3, 0.9, 0.3, 0.8, 0.8, 0.4, 0.4],
                                             [0.1, 0.7, 0.1, 0.7, 0.2, 0.2, 0.6, 0.6]]))

    def test_reorder_parents_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.cpd2.reorder_parents(['A', 'B', 'C'], inplace=False)
            assert("Same ordering provided as current" in str(w[-1].message))
            np_test.assert_array_equal(self.cpd2.get_values(),
                                       np.array([[0.9, 0.3, 0.9, 0.3, 0.8, 0.8, 0.4, 0.4],
                                                 [0.1, 0.7, 0.1, 0.7, 0.2, 0.2, 0.6, 0.6]]))

    def test_get_evidence(self):
        cpd = TabularCPD('grade', 3, [[0.1, 0.1],
                             [0.1, 0.1],
                             [0.8, 0.8]],evidence=['evil'], evidence_card=[2])
        cpd_evidence = cpd.get_evidence()
        self.assertListEqual(cpd_evidence, ['evil'])

    def tearDown(self):
        del self.cpd
