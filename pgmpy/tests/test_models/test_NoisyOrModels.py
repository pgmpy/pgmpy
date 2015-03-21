import unittest
import numpy as np
import numpy.testing as np_test
from pgmpy.models import NoisyOrModel


class TestNoisyOrModelInit(unittest.TestCase):

    def test_init(self):
        model = NoisyOrModel(['x1', 'x2', 'x3'], [2, 3, 2], [[0.6, 0.4],
                                                             [0.2, 0.4, 0.7],
                                                             [0.1, 0.4]])
        np_test.assert_array_equal(model.variables, np.array(['x1', 'x2', 'x3']))
        np_test.assert_array_equal(model.cardinality, np.array([2, 3, 2]))
        self.assertListEqual(model.inhibitor_probability, [[0.6, 0.4],
                                                           [0.2, 0.4, 0.7],
                                                           [0.1, 0.4]])

    def test_exceptions(self):
        self.assertRaises(ValueError, NoisyOrModel, np.array(['x1', 'x2', 'x3']), [2, 2, 2], [[0.1, 0.2],
                                                                                              [1.0, 0.3],
                                                                                              [1.2, 0.1]])
        self.assertRaises(ValueError, NoisyOrModel, np.array(['x1', 'x2', 'x3']), [2, 4], [[0.1, 0.2],
                                                                                           [0.1, 0.4, 0.2, 0.6]])
        self.assertRaises(ValueError, NoisyOrModel, np.array(['x1', 'x2', 'x3']), [2, 3, 2, 3], [[0.1, 0.2],
                                                                                                 [0.6, 0.3, 0.5],
                                                                                                 [0.3, 0.2],
                                                                                                 [0.1, 0.4, 0.3]])
        self.assertRaises(ValueError, NoisyOrModel, np.array(['x1', 'x2', 'x3']), [2, 3, 2], [[0.1, 0.2, 0.4],
                                                                                              [0.4, 0.1, 0.5],
                                                                                              [0.6, 0.1, 0.7]])
        self.assertRaises(ValueError, NoisyOrModel, np.array(['x1', 'x2', 'x3']), [2, 2, 2], [[0.1, 0.1],
                                                                                              [0.1, 0.1]])


class TestNoisyOrModelMethods(unittest.TestCase):

    def setUp(self):
        self.model = NoisyOrModel(['x1', 'x2', 'x3'], [2, 3, 2], [[0.6, 0.4],
                                                                  [0.2, 0.4, 0.7],
                                                                  [0.1, 0.4]])

    def test_add_variables(self):
        self.model.add_variables(['x4'], [3], [0.1, 0.2, 0.4])
        np_test.assert_array_equal(self.model.variables, np.array(['x1', 'x2', 'x3', 'x4']))
        np_test.assert_array_equal(self.model.cardinality, np.array([2, 3, 2, 3]))
        self.assertListEqual(self.model.inhibitor_probability, [[0.6, 0.4],
                                                                [0.2, 0.4, 0.7],
                                                                [0.1, 0.4],
                                                                [0.1, 0.2, 0.4]])

        self.model.add_variables(['x5', 'x6'], [3, 2], [[0.1, 0.2, 0.4], [0.5, 0.5]])
        np_test.assert_array_equal(self.model.variables, np.array(['x1', 'x2', 'x3', 'x4', 'x5', 'x6']))
        np_test.assert_array_equal(self.model.cardinality, np.array([2, 3, 2, 3, 3, 2]))
        self.assertListEqual(self.model.inhibitor_probability, [[0.6, 0.4],
                                                                [0.2, 0.4, 0.7],
                                                                [0.1, 0.4],
                                                                [0.1, 0.2, 0.4],
                                                                [0.1, 0.2, 0.4],
                                                                [0.5, 0.5]])

    def test_del_variables(self):
        self.model.del_variables(['x3'])
        np_test.assert_array_equal(self.model.variables, np.array(['x1', 'x2']))
        np_test.assert_array_equal(self.model.cardinality, np.array([2, 3]))
        self.assertListEqual(self.model.inhibitor_probability, [[0.6, 0.4],
                                                                [0.2, 0.4, 0.7]])

    def test_del_multiple_variables(self):
        self.model.del_variables(['x1', 'x2'])
        np_test.assert_array_equal(self.model.variables, np.array(['x3']))
        np_test.assert_array_equal(self.model.cardinality, np.array([2]))
        self.assertListEqual(self.model.inhibitor_probability, [[0.1, 0.4]])
