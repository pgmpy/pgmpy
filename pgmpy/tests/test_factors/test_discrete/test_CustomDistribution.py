import unittest

import numpy as np
import numpy.testing as np_test

from pgmpy.factors.discrete import CustomDistribution

class TestCustomDistributionInit(unittest.TestCase):
    def test_init(self):
        dist_normal = CustomDistribution(variables=['x1', 'x2', 'x3'],
                                         cardinality=[2, 2, 2],
                                         values=np.ones(8))
        self.assertEqual(list(dist_normal.variables), ['x1', 'x2', 'x3'])
        self.assertEqual(list(dist_normal.cardinality), [2, 2, 2])
        np_test.assert_array_equal(dist_normal.values, [[[1, 1], [1, 1]],
                                                        [[1, 1], [1, 1]]])
        self.assertEqual(list(dist_normal.evidence), [])
        self.assertEqual(list(dist_normal.evidence_card), [])

        dist_evi = CustomDistribution(variables=['x1', 'x2', 'x3'],
                                      cardinality=[2, 2, 2],
                                      values=np.ones(32),
                                      evidence=['y1', 'y2'],
                                      evidence_card=[2, 2])
        self.assertEqual(list(dist_evi.variables), ['x1', 'x2', 'x3', 'y1', 'y2'])
        self.assertEqual(list(dist_evi.cardinality), [2, 2, 2, 2, 2])
        np_test.assert_array_equal(dist_evi.values, np.ones(32).reshape(2, 2, 2, 2, 2))
        self.assertEqual(list(dist_evi.evidence), ['y1', 'y2'])
        self.assertEqual(list(dist_evi.evidence_card), [2, 2])

        # Normal Dist Errors
        self.assertRaises(TypeError, CustomDistribution, variables='x1',
                          cardinality=[2], values=[[1]])
        self.assertRaises(ValueError, CustomDistribution, variables=['x1', 'x2'],
                          cardinality=[1], values=[[1]])
        self.assertRaises(ValueError, CustomDistribution, variables=['x1', 'x2'], cardinality=[2, 2],
                          values=np.ones(3))
        self.assertRaises(ValueError, CustomDistribution, variables=['x1', 'x1'], cardinality=[2, 2],
                          values=np.ones(4))

        # Conditional Dist Errors
        self.assertRaises(ValueError, CustomDistribution, variables=['x1'], cardinality=[2],
                          values=np.ones(8), evidence=['y1', 'y1'], evidence_card=[2, 2])
        self.assertRaises(ValueError, CustomDistribution, variables=['x1'], cardinality=[2],
                          values=np.ones(4), evidence=['x1'], evidence_card=[2])
        self.assertRaises(ValueError, CustomDistribution, variables=['x1'], cardinality=[2],
                          values=np.ones(4), evidence=['y1', 'y2'], evidence_card=[2])


class TestCustomDistributionMethods(unittest.TestCase):
    def setUp(self):
        self.dist_normal = CustomDistribution(variables=['x1', 'x2', 'x3'],
                                              cardinality=[2, 3, 2],
                                              values=np.arange(12))
        self.dist_evi = CustomDistribution(variables=['x1', 'x2', 'x3'],
                                           cardinality=[2, 3, 2],
                                           values=np.arange(72),
                                           evidence=['y1', 'y2'],
                                           evidence_card=[2, 3])

    def test_scope(self):
        self.assertEqual(list(self.dist_normal.scope()), ['x1', 'x2', 'x3'])
        self.assertEqual(list(self.dist_evi.scope()), ['x1', 'x2', 'x3', 'y1', 'y2'])

    def test_get_cardinality(self):
        self.assertEqual(self.dist_normal.get_cardinality(variables=['x1', 'x2']),
                         {'x1': 2, 'x2': 3})
        self.assertEqual(self.dist_evi.get_cardinality(variables=['x1', 'y1']),
                         {'x1': 2, 'y1': 2})
        self.assertEqual(self.dist_normal.get_cardinality(variables=None),
                         {'x1': 2, 'x2': 3, 'x3': 2})
        self.assertEqual(self.dist_evi.get_cardinality(variables=None),
                         {'x1': 2, 'x2': 3, 'x3': 2, 'y1': 2, 'y2': 3})

    def test_get_cardinality_err(self):
        self.assertRaises(TypeError, self.dist_normal.get_cardinality, variables='x1')
        self.assertRaises(ValueError, self.dist_normal.get_cardinality, variables=['y1'])
        self.assertRaises(ValueError, self.dist_evi.get_cardinality, variables=['x1', 'z1'])

    def test_assignment(self):
        self.assertRaises(self.dist_normal.assignment([0, 1, 3]),
                          [[('x1', 0), ('x2', 0), ('x3',0)],
                           [('x1', 0), ('x2', 0), ('x3', 1)],
                           [('x1', 0), ('x2', 1), ('x3', 1)]])

        self.assertRaises(self.dist_evi.assignment([0, 2]),
                          [[('x1', 0), ('x2', 0), ('x3', 0), ('y1', 0), ('y2', 0)],
                           [('x1', 0), ('x2', 0), ('x3', 0), ('y1', 0), ('y2', 2)]])

#    def test_marginalize(self):
#        dist_marg = self.dist_normal.marginalize(['x1'], inplace=False)
#        self.assertEqual(sorted(list(dist_marg.variables)), ['x2', 'x3'])
#        self.assertEqual(dist_marg.get_cardinality(variables=None), {'x2': 3, 'x3': 2})
#        np_test.assert_array_equal(dist_marg.values, np.array([[6, 8], [10, 12], [14, 16]])
