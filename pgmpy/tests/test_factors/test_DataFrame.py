import unittest

import numpy as np
import numpy.testing as np_test

from pgmpy.factors import DataFrame


class TestFactorInit(unittest.TestCase):
	def test_class_init(self):
		phi1 = DataFrame(['x1', 'x2', 'x3'], list(range(12)))
		self.assertListEqual(phi1.variables, ['x1', 'x2', 'x3'])
		np_test.assert_array_equal(phi1.values, np.arange(12).reshape(-1, 3))

		phi2 = DataFrame(['x1', 'x2', 5, (1,2)],
						 np.array([[ 0,  1,  2,  3],
   							       [ 4,  5,  6,  7],
						           [ 8,  9, 10, 11],
						           [12, 13, 14, 15],
						           [16, 17, 18, 19]]))
		self.assertListEqual(phi2.variables, ['x1', 'x2', 5, (1,2)])
		np_test.assert_array_equal(phi2.values, np.arange(20).reshape(-1, 4))

	def test_class_init_error(self):
		self.assertRaises(TypeError, DataFrame, 'x1', np.arange(12))
		self.assertRaises(TypeError, DataFrame, 2, np.arange(12))
		self.assertRaises(TypeError, DataFrame, ['x1', 'x2', 'x3'], ['a', 'b', 'c'])
		self.assertRaises(TypeError, DataFrame, ['x1', 'x2', 'x3'], 'A')
		self.assertRaises(ValueError, DataFrame, ['x1', 'x2', 'x3'], np.arange(10))
		self.assertRaises(ValueError, DataFrame, ['x1', 2], np.arange(13))
