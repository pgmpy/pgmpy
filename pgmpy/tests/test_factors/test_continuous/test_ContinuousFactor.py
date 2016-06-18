import unittest

import numpy as np
import numpy.testing as np_test
from scipy.special import beta
from scipy.stats import multivariate_normal

from pgmpy.factors import ContinuousFactor


class TestContinuousFactor(unittest.TestCase):
    def test_class_init(self):
        pdf1 = lambda x, y: (np.power(x, 1)*np.power(y, 2))/beta(x, y)
        phi1 = ContinuousFactor(['x', 'y'], pdf1)
        self.assertEqual(phi1.variables, ['x', 'y'])
        self.assertEqual(phi1.pdf, pdf1)

        pdf2 = lambda *args: multivariate_normal(args, [0, 0], [[1, 0], [0, 1]])
        phi2 = ContinuousFactor(['x1', 'x2'], pdf2)
        self.assertEqual(phi2.variables, ['x1', 'x2'])
        self.assertEqual(phi2.pdf, pdf2)

        pdf3 = lambda x, y, z: z*(np.power(x, 1)*np.power(y, 2))/beta(x, y)
        phi3 = ContinuousFactor(['x', 'y', 'z'], pdf3)
        self.assertEqual(phi3.variables, ['x', 'y', 'z'])
        self.assertEqual(phi3.pdf, pdf3)

    def test_class_init_typeerror(self):
        pdf1 = lambda x, y: (np.power(x, 1)*np.power(y, 2))/beta(x, y)
        self.assertRaises(TypeError, ContinuousFactor, 'x y', pdf1)
        self.assertRaises(TypeError, ContinuousFactor, 'x', pdf1)

        pdf2 = lambda *args: multivariate_normal(args, [0, 0], [[1, 0], [0, 1]])
        self.assertRaises(TypeError, ContinuousFactor, 'x1 x2', pdf2)
        self.assertRaises(TypeError, ContinuousFactor, 'x1', pdf1)

        pdf3 = lambda x, y, z: z*(np.power(x, 1)*np.power(y, 2))/beta(x, y)
        self.assertRaises(TypeError, ContinuousFactor, 'x y z', pdf3)
        self.assertRaises(TypeError, ContinuousFactor, 'x', pdf3)
