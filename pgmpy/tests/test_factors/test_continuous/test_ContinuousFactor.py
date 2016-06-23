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

        pdf2 = lambda *args: multivariate_normal.pdf(args, [0, 0], [[1, 0], [0, 1]])
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

        pdf2 = lambda *args: multivariate_normal.pdf(args, [0, 0], [[1, 0], [0, 1]])
        self.assertRaises(TypeError, ContinuousFactor, 'x1 x2', pdf2)
        self.assertRaises(TypeError, ContinuousFactor, 'x1', pdf1)

        pdf3 = lambda x, y, z: z*(np.power(x, 1)*np.power(y, 2))/beta(x, y)
        self.assertRaises(TypeError, ContinuousFactor, 'x y z', pdf3)
        self.assertRaises(TypeError, ContinuousFactor, 'x', pdf3)


class TestContinuousFactorMethods(unittest.TestCase):
    def setUp(self):
        self.pdf1 = lambda x, y: (np.power(x, 1)*np.power(y, 2))/beta(x, y)
        self.phi1 = ContinuousFactor(['x', 'y'], self.pdf1)

        self.pdf2 = lambda x1, x2: multivariate_normal.pdf([x1, x2], [0, 0], [[1, 0], [0, 1]])
        self.phi2 = ContinuousFactor(['x1', 'x2'], self.pdf2)

        self.pdf3 = lambda x, y, z: z*(np.power(x, 1)*np.power(y, 2))/beta(x, y)
        self.phi3 = ContinuousFactor(['x', 'y', 'z'], self.pdf3)

        self.pdf4 = lambda x1, x2, x3: multivariate_normal.pdf([x1, x2, x3], [0, 0, 0],
                                                               [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.phi4 = ContinuousFactor(['x1', 'x2', 'x3'], self.pdf4)


    def test_scope(self):
        self.assertEqual(self.phi1.scope(), self.phi1.variables)
        self.assertEqual(self.phi2.scope(), self.phi2.variables)
        self.assertEqual(self.phi3.scope(), self.phi3.variables)

    def test_assignment(self):
        self.assertEqual(self.phi1.assignment(1.212, 2), self.pdf1(1.212, 2))
        self.assertEqual(self.phi2.assignment(1, -2.231), self.pdf2(1, -2.231))
        self.assertEqual(self.phi3.assignment(1.212, 2.213, -3), self.pdf3(1.212, 2.213, -3))

    def test_reduce(self):
        phi1 = self.phi1.copy()
        phi1.reduce([('x', 1)])
        reduced_pdf1 = lambda y: (np.power(1, 1)*np.power(y, 2))/beta(1, y)
        self.assertEqual(phi1.variables, ['y'])
        for inp in np.random.rand(4):
            self.assertEqual(phi1.pdf(inp), reduced_pdf1(inp))
            self.assertEqual(phi1.pdf(y=inp), reduced_pdf1(inp))

        phi1 = self.phi1.reduce([('x', 1)], inplace=False)
        self.assertEqual(phi1.variables, ['y'])
        for inp in np.random.rand(4):
            self.assertEqual(phi1.pdf(inp), reduced_pdf1(inp))
            self.assertEqual(phi1.pdf(y=inp), reduced_pdf1(inp))

        phi2 = self.phi2.copy()
        phi2.reduce([('x2', 7.213)])
        reduced_pdf2 = lambda x1: multivariate_normal.pdf([x1, 7.213], [0, 0], [[1, 0], [0, 1]])
        self.assertEqual(phi2.variables, ['x1'])
        for inp in np.random.rand(4):
            self.assertEqual(phi2.pdf(inp), reduced_pdf2(inp))
            self.assertEqual(phi2.pdf(x1=inp), reduced_pdf2(inp))

        phi2 = self.phi2.reduce([('x2', 7.213)], inplace=False)
        self.assertEqual(phi2.variables, ['x1'])
        for inp in np.random.rand(4):
            self.assertEqual(phi2.pdf(inp), reduced_pdf2(inp))
            self.assertEqual(phi2.pdf(x1=inp), reduced_pdf2(inp))

        phi3 = self.phi3.copy()
        phi3.reduce([('y', 0.112), ('z', 23)])
        reduced_pdf4 = lambda x: 23*(np.power(x, 1)*np.power(0.112, 2))/beta(x, 0.112)
        self.assertEqual(phi3.variables, ['x'])
        for inp in np.random.rand(4):
            self.assertEqual(phi3.pdf(inp), reduced_pdf4(inp))
            self.assertEqual(phi3.pdf(x=inp), reduced_pdf4(inp))

        phi3 = self.phi3.copy()
        phi3.reduce([('y', 0.112)])
        reduced_pdf3 = lambda x, z: z*(np.power(x, 1)*np.power(0.112, 2))/beta(x, 0.112)
        self.assertEqual(phi3.variables, ['x', 'z'])
        for inp in np.random.rand(4, 2):
            self.assertEqual(phi3.pdf(inp[0], inp[1]), reduced_pdf3(inp[0], inp[1]))
            self.assertEqual(phi3.pdf(x=inp[0], z=inp[1]), reduced_pdf3(inp[0], inp[1]))

        phi3 = self.phi3.reduce([('y', 0.112)], inplace=False)
        self.assertEqual(phi3.variables, ['x', 'z'])
        for inp in np.random.rand(4, 2):
            self.assertEqual(phi3.pdf(inp[0], inp[1]), reduced_pdf3(inp[0], inp[1]))
            self.assertEqual(phi3.pdf(x=inp[0], z=inp[1]), reduced_pdf3(inp[0], inp[1]))
            self.assertEqual(phi3.pdf(inp[0], z=inp[1]), reduced_pdf3(inp[0], inp[1]))

        phi3 = self.phi3.reduce([('y', 0.112), ('z', 23)], inplace=False)
        self.assertEqual(phi3.variables, ['x'])
        for inp in np.random.rand(4):
            self.assertEqual(phi3.pdf(inp), reduced_pdf4(inp))
            self.assertEqual(phi3.pdf(x=inp), reduced_pdf4(inp))

    def test_marginalize(self):
        phi2 = self.phi2.copy()
        phi2.marginalize(['x2'])
        marginalized_pdf2 = lambda x1: multivariate_normal.pdf([x1], [0], [[1]])
        self.assertEqual(phi2.variables, ['x1'])
        for inp in np.random.rand(4):
            np_test.assert_almost_equal(phi2.pdf(inp), marginalized_pdf2(inp))

        phi2 = self.phi2.marginalize(['x2'], inplace=False)
        self.assertEqual(phi2.variables, ['x1'])
        for inp in np.random.rand(4):
            np_test.assert_almost_equal(phi2.pdf(inp), marginalized_pdf2(inp))

        phi4 = self.phi4.copy()
        phi4.marginalize(['x2'])
        marginalized_pdf4 = lambda x1, x3: multivariate_normal.pdf([x1, x3], [0, 0], [[1, 0], [0, 1]])
        self.assertEqual(phi4.variables, ['x1', 'x3'])
        for inp in np.random.rand(4, 2):
            np_test.assert_almost_equal(phi4.pdf(inp[0], inp[1]), marginalized_pdf4(inp[0], inp[1]))

        phi4.marginalize(['x3'])
        self.assertEqual(phi4.variables, ['x1'])
        for inp in np.random.rand(1):
            np_test.assert_almost_equal(phi4.pdf(inp), marginalized_pdf2(inp))

        phi4 = self.phi4.marginalize(['x2'], inplace=False)
        self.assertEqual(phi4.variables, ['x1', 'x3'])
        for inp in np.random.rand(4, 2):
            np_test.assert_almost_equal(phi4.pdf(inp[0], inp[1]), marginalized_pdf4(inp[0], inp[1]))

        phi4 = phi4.marginalize(['x3'], inplace=False)
        self.assertEqual(phi4.variables, ['x1'])
        for inp in np.random.rand(1):
            np_test.assert_almost_equal(phi4.pdf(inp), marginalized_pdf2(inp))

    def test_normalize(self):
        pdf2 = lambda x1, x2: 2 * self.pdf2(x1, x2)

        phi2 = ContinuousFactor(['x1', 'x2'], pdf2)
        phi4 = phi2.copy()

        phi4.normalize()
        self.assertEqual(phi4.variables, phi2.variables)
        for inp in np.random.rand(1, 2):
            np_test.assert_almost_equal(phi4.pdf(inp[0], inp[1]), self.pdf2(inp[0], inp[1]))

        phi4 = phi2.normalize(inplace=False)
        self.assertEqual(phi4.variables, phi4.variables)
        for inp in np.random.rand(1, 2):
            np_test.assert_almost_equal(phi4.pdf(inp[0], inp[1]), self.pdf2(inp[0], inp[1]))

    def test_operate(self):
        phi1 = self.phi1.copy()
        phi1.operate(self.phi2, 'product')
        self.assertEqual(phi1.variables, ['x', 'y', 'x1', 'x2'])
        for inp in np.random.rand(4, 4):
            self.assertEqual(phi1.pdf(*inp), self.phi1.pdf(inp[0], inp[1]) * self.phi2.pdf(inp[2], inp[3]))

        phi1 = self.phi1.operate(self.phi2, 'product', inplace=False)
        self.assertEqual(phi1.variables, ['x', 'y', 'x1', 'x2'])
        for inp in np.random.rand(4, 4):
            self.assertEqual(phi1.pdf(*inp), self.phi1.pdf(inp[0], inp[1]) * self.phi2.pdf(inp[2], inp[3]))

        phi1 = self.phi1 * self.phi2
        self.assertEqual(phi1.variables, ['x', 'y', 'x1', 'x2'])
        for inp in np.random.rand(4, 4):
            self.assertEqual(phi1.pdf(*inp), self.phi1.pdf(inp[0], inp[1]) * self.phi2.pdf(inp[2], inp[3]))

        phi3 = self.phi3.copy()
        phi3.operate(self.phi1, 'product')
        self.assertEqual(phi3.variables, ['x', 'y', 'z'])
        for inp in np.random.rand(4, 3):
            self.assertEqual(phi3.pdf(*inp), self.phi3.pdf(*inp) * self.phi1.pdf(inp[0], inp[1]))

        phi3 = self.phi3.operate(self.phi1, 'product', inplace=False)
        self.assertEqual(phi3.variables, ['x', 'y', 'z'])
        for inp in np.random.rand(4, 3):
            self.assertEqual(phi3.pdf(*inp), self.phi3.pdf(*inp) * self.phi1.pdf(inp[0], inp[1]))

        phi3 = self.phi3 * self.phi1
        self.assertEqual(phi3.variables, ['x', 'y', 'z'])
        for inp in np.random.rand(4, 3):
            self.assertEqual(phi3.pdf(*inp), self.phi3.pdf(*inp) * self.phi1.pdf(inp[0], inp[1]))

        phi3 = self.phi3.copy()
        phi3.operate(self.phi1, 'divide')
        self.assertEqual(phi3.variables, ['x', 'y', 'z'])
        for inp in np.random.rand(4, 3):
            self.assertEqual(phi3.pdf(*inp), self.phi3.pdf(*inp) / self.phi1.pdf(inp[0], inp[1]))

        phi3 = self.phi3.operate(self.phi1, 'divide', inplace=False)
        self.assertEqual(phi3.variables, ['x', 'y', 'z'])
        for inp in np.random.rand(4, 3):
            self.assertEqual(phi3.pdf(*inp), self.phi3.pdf(*inp) / self.phi1.pdf(inp[0], inp[1]))

        phi3 = self.phi3 / self.phi1
        self.assertEqual(phi3.variables, ['x', 'y', 'z'])
        for inp in np.random.rand(4, 3):
            self.assertEqual(phi3.pdf(*inp), self.phi3.pdf(*inp) / self.phi1.pdf(inp[0], inp[1]))

        phi4 = self.phi4.copy()
        phi4.operate(self.phi2, 'product')
        self.assertEqual(phi4.variables, ['x1', 'x2', 'x3'])
        for inp in np.random.rand(4, 3):
            self.assertEqual(phi4.pdf(*inp), self.phi4.pdf(*inp) * self.phi2.pdf(inp[0], inp[1]))

        phi4 = self.phi4.operate(self.phi2, 'product', inplace=False)
        self.assertEqual(phi4.variables, ['x1', 'x2', 'x3'])
        for inp in np.random.rand(4, 3):
            self.assertEqual(phi4.pdf(*inp), self.phi4.pdf(*inp) * self.phi2.pdf(inp[0], inp[1]))

        phi4 = self.phi4 * self.phi2
        self.assertEqual(phi4.variables, ['x1', 'x2', 'x3'])
        for inp in np.random.rand(4, 3):
            self.assertEqual(phi4.pdf(*inp), self.phi4.pdf(*inp) * self.phi2.pdf(inp[0], inp[1]))

        phi4 = self.phi4.copy()
        phi4.operate(self.phi2, 'divide')
        self.assertEqual(phi4.variables, ['x1', 'x2', 'x3'])
        for inp in np.random.rand(4, 3):
            self.assertEqual(phi4.pdf(*inp), self.phi4.pdf(*inp) / self.phi2.pdf(inp[0], inp[1]))

        phi4 = self.phi4.operate(self.phi2, 'divide', inplace=False)
        self.assertEqual(phi4.variables, ['x1', 'x2', 'x3'])
        for inp in np.random.rand(4, 3):
            self.assertEqual(phi4.pdf(*inp), self.phi4.pdf(*inp) / self.phi2.pdf(inp[0], inp[1]))

        phi4 = self.phi4 / self.phi2
        self.assertEqual(phi4.variables, ['x1', 'x2', 'x3'])
        for inp in np.random.rand(4, 3):
            self.assertEqual(phi4.pdf(*inp), self.phi4.pdf(*inp) / self.phi2.pdf(inp[0], inp[1]))

    def test_operate_error(self):
        self.assertRaises(TypeError, self.phi1.operate, 1, 'product')
        self.assertRaises(TypeError, self.phi1.operate, 1, 'divide')
        self.assertRaises(TypeError, self.phi1.operate, '1', 'product')
        self.assertRaises(TypeError, self.phi1.operate, '1', 'divide')
        self.assertRaises(TypeError, self.phi1.operate, self.phi2.pdf, 'product')
        self.assertRaises(TypeError, self.phi1.operate, self.phi2.pdf, 'divide')
        self.assertRaises(TypeError, self.phi1.operate, [1], 'product')
        self.assertRaises(TypeError, self.phi1.operate, [1], 'divide')

        self.assertRaises(TypeError, self.phi4.operate, 1, 'product')
        self.assertRaises(TypeError, self.phi4.operate, 1, 'divide')
        self.assertRaises(TypeError, self.phi4.operate, '1', 'product')
        self.assertRaises(TypeError, self.phi4.operate, '1', 'divide')
        self.assertRaises(TypeError, self.phi4.operate, self.phi2.pdf, 'product')
        self.assertRaises(TypeError, self.phi4.operate, self.phi2.pdf, 'divide')
        self.assertRaises(TypeError, self.phi4.operate, [1], 'product')
        self.assertRaises(TypeError, self.phi4.operate, [1], 'divide')

        self.assertRaises(TypeError, self.phi1.operate, 1, 'product', False)
        self.assertRaises(TypeError, self.phi1.operate, 1, 'divide', False)
        self.assertRaises(TypeError, self.phi1.operate, '1', 'product', False)
        self.assertRaises(TypeError, self.phi1.operate, '1', 'divide', False)
        self.assertRaises(TypeError, self.phi1.operate, self.phi2.pdf, 'product', False)
        self.assertRaises(TypeError, self.phi1.operate, self.phi2.pdf, 'divide', False)
        self.assertRaises(TypeError, self.phi1.operate, [1], 'product', False)
        self.assertRaises(TypeError, self.phi1.operate, [1], 'divide', False)

        self.assertRaises(TypeError, self.phi4.operate, 1, 'product', False)
        self.assertRaises(TypeError, self.phi4.operate, 1, 'divide', False)
        self.assertRaises(TypeError, self.phi4.operate, '1', 'product', False)
        self.assertRaises(TypeError, self.phi4.operate, '1', 'divide', False)
        self.assertRaises(TypeError, self.phi4.operate, self.phi2.pdf, 'product', False)
        self.assertRaises(TypeError, self.phi4.operate, self.phi2.pdf, 'divide', False)
        self.assertRaises(TypeError, self.phi4.operate, [1], 'product', False)
        self.assertRaises(TypeError, self.phi4.operate, [1], 'divide', False)

        self.assertRaises(ValueError, self.phi1.__truediv__, self.phi2)
        self.assertRaises(ValueError, self.phi1.__truediv__, self.phi3)
        self.assertRaises(ValueError, self.phi1.__truediv__, self.phi4)
        self.assertRaises(ValueError, self.phi2.__truediv__, self.phi3)
        self.assertRaises(ValueError, self.phi2.__truediv__, self.phi4)

    def test_copy(self):
        copy1 = self.phi1.copy()
        copy2 = self.phi3.copy()

        copy4 = copy1.copy()
        copy5 = copy2.copy()

        self.assertEqual(copy1.variables, copy4.variables)
        self.assertEqual(copy1.pdf, copy4.pdf)
        self.assertEqual(copy2.variables, copy5.variables)
        self.assertEqual(copy2.pdf, copy5.pdf)

        copy1.variables = ['A', 'B']
        self.assertEqual(copy4.variables, self.phi1.variables)
        copy1.pdf = lambda a, b: (a + b) / (a * a + b * b)
        copy1_pdf = lambda a, b: (a + b) / (a * a + b * b)
        self.assertEqual(copy4.pdf, self.phi1.pdf)
        copy4.variables = ['X', 'Y']
        self.assertEqual(copy1.variables, ['A', 'B'])
        copy4.pdf = lambda a, b: a + b
        for inp in np.random.rand(4, 2):
            self.assertEqual(copy1.pdf(inp[0], inp[1]), copy1_pdf(inp[0], inp[1]))

        copy2.reduce([('x', 7.7)])
        reduced_pdf = lambda y, z: z*(np.power(7.7, 1)*np.power(y, 2))/beta(7.7, y)
        self.assertEqual(copy5.variables, self.phi3.variables)
        self.assertEqual(copy5.pdf, self.phi3.pdf)
        copy5.reduce([('x', 11), ('z', 13)])
        self.assertEqual(copy2.variables, ['y', 'z'])
        for inp in np.random.rand(4, 2):
            self.assertEqual(copy2.pdf(inp[0], inp[1]), reduced_pdf(inp[0], inp[1]))

    def tearDown(self):
        del self.pdf1
        del self.pdf2
        del self.pdf3
        del self.phi1
        del self.phi2
        del self.phi3
