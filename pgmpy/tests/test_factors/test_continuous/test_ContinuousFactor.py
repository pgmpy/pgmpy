import unittest

import numpy as np
import numpy.testing as np_test
from scipy.special import beta
from scipy.stats import multivariate_normal

from pgmpy.factors.continuous import ContinuousFactor


class TestContinuousFactor(unittest.TestCase):
    def pdf1(self, x, y):
        return np.power(x, 1) * np.power(y, 2) / beta(x, y)

    def pdf2(self, *args):
        return multivariate_normal.pdf(args, [0, 0], [[1, 0], [0, 1]])

    def pdf3(self, x, y, z):
        return z * (np.power(x, 1) * np.power(y, 2)) / beta(x, y)

    def test_class_init(self):
        phi1 = ContinuousFactor(["x", "y"], self.pdf1)
        self.assertEqual(phi1.scope(), ["x", "y"])
        self.assertEqual(phi1.pdf, self.pdf1)

        phi2 = ContinuousFactor(["x1", "x2"], self.pdf2)
        self.assertEqual(phi2.scope(), ["x1", "x2"])
        self.assertEqual(phi2.pdf, self.pdf2)

        phi3 = ContinuousFactor(["x", "y", "z"], self.pdf3)
        self.assertEqual(phi3.scope(), ["x", "y", "z"])
        self.assertEqual(phi3.pdf, self.pdf3)

    def test_class_init_typeerror(self):
        self.assertRaises(TypeError, ContinuousFactor, "x y", self.pdf1)
        self.assertRaises(TypeError, ContinuousFactor, "x", self.pdf1)

        self.assertRaises(TypeError, ContinuousFactor, "x1 x2", self.pdf2)
        self.assertRaises(TypeError, ContinuousFactor, "x1", self.pdf1)

        self.assertRaises(TypeError, ContinuousFactor, "x y z", self.pdf3)
        self.assertRaises(TypeError, ContinuousFactor, "x", self.pdf3)

        self.assertRaises(TypeError, ContinuousFactor, set(["x", "y"]), self.pdf1)
        self.assertRaises(TypeError, ContinuousFactor, {"x": 1, "y": 2}, self.pdf1)

        self.assertRaises(TypeError, ContinuousFactor, set(["x1", "x2"]), self.pdf2)
        self.assertRaises(TypeError, ContinuousFactor, {"x1": 1, "x2": 2}, self.pdf1)

        self.assertRaises(TypeError, ContinuousFactor, set(["x", "y", "z"]), self.pdf3)
        self.assertRaises(
            TypeError, ContinuousFactor, {"x": 1, "y": 2, "z": 3}, self.pdf3
        )

    def test_class_init_valueerror(self):
        self.assertRaises(ValueError, ContinuousFactor, ["x", "x"], self.pdf1)
        self.assertRaises(ValueError, ContinuousFactor, ["x", "y", "y"], self.pdf1)

        self.assertRaises(ValueError, ContinuousFactor, ["x1", "x1"], self.pdf2)
        self.assertRaises(ValueError, ContinuousFactor, ["x1", "x2", "x2"], self.pdf2)

        self.assertRaises(ValueError, ContinuousFactor, ["x", "x"], self.pdf1)
        self.assertRaises(
            ValueError, ContinuousFactor, ["x", "y", "y", "z", "z"], self.pdf1
        )


class TestContinuousFactorMethods(unittest.TestCase):
    def pdf1(self, x, y):
        return np.power(x, 1) * np.power(y, 2) / beta(x, y)

    def pdf2(self, x1, x2):
        return multivariate_normal.pdf([x1, x2], [0, 0], [[1, 0], [0, 1]])

    def pdf3(self, x, y, z):
        return z * (np.power(x, 1) * np.power(y, 2)) / beta(x, y)

    def pdf4(self, x1, x2, x3):
        return multivariate_normal.pdf(
            [x1, x2, x3], [0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )

    def setUp(self):
        self.phi1 = ContinuousFactor(["x", "y"], self.pdf1)
        self.phi2 = ContinuousFactor(["x1", "x2"], self.pdf2)
        self.phi3 = ContinuousFactor(["x", "y", "z"], self.pdf3)
        self.phi4 = ContinuousFactor(["x1", "x2", "x3"], self.pdf4)

    def test_scope(self):
        self.assertEqual(self.phi1.scope(), self.phi1.scope())
        self.assertEqual(self.phi2.scope(), self.phi2.scope())
        self.assertEqual(self.phi3.scope(), self.phi3.scope())

    def test_assignment(self):
        self.assertEqual(self.phi1.assignment(1.212, 2), self.pdf1(1.212, 2))
        self.assertEqual(self.phi2.assignment(1, -2.231), self.pdf2(1, -2.231))
        self.assertEqual(
            self.phi3.assignment(1.212, 2.213, -3), self.pdf3(1.212, 2.213, -3)
        )

    def test_reduce(self):
        phi1 = self.phi1.copy()
        phi1.reduce([("x", 1)])

        def reduced_pdf1(y):
            return (np.power(1, 1) * np.power(y, 2)) / beta(1, y)

        self.assertEqual(phi1.scope(), ["y"])
        for inp in np.random.rand(4):
            self.assertEqual(phi1.pdf(inp), reduced_pdf1(inp))
            self.assertEqual(phi1.pdf(y=inp), reduced_pdf1(inp))

        phi1 = self.phi1.reduce([("x", 1)], inplace=False)
        self.assertEqual(phi1.scope(), ["y"])
        for inp in np.random.rand(4):
            self.assertEqual(phi1.pdf(inp), reduced_pdf1(inp))
            self.assertEqual(phi1.pdf(y=inp), reduced_pdf1(inp))

        phi2 = self.phi2.copy()
        phi2.reduce([("x2", 7.213)])

        def reduced_pdf2(x1):
            return multivariate_normal.pdf([x1, 7.213], [0, 0], [[1, 0], [0, 1]])

        self.assertEqual(phi2.scope(), ["x1"])
        for inp in np.random.rand(4):
            self.assertEqual(phi2.pdf(inp), reduced_pdf2(inp))
            self.assertEqual(phi2.pdf(x1=inp), reduced_pdf2(inp))

        phi2 = self.phi2.reduce([("x2", 7.213)], inplace=False)
        self.assertEqual(phi2.scope(), ["x1"])
        for inp in np.random.rand(4):
            self.assertEqual(phi2.pdf(inp), reduced_pdf2(inp))
            self.assertEqual(phi2.pdf(x1=inp), reduced_pdf2(inp))

        phi3 = self.phi3.copy()
        phi3.reduce([("y", 0.112), ("z", 23)])

        def reduced_pdf4(x):
            return 23 * (np.power(x, 1) * np.power(0.112, 2)) / beta(x, 0.112)

        self.assertEqual(phi3.scope(), ["x"])
        for inp in np.random.rand(4):
            self.assertEqual(phi3.pdf(inp), reduced_pdf4(inp))
            self.assertEqual(phi3.pdf(x=inp), reduced_pdf4(inp))

        phi3 = self.phi3.copy()
        phi3.reduce([("y", 0.112)])

        def reduced_pdf3(x, z):
            return z * (np.power(x, 1) * np.power(0.112, 2)) / beta(x, 0.112)

        self.assertEqual(phi3.scope(), ["x", "z"])
        for inp in np.random.rand(4, 2):
            self.assertEqual(phi3.pdf(inp[0], inp[1]), reduced_pdf3(inp[0], inp[1]))
            self.assertEqual(phi3.pdf(x=inp[0], z=inp[1]), reduced_pdf3(inp[0], inp[1]))

        phi3 = self.phi3.reduce([("y", 0.112)], inplace=False)
        self.assertEqual(phi3.scope(), ["x", "z"])
        for inp in np.random.rand(4, 2):
            self.assertEqual(phi3.pdf(inp[0], inp[1]), reduced_pdf3(inp[0], inp[1]))
            self.assertEqual(phi3.pdf(x=inp[0], z=inp[1]), reduced_pdf3(inp[0], inp[1]))
            self.assertEqual(phi3.pdf(inp[0], z=inp[1]), reduced_pdf3(inp[0], inp[1]))

        phi3 = self.phi3.reduce([("y", 0.112), ("z", 23)], inplace=False)
        self.assertEqual(phi3.scope(), ["x"])
        for inp in np.random.rand(4):
            self.assertEqual(phi3.pdf(inp), reduced_pdf4(inp))
            self.assertEqual(phi3.pdf(x=inp), reduced_pdf4(inp))

    def test_reduce_error(self):
        self.assertRaises(TypeError, self.phi1.reduce, "x1")
        self.assertRaises(TypeError, self.phi1.reduce, set(["x", "y"]))
        self.assertRaises(TypeError, self.phi1.reduce, {"x": 1, "y": 1})

        self.assertRaises(TypeError, self.phi4.reduce, "x4")
        self.assertRaises(TypeError, self.phi4.reduce, set(["x1", "x2", "x3"]))
        self.assertRaises(TypeError, self.phi4.reduce, {"x1": 1, "x2": 1, "x3": 1})

        self.assertRaises(ValueError, self.phi1.reduce, [("z", 3)])
        self.assertRaises(ValueError, self.phi1.reduce, [("x", 0), ("y", 1), ("z", 4)])

        self.assertRaises(ValueError, self.phi4.reduce, [("x4", 7)])
        self.assertRaises(
            ValueError, self.phi4.reduce, [("x1", 1), ("x2", 2), ("x3", 3), ("x4", 4)]
        )

    def test_marginalize(self):
        phi2 = self.phi2.copy()
        phi2.marginalize(["x2"])
        self.assertEqual(phi2.scope(), ["x1"])
        for inp in np.random.rand(4):
            np_test.assert_almost_equal(
                phi2.pdf(inp), multivariate_normal.pdf([inp], [0], [[1]])
            )

        phi2 = self.phi2.marginalize(["x2"], inplace=False)
        self.assertEqual(phi2.scope(), ["x1"])
        for inp in np.random.rand(4):
            np_test.assert_almost_equal(
                phi2.pdf(inp), multivariate_normal.pdf([inp], [0], [[1]])
            )

        phi4 = self.phi4.copy()
        phi4.marginalize(["x2"])

        self.assertEqual(phi4.scope(), ["x1", "x3"])
        for inp in np.random.rand(4, 2):
            np_test.assert_almost_equal(
                phi4.pdf(inp[0], inp[1]),
                multivariate_normal.pdf([inp[0], inp[1]], [0, 0], [[1, 0], [0, 1]]),
            )

        phi4.marginalize(["x3"])
        self.assertEqual(phi4.scope(), ["x1"])
        for inp in np.random.rand(1):
            np_test.assert_almost_equal(
                phi4.pdf(inp), multivariate_normal.pdf([inp], [0], [[1]])
            )

        phi4 = self.phi4.marginalize(["x2"], inplace=False)
        self.assertEqual(phi4.scope(), ["x1", "x3"])
        for inp in np.random.rand(4, 2):
            np_test.assert_almost_equal(
                phi4.pdf(inp[0], inp[1]),
                multivariate_normal.pdf([inp[0], inp[1]], [0, 0], [[1, 0], [0, 1]]),
            )

        phi4 = phi4.marginalize(["x3"], inplace=False)
        self.assertEqual(phi4.scope(), ["x1"])
        for inp in np.random.rand(1):
            np_test.assert_almost_equal(
                phi4.pdf(inp), multivariate_normal.pdf([inp], [0], [[1]])
            )

    def test_marginalize_error(self):
        self.assertRaises(TypeError, self.phi1.marginalize, "x1")
        self.assertRaises(TypeError, self.phi1.marginalize, set(["x", "y"]))
        self.assertRaises(TypeError, self.phi1.marginalize, {"x": 1, "y": 1})

        self.assertRaises(TypeError, self.phi4.marginalize, "x4")
        self.assertRaises(TypeError, self.phi4.marginalize, set(["x1", "x2", "x3"]))
        self.assertRaises(TypeError, self.phi4.marginalize, {"x1": 1, "x2": 1, "x3": 1})

        self.assertRaises(ValueError, self.phi1.marginalize, ["z"])
        self.assertRaises(ValueError, self.phi1.marginalize, ["x", "y", "z"])

        self.assertRaises(ValueError, self.phi4.marginalize, ["x4"])
        self.assertRaises(ValueError, self.phi4.marginalize, ["x1", "x2", "x3", "x4"])

    def test_normalize(self):
        def pdf2(x1, x2):
            return 2 * self.pdf2(x1, x2)

        phi2 = ContinuousFactor(["x1", "x2"], pdf2)
        phi4 = phi2.copy()

        phi4.normalize()
        self.assertEqual(phi4.scope(), phi2.scope())
        for inp in np.random.rand(1, 2):
            np_test.assert_almost_equal(
                phi4.pdf(inp[0], inp[1]), self.pdf2(inp[0], inp[1])
            )

        phi2.normalize()
        self.assertEqual(phi4.scope(), phi2.scope())
        for inp in np.random.rand(1, 2):
            np_test.assert_almost_equal(
                phi2.pdf(inp[0], inp[1]), phi4.pdf(inp[0], inp[1])
            )

    def test_operate(self):
        phi1 = self.phi1.copy()
        phi1._operate(self.phi2, "product")
        self.assertEqual(phi1.scope(), ["x", "y", "x1", "x2"])
        for inp in np.random.rand(4, 4):
            self.assertEqual(
                phi1.pdf(*inp),
                self.phi1.pdf(inp[0], inp[1]) * self.phi2.pdf(inp[2], inp[3]),
            )

        phi1 = self.phi1._operate(self.phi2, "product", inplace=False)
        self.assertEqual(phi1.scope(), ["x", "y", "x1", "x2"])
        for inp in np.random.rand(4, 4):
            self.assertEqual(
                phi1.pdf(*inp),
                self.phi1.pdf(inp[0], inp[1]) * self.phi2.pdf(inp[2], inp[3]),
            )

        phi1 = self.phi1 * self.phi2
        self.assertEqual(phi1.scope(), ["x", "y", "x1", "x2"])
        for inp in np.random.rand(4, 4):
            self.assertEqual(
                phi1.pdf(*inp),
                self.phi1.pdf(inp[0], inp[1]) * self.phi2.pdf(inp[2], inp[3]),
            )

        phi3 = self.phi3.copy()
        phi3._operate(self.phi1, "product")
        self.assertEqual(phi3.scope(), ["x", "y", "z"])
        for inp in np.random.rand(4, 3):
            self.assertEqual(
                phi3.pdf(*inp), self.phi3.pdf(*inp) * self.phi1.pdf(inp[0], inp[1])
            )

        phi3 = self.phi3._operate(self.phi1, "product", inplace=False)
        self.assertEqual(phi3.scope(), ["x", "y", "z"])
        for inp in np.random.rand(4, 3):
            self.assertEqual(
                phi3.pdf(*inp), self.phi3.pdf(*inp) * self.phi1.pdf(inp[0], inp[1])
            )

        phi3 = self.phi3 * self.phi1
        self.assertEqual(phi3.scope(), ["x", "y", "z"])
        for inp in np.random.rand(4, 3):
            self.assertEqual(
                phi3.pdf(*inp), self.phi3.pdf(*inp) * self.phi1.pdf(inp[0], inp[1])
            )

        phi3 = self.phi3.copy()
        phi3._operate(self.phi1, "divide")
        self.assertEqual(phi3.scope(), ["x", "y", "z"])
        for inp in np.random.rand(4, 3):
            self.assertEqual(
                phi3.pdf(*inp), self.phi3.pdf(*inp) / self.phi1.pdf(inp[0], inp[1])
            )

        phi3 = self.phi3._operate(self.phi1, "divide", inplace=False)
        self.assertEqual(phi3.scope(), ["x", "y", "z"])
        for inp in np.random.rand(4, 3):
            self.assertEqual(
                phi3.pdf(*inp), self.phi3.pdf(*inp) / self.phi1.pdf(inp[0], inp[1])
            )

        phi3 = self.phi3 / self.phi1
        self.assertEqual(phi3.scope(), ["x", "y", "z"])
        for inp in np.random.rand(4, 3):
            self.assertEqual(
                phi3.pdf(*inp), self.phi3.pdf(*inp) / self.phi1.pdf(inp[0], inp[1])
            )

        phi4 = self.phi4.copy()
        phi4._operate(self.phi2, "product")
        self.assertEqual(phi4.scope(), ["x1", "x2", "x3"])
        for inp in np.random.rand(4, 3):
            self.assertEqual(
                phi4.pdf(*inp), self.phi4.pdf(*inp) * self.phi2.pdf(inp[0], inp[1])
            )

        phi4 = self.phi4._operate(self.phi2, "product", inplace=False)
        self.assertEqual(phi4.scope(), ["x1", "x2", "x3"])
        for inp in np.random.rand(4, 3):
            self.assertEqual(
                phi4.pdf(*inp), self.phi4.pdf(*inp) * self.phi2.pdf(inp[0], inp[1])
            )

        phi4 = self.phi4 * self.phi2
        self.assertEqual(phi4.scope(), ["x1", "x2", "x3"])
        for inp in np.random.rand(4, 3):
            self.assertEqual(
                phi4.pdf(*inp), self.phi4.pdf(*inp) * self.phi2.pdf(inp[0], inp[1])
            )

        phi4 = self.phi4.copy()
        phi4._operate(self.phi2, "divide")
        self.assertEqual(phi4.scope(), ["x1", "x2", "x3"])
        for inp in np.random.rand(4, 3):
            self.assertEqual(
                phi4.pdf(*inp), self.phi4.pdf(*inp) / self.phi2.pdf(inp[0], inp[1])
            )

        phi4 = self.phi4._operate(self.phi2, "divide", inplace=False)
        self.assertEqual(phi4.scope(), ["x1", "x2", "x3"])
        for inp in np.random.rand(4, 3):
            self.assertEqual(
                phi4.pdf(*inp), self.phi4.pdf(*inp) / self.phi2.pdf(inp[0], inp[1])
            )

        phi4 = self.phi4 / self.phi2
        self.assertEqual(phi4.scope(), ["x1", "x2", "x3"])
        for inp in np.random.rand(4, 3):
            self.assertEqual(
                phi4.pdf(*inp), self.phi4.pdf(*inp) / self.phi2.pdf(inp[0], inp[1])
            )

    def test_operate_error(self):
        self.assertRaises(TypeError, self.phi1._operate, 1, "product")
        self.assertRaises(TypeError, self.phi1._operate, 1, "divide")
        self.assertRaises(TypeError, self.phi1._operate, "1", "product")
        self.assertRaises(TypeError, self.phi1._operate, "1", "divide")
        self.assertRaises(TypeError, self.phi1._operate, self.phi2.pdf, "product")
        self.assertRaises(TypeError, self.phi1._operate, self.phi2.pdf, "divide")
        self.assertRaises(TypeError, self.phi1._operate, [1], "product")
        self.assertRaises(TypeError, self.phi1._operate, [1], "divide")

        self.assertRaises(TypeError, self.phi4._operate, 1, "product")
        self.assertRaises(TypeError, self.phi4._operate, 1, "divide")
        self.assertRaises(TypeError, self.phi4._operate, "1", "product")
        self.assertRaises(TypeError, self.phi4._operate, "1", "divide")
        self.assertRaises(TypeError, self.phi4._operate, self.phi2.pdf, "product")
        self.assertRaises(TypeError, self.phi4._operate, self.phi2.pdf, "divide")
        self.assertRaises(TypeError, self.phi4._operate, [1], "product")
        self.assertRaises(TypeError, self.phi4._operate, [1], "divide")

        self.assertRaises(TypeError, self.phi1._operate, 1, "product", False)
        self.assertRaises(TypeError, self.phi1._operate, 1, "divide", False)
        self.assertRaises(TypeError, self.phi1._operate, "1", "product", False)
        self.assertRaises(TypeError, self.phi1._operate, "1", "divide", False)
        self.assertRaises(
            TypeError, self.phi1._operate, self.phi2.pdf, "product", False
        )
        self.assertRaises(TypeError, self.phi1._operate, self.phi2.pdf, "divide", False)
        self.assertRaises(TypeError, self.phi1._operate, [1], "product", False)
        self.assertRaises(TypeError, self.phi1._operate, [1], "divide", False)

        self.assertRaises(TypeError, self.phi4._operate, 1, "product", False)
        self.assertRaises(TypeError, self.phi4._operate, 1, "divide", False)
        self.assertRaises(TypeError, self.phi4._operate, "1", "product", False)
        self.assertRaises(TypeError, self.phi4._operate, "1", "divide", False)
        self.assertRaises(
            TypeError, self.phi4._operate, self.phi2.pdf, "product", False
        )
        self.assertRaises(TypeError, self.phi4._operate, self.phi2.pdf, "divide", False)
        self.assertRaises(TypeError, self.phi4._operate, [1], "product", False)
        self.assertRaises(TypeError, self.phi4._operate, [1], "divide", False)

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

        self.assertEqual(copy1.scope(), copy4.scope())
        self.assertEqual(copy1.pdf, copy4.pdf)
        self.assertEqual(copy2.scope(), copy5.scope())
        self.assertEqual(copy2.pdf, copy5.pdf)

    # TODO: Fix these
    #        copy1.variables = ['A', 'B']
    #        self.assertEqual(copy4.scope(), self.phi1.scope())

    #        def pdf(a, b):
    #            return (a + b) / (a * a + b * b)
    #        copy1._pdf = pdf
    #        copy1_pdf = pdf
    #        self.assertEqual(copy4.pdf, self.phi1.pdf)
    #        copy4.variables = ['X', 'Y']
    #        self.assertEqual(copy1.scope(), ['x', 'y'])
    #        copy4._pdf = lambda a, b: a + b
    #        for inp in np.random.rand(4, 2):
    #            self.assertEqual(copy1.pdf(inp[0], inp[1]), copy1_pdf(inp[0], inp[1]))

    #        copy2.reduce([('x', 7.7)])

    #        def reduced_pdf(y, z):
    #            return z*(np.power(7.7, 1) * np.power(y, 2)) / beta(7.7, y)
    #        self.assertEqual(copy5.scope(), self.phi3.scope())
    #        self.assertEqual(copy5.pdf, self.phi3.pdf)
    #        copy5.reduce([('x', 11), ('z', 13)])
    #        self.assertEqual(copy2.scope(), ['y', 'z'])
    #        for inp in np.random.rand(4, 2):
    #            self.assertEqual(copy2.pdf(inp[0], inp[1]), reduced_pdf(inp[0], inp[1]))

    def tearDown(self):
        del self.phi1
        del self.phi2
        del self.phi3
