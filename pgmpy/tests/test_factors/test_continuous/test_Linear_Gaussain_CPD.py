import unittest

import numpy.testing as np_test

from pgmpy.factors.continuous import LinearGaussianCPD


class TestLGCPD(unittest.TestCase):
    def test_class_init(self):
        cpd1 = LinearGaussianCPD('x', [0.23], 0.56)
        self.assertEqual(cpd1.variable, 'x')
        self.assertEqual(cpd1.beta_0, 0.23)
        self.assertEqual(cpd1.variance, 0.56)

        cpd2 = LinearGaussianCPD('y', [0.67, 1, 4.56, 8], 2,
                                 ['x1', 'x2', 'x3'])
        self.assertEqual(cpd2.variable, 'y')
        self.assertEqual(cpd2.beta_0, 0.67)
        self.assertEqual(cpd2.variance, 2)
        self.assertEqual(cpd2.evidence, ['x1', 'x2', 'x3'])
        np_test.assert_array_equal(cpd2.beta_vector, [1, 4.56, 8])

        self.assertRaises(ValueError, LinearGaussianCPD, 'x', [1, 1, 2], 2,
                          ['a', 'b', 'c'])
        self.assertRaises(ValueError, LinearGaussianCPD, 'x', [1, 1, 2, 3], 2,
                          ['a', 'b'])

    def test_pdf(self):
        cpd1 = LinearGaussianCPD('x', [0.23], 0.56)
        cpd2 = LinearGaussianCPD('y', [0.67, 1, 4.56, 8], 2, ['x1', 'x2', 'x3'])
        np_test.assert_almost_equal(cpd1.assignment(1), 0.3139868)
        np_test.assert_almost_equal(cpd2.assignment(1, 1.2, 2.3, 3.4), 1.076e-162)

    def test_copy(self):
        cpd = LinearGaussianCPD('y', [0.67, 1, 4.56, 8], 2, ['x1', 'x2', 'x3'])
        copy = cpd.copy()

        self.assertEqual(cpd.variable, copy.variable)
        self.assertEqual(cpd.beta_0, copy.beta_0)
        self.assertEqual(cpd.variance, copy.variance)
        np_test.assert_array_equal(cpd.beta_vector, copy.beta_vector)
        self.assertEqual(cpd.evidence, copy.evidence)

        cpd.variable = 'z'
        self.assertEqual(copy.variable, 'y')
        cpd.variance = 0
        self.assertEqual(copy.variance, 2)
        cpd.beta_0 = 1
        self.assertEqual(copy.beta_0, 0.67)
        cpd.evidence = ['p', 'q', 'r']
        self.assertEqual(copy.evidence, ['x1', 'x2', 'x3'])
        cpd.beta_vector = [2, 2, 2]
        np_test.assert_array_equal(copy.beta_vector, [1, 4.56, 8])

        copy = cpd.copy()

        copy.variable = 'k'
        self.assertEqual(cpd.variable, 'z')
        copy.variance = 0.3
        self.assertEqual(cpd.variance, 0)
        copy.beta_0 = 1.5
        self.assertEqual(cpd.beta_0, 1)
        copy.evidence = ['a', 'b', 'c']
        self.assertEqual(cpd.evidence, ['p', 'q', 'r'])
        copy.beta_vector = [2.2, 2.2, 2.2]
        np_test.assert_array_equal(cpd.beta_vector, [2, 2, 2])

    def test_str(self):
        cpd1 = LinearGaussianCPD('x', [0.23], 0.56)
        cpd2 = LinearGaussianCPD('y', [0.67, 1, 4.56, 8], 2, ['x1', 'x2', 'x3'])
        self.assertEqual(cpd1.__str__(), "P(x) = N(0.23; 0.56)")
        self.assertEqual(cpd2.__str__(), "P(y | x1, x2, x3) = N(1.0*x1 + "
                                         "4.56*x2 + 8.0*x3 + 0.67; 2)")
