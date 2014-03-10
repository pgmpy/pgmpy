import unittest
import numpy as np
from pgmpy.Factor import Factor
from pgmpy import Exceptions
from collections import OrderedDict

if __name__ == '__main__':
        unittest.main()

class TestFactorOperations(unittest.TestCase):

    def setUp(self):
        self.phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], np.ones(12))

    def test_init_Factor(self):
        self.assertEqual(self.phi.variables, OrderedDict([('x1', ['x1_0', 'x1_1']),
                                                          ('x2', ['x2_0', 'x2_1', 'x2_2']),
                                                          ('x3', ['x3_0', 'x3_1'])]))
        self.assertTrue((self.phi.cardinality == np.array([2, 3, 2])).all())

    def test_init_incompetant_value(self):
        self.assertRaises(Exceptions.SizeError, Factor, ['x1', 'x2', 'x3'], [2, 3, 2], np.ones(8))

    def tearDown(self):
        del self.phi
