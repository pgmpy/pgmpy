import unittest

from pgmpy import independencies
from pgmpy import exceptions
from pgmpy.extern.six.moves import zip
from pgmpy.independencies import IndependenceAssertion


class TestIndependenceAssertion(unittest.TestCase):
    def setUp(self):
        self.assertion = independencies.IndependenceAssertion()

    def test_return_list_if_str(self):
        self.assertListEqual(self.assertion._return_list_if_str('U'), ['U'])
        self.assertListEqual(self.assertion._return_list_if_str(['U', 'V']), ['U', 'V'])

    def test_set_assertion(self):
        self.assertion.set_assertion('U', 'V', 'Z')
        self.assertSetEqual(self.assertion.event1, {'U'})
        self.assertSetEqual(self.assertion.event2, {'V'})
        self.assertSetEqual(self.assertion.event3, {'Z'})
        self.assertion.set_assertion(['U', 'V'], ['Y', 'Z'], ['A', 'B'])
        self.assertSetEqual(self.assertion.event1, {'U', 'V'})
        self.assertSetEqual(self.assertion.event2, {'Y', 'Z'})
        self.assertSetEqual(self.assertion.event3, {'A', 'B'})
        self.assertion.set_assertion(['U', 'V'], ['Y', 'Z'])
        self.assertSetEqual(self.assertion.event1, {'U', 'V'})
        self.assertSetEqual(self.assertion.event2, {'Y', 'Z'})
        self.assertFalse(self.assertion.event3, {})

    def test_get_assertion(self):
        self.assertion.set_assertion('U', 'V', 'Z')
        self.assertTupleEqual(self.assertion.get_assertion(), ({'U'}, {'V'}, {'Z'}))
        self.assertion.set_assertion('U', 'V')
        self.assertTupleEqual(self.assertion.get_assertion(), ({'U'}, {'V'}, set()))

    def test_init(self):
        self.assertion1 = independencies.IndependenceAssertion('U', 'V', 'Z')
        self.assertSetEqual(self.assertion1.event1, {'U'})
        self.assertSetEqual(self.assertion1.event2, {'V'})
        self.assertSetEqual(self.assertion1.event3, {'Z'})
        self.assertion1 = independencies.IndependenceAssertion(['U', 'V'], ['Y', 'Z'], ['A', 'B'])
        self.assertSetEqual(self.assertion1.event1, {'U', 'V'})
        self.assertSetEqual(self.assertion1.event2, {'Y', 'Z'})
        self.assertSetEqual(self.assertion1.event3, {'A', 'B'})

    def test_init_exceptions(self):
        self.assertRaises(exceptions.RequiredError, independencies.IndependenceAssertion, event2=['U'], event3='V')
        self.assertRaises(exceptions.RequiredError, independencies.IndependenceAssertion, event2=['U'])
        self.assertRaises(exceptions.RequiredError, independencies.IndependenceAssertion, event3=['Z'])
        self.assertRaises(exceptions.RequiredError, independencies.IndependenceAssertion, event1=['U'])
        self.assertRaises(exceptions.RequiredError, independencies.IndependenceAssertion, event1=['U'], event3=['Z'])        

    def tearDown(self):
        del self.assertion


class TestIndependeciesAssertionEq(unittest.TestCase):
    def setUp(self):
        self.i1 = IndependenceAssertion('a', 'b', 'c')
        self.i2 = IndependenceAssertion('a', 'b')
        self.i3 = IndependenceAssertion('a', ['b','c','d'])
        self.i4 = IndependenceAssertion('a', ['b','c','d'], 'e')
        self.i5 = IndependenceAssertion('a', ['d','c','b'], 'e')
        self.i6 = IndependenceAssertion('a', ['d','c'], ['e','b'])
        self.i7 = IndependenceAssertion('a', ['c','d'], ['b','e'])
        self.i8 = IndependenceAssertion('a', ['f','d'], ['b','e'])
        self.i9 = IndependenceAssertion('a', ['d','k','b'], 'e')
        self.i10 = IndependenceAssertion(['k','b','d'], 'a', 'e')

    def test_eq1(self):
        self.assertFalse(self.i1 == 'a')
        self.assertFalse(self.i2 == 1)
        self.assertFalse(self.i4 == [2,'a'])
        self.assertFalse(self.i6 == 'c')

    def test_eq2(self):
        self.assertFalse(self.i1 == self.i2)
        self.assertFalse(self.i1 == self.i3)
        self.assertFalse(self.i2 == self.i4)
        self.assertFalse(self.i3 == self.i6)

    def test_eq3(self):
        self.assertTrue(self.i4 == self.i5)
        self.assertTrue(self.i6 == self.i7)
        self.assertFalse(self.i7 == self.i8)
        self.assertFalse(self.i4 == self.i9)
        self.assertFalse(self.i5 == self.i9)
        self.assertTrue(self.i10 == self.i9)
        self.assertTrue(self.i10 != self.i8)

    def tearDown(self):
        del self.i1
        del self.i2
        del self.i3
        del self.i4
        del self.i5
        del self.i6
        del self.i7
        del self.i8
        del self.i9
        del self.i10

class TestIndependencies(unittest.TestCase):
    def setUp(self):
        self.Independencies = independencies.Independencies()
        self.Independencies3 = independencies.Independencies(['a', ['b', 'c', 'd'], ['e', 'f', 'g']],
                                                             ['c', ['d', 'e' ,'f'], ['g' , 'h']])
        self.Independencies4 = independencies.Independencies([['f', 'd', 'e'], 'c', ['h', 'g']],
                                                             [['b', 'c', 'd'], 'a', ['f', 'g', 'e']])
        self.Independencies5 = independencies.Independencies(['a', ['b', 'c', 'd'], ['e', 'f', 'g']],
                                                             ['c', ['d', 'e', 'f'], 'g'])

    def test_init(self):
        self.Independencies1 = independencies.Independencies(['X', 'Y', 'Z'])
        self.assertEqual(self.Independencies1, independencies.Independencies(['X', 'Y', 'Z']))
        self.Independencies2 = independencies.Independencies()
        self.assertEqual(self.Independencies2, independencies.Independencies())

    def test_add_assertions(self):
        self.Independencies1 = independencies.Independencies()
        self.Independencies1.add_assertions(['X', 'Y', 'Z'])
        self.assertEqual(self.Independencies1, independencies.Independencies(['X', 'Y', 'Z']))
        self.Independencies2 = independencies.Independencies()
        self.Independencies2.add_assertions(['A', 'B', 'C'], ['D', 'E', 'F'])
        self.assertEqual(self.Independencies2, independencies.Independencies(['A', 'B', 'C'], ['D', 'E', 'F']))

    def test_get_assertions(self):
        self.Independencies1 = independencies.Independencies()
        self.Independencies1.add_assertions(['X', 'Y', 'Z'])
        self.assertEqual(self.Independencies1.independencies, self.Independencies1.get_assertions())
        self.Independencies2 = independencies.Independencies(['A', 'B', 'C'], ['D', 'E', 'F'])
        self.assertEqual(self.Independencies2.independencies, self.Independencies2.get_assertions())

    def test_e1(self):
        self.assertTrue(self.Independencies3 == self.Independencies4)
        self.assertFalse(self.Independencies3 != self.Independencies4)
        self.assertTrue(self.Independencies3 != self.Independencies5)
        self.assertFalse(self.Independencies4 == self.Independencies5)

    def tearDown(self):
        del self.Independencies
        del self.Independencies3
        del self.Independencies4
        del self.Independencies5


if __name__ == '__main__':
    unittest.main()
