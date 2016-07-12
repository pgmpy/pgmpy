import unittest

from pgmpy.independencies import Independencies, IndependenceAssertion


class TestIndependenceAssertion(unittest.TestCase):
    def setUp(self):
        self.assertion = IndependenceAssertion()

    def test_return_list_if_str(self):
        self.assertListEqual(self.assertion._return_list_if_str('U'), ['U'])
        self.assertListEqual(self.assertion._return_list_if_str(['U', 'V']), ['U', 'V'])

    def test_get_assertion(self):
        self.assertTupleEqual(IndependenceAssertion('U', 'V', 'Z').get_assertion(), ({'U'}, {'V'}, {'Z'}))
        self.assertTupleEqual(IndependenceAssertion('U', 'V').get_assertion(), ({'U'}, {'V'}, set()))

    def test_init(self):
        self.assertion1 = IndependenceAssertion('U', 'V', 'Z')
        self.assertSetEqual(self.assertion1.event1, {'U'})
        self.assertSetEqual(self.assertion1.event2, {'V'})
        self.assertSetEqual(self.assertion1.event3, {'Z'})
        self.assertion1 = IndependenceAssertion(['U', 'V'], ['Y', 'Z'], ['A', 'B'])
        self.assertSetEqual(self.assertion1.event1, {'U', 'V'})
        self.assertSetEqual(self.assertion1.event2, {'Y', 'Z'})
        self.assertSetEqual(self.assertion1.event3, {'A', 'B'})

    def test_init_exceptions(self):
        self.assertRaises(ValueError, IndependenceAssertion, event2=['U'], event3='V')
        self.assertRaises(ValueError, IndependenceAssertion, event2=['U'])
        self.assertRaises(ValueError, IndependenceAssertion, event3=['Z'])
        self.assertRaises(ValueError, IndependenceAssertion, event1=['U'])
        self.assertRaises(ValueError, IndependenceAssertion, event1=['U'], event3=['Z'])

    def tearDown(self):
        del self.assertion


class TestIndependeciesAssertionEq(unittest.TestCase):
    def setUp(self):
        self.i1 = IndependenceAssertion('a', 'b', 'c')
        self.i2 = IndependenceAssertion('a', 'b')
        self.i3 = IndependenceAssertion('a', ['b', 'c', 'd'])
        self.i4 = IndependenceAssertion('a', ['b', 'c', 'd'], 'e')
        self.i5 = IndependenceAssertion('a', ['d', 'c', 'b'], 'e')
        self.i6 = IndependenceAssertion('a', ['d', 'c'], ['e', 'b'])
        self.i7 = IndependenceAssertion('a', ['c', 'd'], ['b', 'e'])
        self.i8 = IndependenceAssertion('a', ['f', 'd'], ['b', 'e'])
        self.i9 = IndependenceAssertion('a', ['d', 'k', 'b'], 'e')
        self.i10 = IndependenceAssertion(['k', 'b', 'd'], 'a', 'e')

    def test_eq1(self):
        self.assertFalse(self.i1 == 'a')
        self.assertFalse(self.i2 == 1)
        self.assertFalse(self.i4 == [2, 'a'])
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
        self.Independencies = Independencies()
        self.Independencies3 = Independencies(['a', ['b', 'c', 'd'], ['e', 'f', 'g']],
                                              ['c', ['d', 'e', 'f'], ['g', 'h']])
        self.Independencies4 = Independencies([['f', 'd', 'e'], 'c', ['h', 'g']],
                                              [['b', 'c', 'd'], 'a', ['f', 'g', 'e']])
        self.Independencies5 = Independencies(['a', ['b', 'c', 'd'], ['e', 'f', 'g']],
                                              ['c', ['d', 'e', 'f'], 'g'])

    def test_init(self):
        self.Independencies1 = Independencies(['X', 'Y', 'Z'])
        self.assertEqual(self.Independencies1, Independencies(['X', 'Y', 'Z']))
        self.Independencies2 = Independencies()
        self.assertEqual(self.Independencies2, Independencies())

    def test_add_assertions(self):
        self.Independencies1 = Independencies(['X', 'Y', 'Z'])
        self.assertEqual(self.Independencies1, Independencies(['X', 'Y', 'Z']))
        self.Independencies2 = Independencies(['A', 'B', 'C'], ['D', 'E', 'F'])
        self.assertEqual(self.Independencies2, Independencies(['A', 'B', 'C'], ['D', 'E', 'F']))

    def test_get_assertions(self):
        self.Independencies1 = Independencies(['X', 'Y', 'Z'])
        self.assertEqual(self.Independencies1.independencies, self.Independencies1.get_assertions())
        self.Independencies2 = Independencies(['A', 'B', 'C'], ['D', 'E', 'F'])
        self.assertEqual(self.Independencies2.independencies, self.Independencies2.get_assertions())

    def test_closure(self):
        ind1 = Independencies(('A', ['B', 'C'], 'D'))
        self.assertEqual(ind1.closure(), Independencies(('A', ['B', 'C'], 'D'),
                                                        ('A', 'B', ['C', 'D']),
                                                        ('A', 'C', ['B', 'D']),
                                                        ('A', 'B', 'D'),
                                                        ('A', 'C', 'D')))
        ind2 = Independencies(('W', ['X', 'Y', 'Z']))
        self.assertEqual(ind2.closure(),
                         Independencies(
                             ('W', 'Y'), ('W', 'Y', 'X'), ('W', 'Y', 'Z'), ('W', 'Y', ['X', 'Z']),
                             ('W', ['Y', 'X']), ('W', 'X', ['Y', 'Z']), ('W', ['X', 'Z'], 'Y'),
                             ('W', 'X'), ('W', ['X', 'Z']), ('W', ['Y', 'Z'], 'X'),
                             ('W', ['Y', 'X', 'Z']), ('W', 'X', 'Z'), ('W', ['Y', 'Z']),
                             ('W', 'Z', 'X'), ('W', 'Z'), ('W', ['Y', 'X'], 'Z'), ('W', 'X', 'Y'),
                             ('W', 'Z', ['Y', 'X']), ('W', 'Z', 'Y')))
        ind3 = Independencies(('c', 'a', ['b', 'e', 'd']), (['e', 'c'], 'b', ['a', 'd']), (['b', 'd'], 'e', 'a'),
                              ('e', ['b', 'd'], 'c'), ('e', ['b', 'c'], 'd'), (['e', 'c'], 'a', 'b'))
        self.assertEqual(len(ind3.closure().get_assertions()), 78)

    def test_entails(self):
        ind1 = Independencies([['A', 'B'], ['C', 'D'], 'E'])
        ind2 = Independencies(['A', 'C', 'E'])
        self.assertTrue(ind1.entails(ind2))
        self.assertFalse(ind2.entails(ind1))
        ind3 = Independencies(('W', ['X', 'Y', 'Z']))
        self.assertTrue(ind3.entails(ind3.closure()))
        self.assertTrue(ind3.closure().entails(ind3))

    def test_is_equivalent(self):
        ind1 = Independencies(['X', ['Y', 'W'], 'Z'])
        ind2 = Independencies(['X', 'Y', 'Z'], ['X', 'W', 'Z'])
        ind3 = Independencies(['X', 'Y', 'Z'], ['X', 'W', 'Z'], ['X', 'Y', ['W', 'Z']])
        self.assertFalse(ind1.is_equivalent(ind2))
        self.assertTrue(ind1.is_equivalent(ind3))

    def test_eq(self):
        self.assertTrue(self.Independencies3 == self.Independencies4)
        self.assertFalse(self.Independencies3 != self.Independencies4)
        self.assertTrue(self.Independencies3 != self.Independencies5)
        self.assertFalse(self.Independencies4 == self.Independencies5)
        self.assertFalse(Independencies() == Independencies(['A', 'B', 'C']))
        self.assertFalse(Independencies(['A', 'B', 'C']) == Independencies())
        self.assertTrue(Independencies() == Independencies())

    def tearDown(self):
        del self.Independencies
        del self.Independencies3
        del self.Independencies4
        del self.Independencies5


if __name__ == '__main__':
    unittest.main()
