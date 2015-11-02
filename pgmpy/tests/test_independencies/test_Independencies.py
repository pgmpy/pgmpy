import unittest

from pgmpy import independencies
from pgmpy import exceptions
from pgmpy.extern.six.moves import zip


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


class TestIndependencies(unittest.TestCase):
    def setUp(self):
        self.Independencies1 = independencies.Independencies()
        self.Independencies = independencies.Independencies()

    def test_init(self):
        self.Independencies1 = independencies.Independencies(['X', 'Y', 'Z'])
        self.assertEqual(self.Independencies1, independencies.Independencies(['X', 'Y', 'Z']))

    def test_add_assertions(self):
        self.Independencies1.add_assertions(['A', 'B', 'C'], ['D', 'E', 'F'])
        self.assertEqual(self.Independencies1, independencies.Independencies(['D', 'E', 'F'], ['A', 'B', 'C']))

    def test_get_assertions(self):
        self.Independencies1.add_assertions(['A', 'B', 'C'])
        self.assertEqual(self.Independencies1.get_assertions(), self.Independencies1.independencies)

    def tearUp(self):
        del self.Independencies


if __name__ == '__main__':
    unittest.main()
