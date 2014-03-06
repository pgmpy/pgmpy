import unittest
from pgmpy import Independencies
from pgmpy import Exceptions


class TestIndependenceAssertion(unittest.TestCase):

    def setUp(self):
        self.assertion = Independencies.IndependenceAssertion()

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
        self.assertion1 = Independencies.IndependenceAssertion('U', 'V', 'Z')
        self.assertSetEqual(self.assertion1.event1, {'U'})
        self.assertSetEqual(self.assertion1.event2, {'V'})
        self.assertSetEqual(self.assertion1.event3, {'Z'})
        self.assertion1 = Independencies.IndependenceAssertion(['U', 'V'], ['Y', 'Z'], ['A', 'B'])
        self.assertSetEqual(self.assertion1.event1, {'U', 'V'})
        self.assertSetEqual(self.assertion1.event2, {'Y', 'Z'})
        self.assertSetEqual(self.assertion1.event3, {'A', 'B'})

    def test_init_exceptions(self):
        self.assertRaises(Exceptions.RequiredError, Independencies.IndependenceAssertion, event2=['U'], event3='V')
        self.assertRaises(Exceptions.RequiredError, Independencies.IndependenceAssertion, event2=['U'])
        self.assertRaises(Exceptions.RequiredError, Independencies.IndependenceAssertion, event3=['Z'])
        self.assertRaises(Exceptions.RequiredError, Independencies.IndependenceAssertion, event1=['U'])
        self.assertRaises(Exceptions.RequiredError, Independencies.IndependenceAssertion, event1=['U'], event3=['Z'])

    def tearDown(self):
        del self.assertion


class IndependenciesTestCase:

    def assertIndependenceAssertionEqual(self, assertion1, assertion2):
        if not (assertion1.event1 == assertion2.event1):
            raise AssertionError(str(assertion1.event1) + "is not equal to" + str(assertion2.event1))
        if not (assertion1.event2 == assertion2.event2):
            raise AssertionError(str(assertion1.event2) + "is not equal to" + str(assertion2.event2))
        if not (assertion1.event3 == assertion2.event3):
            raise AssertionError(str(assertion1.event3) + "is not equal to" + str(assertion2.event3))

    def assertIndependenciesEqual(self, Independencies1, Independencies2):
        if len(Independencies1) == len(Independencies2):
            for map1, map2 in zip(Independencies1, Independencies2):
                self.assertIndependenceAssertionEqual(map1, map2)


class TestIndependencies(unittest.TestCase, IndependenciesTestCase):

    def setUp(self):
        self.Independencies1 = Independencies.Independencies()

    def test_init(self):
        self.Independencies2 = Independencies.Independencies(['X', 'Y'])
        #self.assertIndependenceAssertionEqual(self.Independencies2.independencies.pop, Independencies.IndependenceAssertion('X', 'Y'))
        #self.Independencies2 = Independencies.IMap(['X', 'Y', 'Z'])
        #self.assertIndependenceAssertionEqual(self.Independencies2.independencies.pop, Independencies.IndependenceAssertion('X', 'Y', 'Z'))
        #self.Independencies2 = Independencies.IMap(['X', 'Y'], ['A', 'B', 'C'], [['L'], ['M', 'N'], 'O'])
        #self.assertIndependenciesEqual(self.Independencies2.Independencies, {Independencies.IndependenceAssertion('X', 'Y'),
        #                                       Independencies.IndependenceAssertion('A', 'B', 'C'),
        #                                       Independencies.IndependenceAssertion(['L'], ['M', 'N'], 'O')})

    def test_add_assertions(self):
        self.Independencies1.add_assertions(['X', 'Y', 'Z'])
        #self.assertIndependenceAssertionEqual(self.Independencies1.independencies.pop, Independencies.IndependenceAssertion(['X', 'Y', 'Z']))

    def test_get_Independencies(self):
        self.Independencies1.add_assertions(['X', 'Y', 'Z'])
        self.assertIndependenciesEqual(self.Independencies1.independencies, {Independencies.IndependenceAssertion('X', 'Y', 'Z')})
        self.Independencies1.add_assertions([['A', 'B'], ['C', 'D'], ['E', 'F']])
        #self.assertIndependenciesEqual(self.Independencies1.independencies, {Independencies.IndependenceAssertion('X', 'Y', 'Z'),
        #                                       Independencies.IndependenceAssertion(['A', 'B'], ['C', 'D'], ['E', 'F'])})

    def tearUp(self):
        del self.Independencies