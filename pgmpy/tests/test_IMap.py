import unittest
import BayesianModel.IMap as imap
from Exceptions import Exceptions


class TestIndependenceAssertion(unittest.TestCase):

    def setUp(self):
        self.assertion = imap.IndependenceAssertion()

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
        self.assertSetEqual(self.assertion.event3, {})

    def test_get_assertion(self):
        self.assertion.set_assertion('U', 'V', 'Z')
        self.assertTupleEqual(self.assertion.get_assertion(), ({'U'}, {'V'}, {'Z'}))
        self.assertion.set_assertion('U', 'V')
        self.assertTupleEqual(self.assertion.get_assertion(), ({'U'}, {'V'}))

    def test_init(self):
        self.assertion1 = imap.IndependenceAssertion('U', 'V', 'Z')
        self.assertSetEqual(self.assertion1.event1, {'U'})
        self.assertSetEqual(self.assertion1.event2, {'V'})
        self.assertSetEqual(self.assertion2.event3, {'Z'})
        self.assertion1 = imap.IndependentceAssertion(['U', 'V'], ['Y', 'Z'], ['A', 'B'])
        self.assertSetEqual(self.assertion1.event1, {'U', 'V'})
        self.assertSetEqual(self.assertion1.event2, {'Y', 'Z'})
        self.assertSetEqual(self.assertion1.event3, {'A', 'B'})

    def test_init_exceptions(self):
        self.assertRaises(Exceptions.Required, imap.IndependenceAssertion, event2=['U'], event3='V')
        self.assertRaises(Exceptions.Required, imap.IndependenceAssertion, event2=['U'])
        self.assertRaises(Exceptions.Required, imap.IndependenceAssertion, event3=['Z'])
        self.assertRaises(Exceptions.Required, imap.IndependenceAssertion, event1=['U'])
        self.assertRaises(Exceptions.Required, imap.IndependenceAssertion, event1=['U'], event3=['Z'])

    def tearDown(self):
        del self.assertion


class IMapTestCase:

    def assertIndependenceAssertionEqual(self, assertion1, assertion2):
        if not (assertion1.event1 == assertion2.event1):
            raise AssertionError(assertion1.event1 + "is not equal to" + assertion2.event1)
        if not (assertion1.event2 == assertion2.event2):
            raise AssertionError(assertion1.event2 + "is not equal to" + assertion2.event2)
        if not (assertion1.event3 == assertion2.event3):
            raise AssertionError(assertion1.event3 + "is not equal to" + assertion2.event3)

    def assertImapEqual(self, imap1, imap2):
        if len(imap1) == len(imap2):
            for map1, map2 in zip(imap1, imap2):
                self.assertIndependenceAssertionEqual(map1, map2)


class TestIMap(unittest.TestCase, IMapTestCase):

    def setUp(self):
        self.imap1 = imap.IMap()

    def test_init(self):
        self.imap2 = imap.IMap(['X', 'Y'])
        self.assertIndependenceAssertionEqual(self.imap2.imap.pop, imap.IndependenceAssertion('X', 'Y'))
        self.imap2 = imap.IMap(['X', 'Y', 'Z'])
        self.assertIndependenceAssertionEqual(self.imap2.imap.pop, imap.IndependenceAssertion('X', 'Y', 'Z'))
        self.imap2 = imap.IMap(['X', 'Y'], ['A', 'B', 'C'], [['L'], ['M', 'N'], 'O'])
        self.assertImapEqual(self.imap2.imap, {imap.IndependenceAssertion('X', 'Y'),
                                               imap.IndependenceAssertion('A', 'B', 'C'),
                                               imap.IndependenceAssertion(['L'], ['M', 'N'], 'O')})

    def test_add_assertions(self):
        self.imap1.add_assertions(['X', 'Y', 'Z'])
        self.assertIndependenceAssertionEqual(self.imap1.imap.pop, imap.IndependenceAssertion(['X', 'Y', 'Z']))

    def test_get_imap(self):
        self.imap1.add_assertions(['X', 'Y', 'Z'])
        self.assertImapEqual(self.imap1.imap, {imap.IndependenceAssertion('X', 'Y', 'Z')})
        self.imap1.add_assertions([['A', 'B'], ['C', 'D'], ['E', 'F']])
        self.assertImapEqual(self.imap1.imap, {imap.IndependenceAssertion('X', 'Y', 'Z'),
                                               imap.IndependenceAssertion(['A', 'B'], ['C', 'D'], ['E', 'F'])})

    def tearUp(self):
        del self.imap