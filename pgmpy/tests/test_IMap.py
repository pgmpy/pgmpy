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