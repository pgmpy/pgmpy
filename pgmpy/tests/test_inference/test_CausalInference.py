import unittest

from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.inference.CausalInference import CausalInference


class TestBackdoorPaths(unittest.TestCase):
    """
    These tests are drawn from games presented in The Book of Why by Judea Pearl.

    TODO:
      * Tests that can assert over sets of confoundering variables
      * Tests that don't assume that X is the treatment and Y is the outcome
    """
    def test_game1(self):
        game1 = BayesianModel([('X', 'A'),
                               ('A', 'Y'),
                               ('A', 'B')])
        deconfounders = game1.get_deconfounders(treatment="X", outcome="Y")
        self.assertEqual(deconfounders, [])

    def test_game2(self):
        game2 = BayesianModel([('X', 'E'),
                               ('E', 'Y'),
                               ('A', 'X'),
                               ('A', 'B'),
                               ('B', 'C'),
                               ('D', 'B'),
                               ('D', 'E')])
        deconfounders = game2.get_deconfounders(treatment="X", outcome="Y")
        self.assertEqual(deconfounders, [])

    def test_game3(self):
        game3 = BayesianModel([('X', 'Y'),
                               ('X', 'A'),
                               ('B', 'A'),
                               ('B', 'Y'),
                               ('B', 'X')])
        deconfounders = game3.get_deconfounders(treatment="X",
                                                outcome="Y",
                                                maxdepth=1)
        self.assertEqual(deconfounders, [('B',)])

    def test_game4(self):
        game4 = BayesianModel([('A', 'X'),
                               ('A', 'B'),
                               ('C', 'B'),
                               ('C', 'Y')])
        deconfounders = game4.get_deconfounders(treatment="X", outcome="Y")
        self.assertEqual(deconfounders, [])

    def test_game5(self):
        game5 = BayesianModel([('A', 'X'),
                               ('A', 'B'),
                               ('C', 'B'),
                               ('C', 'Y'),
                               ('X', 'Y'),
                               ('B', 'X')])
        deconfounders = game5.get_deconfounders(treatment="X",
                                                outcome="Y",
                                                maxdepth=1)
        self.assertEqual(deconfounders, [('C',),])
