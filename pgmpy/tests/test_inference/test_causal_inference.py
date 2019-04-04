import unittest

from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.inference.causal_inference import CausalInference


class TestCausalInferenceMethods(unittest.TestCase):
    def setup(self):
        self.game1 = BayesianModel([('X', 'A'),
                                    ('A', 'Y'),
                                    ('A', 'B')])
        self.inference1 = CausalInference(self.game1)

        self.game3 = BayesianModel([('X', 'Y'),
                                    ('X', 'A'),
                                    ('B', 'A'),
                                    ('B', 'Y'),
                                    ('B', 'X')])
        self.inference3 = CausalInference(self.game3)

        self.game5 = BayesianModel([('A', 'X'),
                                    ('A', 'B'),
                                    ('C', 'B'),
                                    ('C', 'Y'),
                                    ('X', 'Y'),
                                    ('B', 'X')])
        self.inference5 = CausalInference(self.game5)

    def active_backdoor_game1(self):
        active_bds, bdg, bdr = self.inference1.check_active_backdoors(treatment="X", outcome="Y")
        self.assertEqual(active_bds, False)
        self.assertEqual(bdr, set())

    def teardown(self):
        del self.game


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
        inference = CausalInference(model=game1)
        deconfounders = inference.get_deconfounders(treatment="X", outcome="Y")
        self.assertEqual(deconfounders, set())

    def test_game2(self):
        game2 = BayesianModel([('X', 'E'),
                               ('E', 'Y'),
                               ('A', 'X'),
                               ('A', 'B'),
                               ('B', 'C'),
                               ('D', 'B'),
                               ('D', 'E')])
        inference = CausalInference(model=game2)
        deconfounders = inference.get_deconfounders(treatment="X", outcome="Y")
        self.assertEqual(deconfounders, set())

    def test_game3(self):
        game3 = BayesianModel([('X', 'Y'),
                               ('X', 'A'),
                               ('B', 'A'),
                               ('B', 'Y'),
                               ('B', 'X')])
        inference = CausalInference(model=game3)
        deconfounders = inference.get_deconfounders(treatment="X",
                                                    outcome="Y",
                                                    maxdepth=1)
        self.assertEqual(deconfounders, {frozenset({'B'})})

    def test_game4(self):
        game4 = BayesianModel([('A', 'X'),
                               ('A', 'B'),
                               ('C', 'B'),
                               ('C', 'Y')])
        inference = CausalInference(model=game4)
        deconfounders = inference.get_deconfounders(treatment="X", outcome="Y")
        self.assertEqual(deconfounders, set())

    def test_game5(self):
        game5 = BayesianModel([('A', 'X'),
                               ('A', 'B'),
                               ('C', 'B'),
                               ('C', 'Y'),
                               ('X', 'Y'),
                               ('B', 'X')])
        inference = CausalInference(model=game5)
        deconfounders = inference.get_deconfounders(treatment="X",
                                                    outcome="Y",
                                                    maxdepth=1)
        self.assertEqual(deconfounders, {frozenset({'C'})})
