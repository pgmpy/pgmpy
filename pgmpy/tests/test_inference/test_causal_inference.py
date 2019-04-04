import unittest

from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.inference.causal_inference import CausalInference


class TestCausalInferenceMethods(unittest.TestCase):

    def test_active_backdoor_game1(self):
        game1 = BayesianModel([('X', 'A'),
                               ('A', 'Y'),
                               ('A', 'B')])
        inference1 = CausalInference(game1)
        active_bds, bdg, bdr = inference1.check_active_backdoors(treatment="X", outcome="Y")
        self.assertEqual(active_bds, False)
        self.assertEqual(bdr, set())

    def test_active_backdoor_game3(self):
        game3 = BayesianModel([('X', 'Y'),
                               ('X', 'A'),
                               ('B', 'A'),
                               ('B', 'Y'),
                               ('B', 'X')])
        inference3 = CausalInference(game3)
        active_bds, bdg, bdr = inference3.check_active_backdoors(treatment="X", outcome="Y")
        self.assertEqual(active_bds, True)
        self.assertEqual(bdr, {"B"})

    def test_active_backdoor_game5(self):
        game5 = BayesianModel([('A', 'X'),
                               ('A', 'B'),
                               ('C', 'B'),
                               ('C', 'Y'),
                               ('X', 'Y'),
                               ('B', 'X')])
        inference5 = CausalInference(game5)
        active_bds, bdg, bdr = inference5.check_active_backdoors(treatment="X", outcome="Y")
        self.assertEqual(active_bds, True)
        self.assertEqual(bdr, {"A", "B"})


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
        deconfounders = inference.get_deconfounders(treatment="X", outcome="Y")
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
        deconfounders = inference.get_deconfounders(treatment="X", outcome="Y")
        print(deconfounders)
        self.assertEqual(deconfounders, {frozenset({'C'}),
                                         frozenset({'A', 'B'})})
