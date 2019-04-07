import unittest

from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.inference.causal_inference import CausalInference


class TestCausalInferenceMethods(unittest.TestCase):

    def test_display_methods(self):
        game1 = BayesianModel([('X', 'A'),
                               ('A', 'Y'),
                               ('A', 'B')])
        inference1 = CausalInference(game1)
        self.assertEqual("CausalInference(A, B, X, Y)", inference1.__repr__())
        self.assertEqual("P(X)P(A|X)P(B|A)P(Y|A)", inference1.get_distribution())

    def test_do_operator(self):
        game1 = BayesianModel([('X', 'A'),
                               ('A', 'Y'),
                               ('A', 'B')])
        inference1 = CausalInference(game1)
        dag_do_x = inference1.do("A").dag
        self.assertEqual(set(dag_do_x.nodes()), set(game1.nodes()))
        self.assertEqual(sorted(list(dag_do_x.edges())), [('A', 'B'), ('A', 'Y')])


class TestBackdoorPaths(unittest.TestCase):
    """
    These tests are drawn from games presented in The Book of Why by Judea Pearl. See the Jupyter Notebook called
    Causal Games in the examples folder for further explanation about each of these.
    """
    def test_game1(self):
        game1 = BayesianModel([('X', 'A'),
                               ('A', 'Y'),
                               ('A', 'B')])
        inference = CausalInference(model=game1)
        deconfounders = inference.get_all_backdoor_adjustment_sets(X="X", Y="Y")
        self.assertEqual(deconfounders, frozenset([]))

    def test_game2(self):
        game2 = BayesianModel([('X', 'E'),
                               ('E', 'Y'),
                               ('A', 'X'),
                               ('A', 'B'),
                               ('B', 'C'),
                               ('D', 'B'),
                               ('D', 'E')])
        inference = CausalInference(model=game2)
        deconfounders = inference.get_all_backdoor_adjustment_sets(X="X", Y="Y")
        self.assertEqual(deconfounders, frozenset([]))

    def test_game3(self):
        game3 = BayesianModel([('X', 'Y'),
                               ('X', 'A'),
                               ('B', 'A'),
                               ('B', 'Y'),
                               ('B', 'X')])
        inference = CausalInference(model=game3)
        deconfounders = inference.get_all_backdoor_adjustment_sets(X="X", Y="Y")
        self.assertEqual(deconfounders, frozenset({frozenset({'B'})}))

    def test_game4(self):
        game4 = BayesianModel([('A', 'X'),
                               ('A', 'B'),
                               ('C', 'B'),
                               ('C', 'Y')])
        inference = CausalInference(model=game4)
        deconfounders = inference.get_all_backdoor_adjustment_sets(X="X", Y="Y")
        self.assertEqual(deconfounders, frozenset([]))

    def test_game5(self):
        game5 = BayesianModel([('A', 'X'),
                               ('A', 'B'),
                               ('C', 'B'),
                               ('C', 'Y'),
                               ('X', 'Y'),
                               ('B', 'X')])
        inference = CausalInference(model=game5)
        deconfounders = inference.get_all_backdoor_adjustment_sets(X="X", Y="Y")
        self.assertEqual(deconfounders, frozenset({frozenset({'C'}),
                                                   frozenset({'A', 'B'})}))

    def test_game6(self):
        game6 = BayesianModel([('X', 'F'),
                               ('C', 'X'),
                               ('A', 'C'),
                               ('A', 'D'),
                               ('B', 'D'),
                               ('B', 'E'),
                               ('D', 'X'),
                               ('D', 'Y'),
                               ('E', 'Y'),
                               ('F', 'Y')])
        inference = CausalInference(model=game6)
        deconfounders = inference.get_all_backdoor_adjustment_sets(X="X", Y="Y")
        print(deconfounders)
        self.assertEqual(deconfounders, frozenset({frozenset({'C', 'D'}),
                                                   frozenset({'A', 'D'}),
                                                   frozenset({'D', 'E'}),
                                                   frozenset({'B', 'D'})}))
