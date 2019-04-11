import unittest

from pgmpy.models.CausalGraph import CausalGraph


class TestCausalGraphEssentials(unittest.TestCase):
    def test_null_graph(self):
        self.assertTrue(isinstance(CausalGraph(), CausalGraph))

    def test_graph_creation(self):
        game1 = CausalGraph([('X', 'A'),
                             ('A', 'Y'),
                             ('A', 'B')])
        self.assertTrue(isinstance(game1, CausalGraph))   


class TestCausalGraphMethods(unittest.TestCase):

    def test_display_methods(self):
        game1 = CausalGraph([('X', 'A'),
                             ('A', 'Y'),
                             ('A', 'B')])
        self.assertEqual("CausalGraph(A, B, X, Y)", game1.__repr__())
        game1.get_distribution()

    def test_do_operator(self):
        game1 = CausalGraph([('X', 'A'),
                             ('A', 'Y'),
                             ('A', 'B')])
        dag_do_x = game1.do("A")
        self.assertEqual(set(dag_do_x.nodes()), set(game1.nodes()))
        self.assertEqual(sorted(list(dag_do_x.edges())), [('A', 'B'), ('A', 'Y')])

    def test_is_d_separated(self):
        game1 = CausalGraph([('X', 'A'),
                             ('A', 'Y'),
                             ('A', 'B')])
        self.assertFalse(game1._is_d_separated("X", "Y", Z=None))
        self.assertTrue(game1._is_d_separated("X", "Y", Z='A'))

    
class TestBackdoorPaths(unittest.TestCase):
    """
    These tests are drawn from games presented in The Book of Why by Judea Pearl. See the Jupyter Notebook called
    Causal Games in the examples folder for further explanation about each of these.
    """
    def test_game1(self):
        game1 = CausalGraph([('X', 'A'),
                             ('A', 'Y'),
                             ('A', 'B')])
        self.assertTrue(game1.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = game1.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(deconfounders, frozenset())

    def test_game2(self):
        game2 = CausalGraph([('X', 'E'),
                             ('E', 'Y'),
                             ('A', 'B'),
                             ('A', 'X'),
                             ('B', 'C'),
                             ('D', 'B'),
                             ('D', 'E')])
        self.assertTrue(game2.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = game2.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(deconfounders, frozenset())

    def test_game3(self):
        game3 = CausalGraph([('X', 'Y'),
                             ('X', 'A'),
                             ('B', 'A'),
                             ('B', 'Y'),
                             ('B', 'X')])
        self.assertFalse(game3.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = game3.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(deconfounders, frozenset({frozenset({'B'})}))

    def test_game4(self):
        game4 = CausalGraph([('A', 'X'),
                             ('A', 'B'),
                             ('C', 'B'),
                             ('C', 'Y')])
        self.assertTrue(game4.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = game4.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(deconfounders, frozenset())

    def test_game5(self):
        game5 = CausalGraph([('A', 'X'),
                             ('A', 'B'),
                             ('C', 'B'),
                             ('C', 'Y'),
                             ('X', 'Y'),
                             ('B', 'X')])
        self.assertFalse(game5.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = game5.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(deconfounders, frozenset({frozenset({'C'}),
                                                   frozenset({'A', 'B'})}))

    def test_game6(self):
        game6 = CausalGraph([('X', 'F'),
                             ('C', 'X'),
                             ('A', 'C'),
                             ('A', 'D'),
                             ('B', 'D'),
                             ('B', 'E'),
                             ('D', 'X'),
                             ('D', 'Y'),
                             ('E', 'Y'),
                             ('F', 'Y')])
        self.assertFalse(game6.is_valid_backdoor_adjustment_set("X", "Y"))
        deconfounders = game6.get_all_backdoor_adjustment_sets("X", "Y")
        self.assertEqual(deconfounders, frozenset({frozenset({'C', 'D'}),
                                                   frozenset({'A', 'D'}),
                                                   frozenset({'D', 'E'}),
                                                   frozenset({'B', 'D'})}))


class TestFrontdoorPaths(unittest.TestCase):
    """
    These tests are drawn from games presented in The Book of Why by Judea Pearl. See the Jupyter Notebook called
    Causal Games in the examples folder for further explanation about each of these.
    """
    def test_game1(self):
        game1 = CausalGraph([('X', 'A'),
                             ('A', 'Y'),
                             ('A', 'B')])
        deconfounders = game1.get_all_frontdoor_adjustment_sets("X", "Y")
        self.assertEqual(deconfounders, frozenset({frozenset(['A'])}))

    def test_game4(self):
        game4 = CausalGraph([('A', 'X'),
                             ('A', 'B'),
                             ('C', 'B'),
                             ('C', 'Y')])
        deconfounders = game4.get_all_frontdoor_adjustment_sets("X", "Y")
        self.assertEqual(deconfounders, frozenset())

    def test_game6(self):
        game6 = CausalGraph([('X', 'F'),
                             ('C', 'X'),
                             ('A', 'C'),
                             ('A', 'D'),
                             ('B', 'D'),
                             ('B', 'E'),
                             ('D', 'X'),
                             ('D', 'Y'),
                             ('E', 'Y'),
                             ('F', 'Y')])
        adjustment_sets = game6.get_all_frontdoor_adjustment_sets("X", "Y")
        self.assertEqual(adjustment_sets, frozenset({frozenset({'F'})}))

    def test_game7(self):
        game7 = CausalGraph([('X', 'A'),
                             ('A', 'Y'),
                             ('B', 'X'),
                             ('B', 'Y')])
        adjustment_sets = game7.get_all_frontdoor_adjustment_sets("X", "Y")
        self.assertEqual(adjustment_sets, frozenset({frozenset({'A'})}))
