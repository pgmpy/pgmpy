import unittest
from pgmpy.models import DiscreteMarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor

class TestToJunctionTree(unittest.TestCase):
    def test_junction_tree_conversion(self):
        mm = DiscreteMarkovNetwork()
        mm.add_nodes_from(['A', 'B', 'C'])
        mm.add_edges_from([('A', 'B'), ('B', 'C')])

        factor_ab = DiscreteFactor(['A', 'B'], [2, 2], [0.1, 0.9, 0.2, 0.8])
        factor_bc = DiscreteFactor(['B', 'C'], [2, 2], [0.3, 0.7, 0.4, 0.6])
        mm.add_factors(factor_ab, factor_bc)

        jt = mm.to_junction_tree()

        self.assertIsNotNone(jt)
        self.assertGreaterEqual(len(jt.nodes()), 1)

        all_vars = set()
        for node in jt.nodes():
            all_vars.update(node)
        self.assertTrue({'A', 'B', 'C'}.issubset(all_vars))

    def test_empty_factors(self):
        mm = DiscreteMarkovNetwork()
        mm.add_nodes_from(['X', 'Y'])
        mm.add_edges_from([('X', 'Y')])

        try:
            jt = mm.to_junction_tree()
            self.assertIsNotNone(jt)
        except Exception as e:
            self.fail(f"to_junction_tree raised an exception on empty factors: {e}")


def test_to_junction_tree_basic():
    model = DiscreteMarkovNetwork()
    model.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    jt = model.to_junction_tree()

    # Print nodes for debugging
    print("JT nodes:", jt.nodes())

    # Check that all nodes are valid cliques (sets/tuples of variables)
    cliques = list(jt.nodes())
    assert all(isinstance(clique, (tuple, set, frozenset)) for clique in cliques)

    # Check that expected cliques exist
    expected_clique = {"A", "B", "C"}
    assert any(expected_clique.issubset(clique) for clique in cliques)
