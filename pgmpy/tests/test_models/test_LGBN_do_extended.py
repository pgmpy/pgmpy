import unittest

from numpy.testing import assert_allclose

from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


class TestLGBNDo(unittest.TestCase):
    """Tests for LinearGaussianBayesianNetwork.do()."""

    def setUp(self):
        """Chain  X --0.5--> Y ---(-1)--> Z  with intercepts 0, 2, 1."""
        self.model = LinearGaussianBayesianNetwork([("X", "Y"), ("Y", "Z")])
        self.model.add_cpds(
            LinearGaussianCPD("X", [0], 1),
            LinearGaussianCPD("Y", [2, 0.5], 1, ["X"]),
            LinearGaussianCPD("Z", [1, -1], 1, ["Y"]),
        )

    # ------------------------------------------------------------------ #
    #  A. Single parent intervention  (X -> Y)                            #
    # ------------------------------------------------------------------ #
    def test_basic_intervention_intercept_shift(self):
        """do(X=3):  Y's intercept shifts  2 + 0.5*3 = 3.5."""
        m = self.model.do("X", values={"X": 3.0})

        # Y becomes a root node with updated intercept.
        cpd_y = m.get_cpds("Y")
        assert_allclose(cpd_y.beta, [3.5], atol=1e-12)
        self.assertEqual(cpd_y.evidence, [])
        self.assertEqual(cpd_y.std, 1)

        # X has become a point-mass at 3.
        cpd_x = m.get_cpds("X")
        assert_allclose(cpd_x.beta, [3.0], atol=1e-12)
        self.assertEqual(cpd_x.std, 0)

        # Z (not a direct child of X) is unchanged.
        cpd_z = m.get_cpds("Z")
        assert_allclose(cpd_z.beta, [1, -1], atol=1e-12)
        self.assertEqual(cpd_z.evidence, ["Y"])

    # ------------------------------------------------------------------ #
    #  B. Multiple parents, both intervened  ({X, W} -> Y)                #
    # ------------------------------------------------------------------ #
    def test_multiple_parents_both_intervened(self):
        """Y = 1 + 2X + 3W.  do(X=1, W=2) → intercept = 1 + 2*1 + 3*2 = 9."""
        model = LinearGaussianBayesianNetwork([("X", "Y"), ("W", "Y")])
        model.add_cpds(
            LinearGaussianCPD("X", [0], 1),
            LinearGaussianCPD("W", [0], 1),
            LinearGaussianCPD("Y", [1, 2, 3], 1, ["X", "W"]),
        )

        m = model.do(["X", "W"], values={"X": 1.0, "W": 2.0})

        cpd_y = m.get_cpds("Y")
        assert_allclose(cpd_y.beta, [9.0], atol=1e-12)
        self.assertEqual(cpd_y.evidence, [])
        self.assertEqual(cpd_y.std, 1)

        # Both intervened nodes are point-masses with 0 variance.
        for node, val in [("X", 1.0), ("W", 2.0)]:
            cpd = m.get_cpds(node)
            assert_allclose(cpd.beta, [val], atol=1e-12)
            self.assertEqual(cpd.std, 0)

    # ------------------------------------------------------------------ #
    #  B′. Partial intervention — one of two parents                      #
    # ------------------------------------------------------------------ #
    def test_partial_intervention_one_parent(self):
        """Y = 1 + 2X + 3W.  do(X=4) only → Y = (1+8) + 3W = 9 + 3W."""
        model = LinearGaussianBayesianNetwork([("X", "Y"), ("W", "Y")])
        model.add_cpds(
            LinearGaussianCPD("X", [0], 1),
            LinearGaussianCPD("W", [0], 1),
            LinearGaussianCPD("Y", [1, 2, 3], 1, ["X", "W"]),
        )

        m = model.do("X", values={"X": 4.0})

        cpd_y = m.get_cpds("Y")
        assert_allclose(cpd_y.beta, [9.0, 3.0], atol=1e-12)
        self.assertEqual(cpd_y.evidence, ["W"])

        # W → Y edge still present; X → Y edge severed.
        self.assertIn(("W", "Y"), list(m.edges()))
        self.assertNotIn(("X", "Y"), list(m.edges()))

    # ------------------------------------------------------------------ #
    #  C. Intervening on a root node (no parents)                         #
    # ------------------------------------------------------------------ #
    def test_intervention_on_root_node(self):
        """do(X=5):  X already has no parents → CPD becomes point-mass at 5."""
        m = self.model.do("X", values={"X": 5.0})

        cpd_x = m.get_cpds("X")
        assert_allclose(cpd_x.beta, [5.0], atol=1e-12)
        self.assertEqual(cpd_x.std, 0)

        # Y's intercept: 2 + 0.5*5 = 4.5
        cpd_y = m.get_cpds("Y")
        assert_allclose(cpd_y.beta, [4.5], atol=1e-12)

    # ------------------------------------------------------------------ #
    #  C′. Intervening on a leaf node (no children)                       #
    # ------------------------------------------------------------------ #
    def test_intervention_on_leaf_node(self):
        """do(Z=7):  Z has no children → only Z's CPD changes."""
        m = self.model.do("Z", values={"Z": 7.0})

        cpd_z = m.get_cpds("Z")
        assert_allclose(cpd_z.beta, [7.0], atol=1e-12)
        self.assertEqual(cpd_z.std, 0)
        self.assertEqual(cpd_z.evidence, [])

        # Y and X CPDs untouched.
        cpd_y = m.get_cpds("Y")
        assert_allclose(cpd_y.beta, [2, 0.5], atol=1e-12)
        self.assertEqual(cpd_y.evidence, ["X"])

    # ------------------------------------------------------------------ #
    #  D. inplace=True modifies the original object                       #
    # ------------------------------------------------------------------ #
    def test_inplace_true_modifies_original(self):
        """inplace=True must mutate the caller and return the same object."""
        result = self.model.do("X", values={"X": 3.0}, inplace=True)

        self.assertIs(result, self.model)

        cpd_x = self.model.get_cpds("X")
        assert_allclose(cpd_x.beta, [3.0], atol=1e-12)
        self.assertEqual(cpd_x.std, 0)

        cpd_y = self.model.get_cpds("Y")
        assert_allclose(cpd_y.beta, [3.5], atol=1e-12)

    def test_inplace_false_leaves_original_unchanged(self):
        """Default (inplace=False) must not touch the original model."""
        orig_edges = set(self.model.edges())
        orig_y_beta = list(self.model.get_cpds("Y").beta)

        m_do = self.model.do("X", values={"X": 3.0})

        # Original model unchanged.
        self.assertEqual(set(self.model.edges()), orig_edges)
        assert_allclose(self.model.get_cpds("Y").beta, orig_y_beta, atol=1e-12)

        # New model has the shift.
        assert_allclose(m_do.get_cpds("Y").beta, [3.5], atol=1e-12)

    # ------------------------------------------------------------------ #
    #  E. Structural integrity: check_model() after do()                  #
    # ------------------------------------------------------------------ #
    def test_check_model_passes_after_do_with_values(self):
        """check_model() must pass on the mutilated model."""
        m = self.model.do("X", values={"X": 3.0})
        self.assertTrue(m.check_model())

    def test_check_model_passes_after_do_without_values(self):
        """Graph-surgery-only do() also leaves a valid model."""
        m = self.model.do("X")
        self.assertTrue(m.check_model())

    # ------------------------------------------------------------------ #
    #  F. do() without values (graph surgery only, equiv. to do(X=0))     #
    # ------------------------------------------------------------------ #
    def test_do_without_values(self):
        """No values → X set to 0, no intercept shift (≡ do(X=0))."""
        m = self.model.do("X")

        cpd_x = m.get_cpds("X")
        assert_allclose(cpd_x.beta, [0], atol=1e-12)
        self.assertEqual(cpd_x.std, 0)

        # Y's beta_0 stays at 2 (shift is 0.5*0 = 0).
        cpd_y = m.get_cpds("Y")
        assert_allclose(cpd_y.beta, [2], atol=1e-12)
        self.assertEqual(cpd_y.evidence, [])

    # ------------------------------------------------------------------ #
    #  G. Invalid node raises ValueError                                  #
    # ------------------------------------------------------------------ #
    def test_do_invalid_node_raises(self):
        with self.assertRaises(ValueError):
            self.model.do("NONEXISTENT", values={"NONEXISTENT": 1.0})

    # ------------------------------------------------------------------ #
    #  H. Non-intervened structure preserved                              #
    # ------------------------------------------------------------------ #
    def test_do_preserves_non_intervened_structure(self):
        """Edges not involving the intervened node stay intact."""
        m = self.model.do("X", values={"X": 3.0})

        self.assertIn(("Y", "Z"), list(m.edges()))
        self.assertNotIn(("X", "Y"), list(m.edges()))

        cpd_z = m.get_cpds("Z")
        assert_allclose(cpd_z.beta, [1, -1], atol=1e-12)
        self.assertEqual(cpd_z.evidence, ["Y"])

    # ------------------------------------------------------------------ #
    #  I. Intermediate-node intervention                                  #
    # ------------------------------------------------------------------ #
    def test_do_on_intermediate_node(self):
        """do(Y=10):  X→Y severed, Z intercept = 1 + (-1)*10 = −9."""
        m = self.model.do("Y", values={"Y": 10.0})

        cpd_y = m.get_cpds("Y")
        assert_allclose(cpd_y.beta, [10.0], atol=1e-12)
        self.assertEqual(cpd_y.std, 0)
        self.assertEqual(list(m.get_parents("Y")), [])

        cpd_z = m.get_cpds("Z")
        assert_allclose(cpd_z.beta, [-9.0], atol=1e-12)
        self.assertEqual(cpd_z.evidence, [])

        # X stays untouched.
        cpd_x = m.get_cpds("X")
        assert_allclose(cpd_x.beta, [0], atol=1e-12)
        self.assertEqual(cpd_x.std, 1)

        self.assertTrue(m.check_model())

    # ------------------------------------------------------------------ #
    #  J. Integration: simulate(do=…) ≡ do() + simulate()                #
    # ------------------------------------------------------------------ #
    def test_simulate_matches_do_then_joint(self):
        """simulate(do=…) must yield same moments as do() + joint Gaussian."""
        # Compute expected moments via do() → to_joint_gaussian().
        m_do = self.model.do("X", values={"X": 2.0})
        m_do.remove_node("X")
        mean_expected, _ = m_do.to_joint_gaussian()

        # Sample via simulate(do=...).
        df = self.model.simulate(n_samples=80_000, do={"X": 2.0}, seed=42)
        sample_means = df[["Y", "Z"]].mean().values

        assert_allclose(sample_means, mean_expected, atol=0.05)

    # ------------------------------------------------------------------ #
    #  K. Intervened nodes always have zero variance                      #
    # ------------------------------------------------------------------ #
    def test_intervened_node_has_zero_variance(self):
        """Every intervened node's std must be exactly 0."""
        m = self.model.do(["X", "Y"], values={"X": 1.0, "Y": 2.0})

        for node in ["X", "Y"]:
            cpd = m.get_cpds(node)
            self.assertEqual(cpd.std, 0, f"{node} std should be 0")

        # Uninverted node retains original std.
        self.assertEqual(m.get_cpds("Z").std, 1)


if __name__ == "__main__":
    unittest.main()
