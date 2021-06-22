import unittest

import pandas as pd
import numpy as np
import networkx as nx

from pgmpy.estimators import PC
from pgmpy.models import BayesianModel
from pgmpy.independencies import Independencies
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling

# This class tests examples from: Le, Thuc, et al. "A fast PC algorithm for
# high dimensional causal discovery with multi-core PCs." IEEE/ACM transactions
# on computational biology and bioinformatics (2016).
class TestPCFakeCITest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.fake_data = pd.DataFrame(
            np.random.random((1000, 4)), columns=["A", "B", "C", "D"]
        )
        self.estimator = PC(self.fake_data)

    @staticmethod
    def fake_ci_t(X, Y, Z=[], **kwargs):
        """
        A mock CI testing function which gives False for every condition
        except for the following:
            1. B \u27C2 C
            2. B \u27C2 D
            3. C \u27C2 D
            4. A \u27C2 B | C
            5. A \u27C2 C | B
        """
        Z = list(Z)
        if X == "B":
            if Y == "C" or Y == "D":
                return True
            elif Y == "A" and Z == ["C"]:
                return True
        elif X == "C" and Y == "D" and Z == []:
            return True
        elif X == "D" and Y == "C" and Z == []:
            return True
        elif Y == "B":
            if X == "C" or X == "D":
                return True
            elif X == "A" and Z == ["C"]:
                return True
        elif X == "A" and Y == "C" and Z == ["B"]:
            return True
        elif X == "C" and Y == "A" and Z == ["B"]:
            return True
        return False

    def test_build_skeleton_orig(self):
        skel, sep_set = self.estimator.build_skeleton(
            ci_test=TestPCFakeCITest.fake_ci_t, variant="orig"
        )
        expected_edges = {("A", "C"), ("A", "D")}
        for u, v in skel.edges():
            self.assertTrue(((u, v) in expected_edges) or ((v, u) in expected_edges))

        skel, sep_set = self.estimator.build_skeleton(
            ci_test=TestPCFakeCITest.fake_ci_t, max_cond_vars=0, variant="orig"
        )
        expected_edges = {("A", "B"), ("A", "C"), ("A", "D")}
        for u, v in skel.edges():
            self.assertTrue(((u, v) in expected_edges) or ((v, u) in expected_edges))

    def test_build_skeleton_stable(self):
        skel, sep_set = self.estimator.build_skeleton(
            ci_test=TestPCFakeCITest.fake_ci_t, variant="stable"
        )
        expected_edges = {("A", "C"), ("A", "D")}
        for u, v in skel.edges():
            self.assertTrue(((u, v) in expected_edges) or ((v, u) in expected_edges))

        skel, sep_set = self.estimator.build_skeleton(
            ci_test=TestPCFakeCITest.fake_ci_t, max_cond_vars=0, variant="stable"
        )
        expected_edges = {("A", "B"), ("A", "C"), ("A", "D")}
        for u, v in skel.edges():
            self.assertTrue(((u, v) in expected_edges) or ((v, u) in expected_edges))


class TestPCEstimatorFromIndependencies(unittest.TestCase):
    def test_build_skeleton_from_ind(self):
        # Specify a set of independencies
        for variant in ["orig", "stable", "parallel"]:
            ind = Independencies(["B", "C"], ["A", ["B", "C"], "D"])
            ind = ind.closure()
            estimator = PC(independencies=ind)
            skel, sep_sets = estimator.estimate(
                variant=variant, ci_test="independence_match", return_type="skeleton"
            )

            expected_edges = {("A", "D"), ("B", "D"), ("C", "D")}
            expected_sepsets = {
                frozenset(("A", "C")): tuple(),
                frozenset(("A", "B")): tuple(),
                frozenset(("C", "B")): tuple(),
            }
            for u, v in skel.edges():
                self.assertTrue(
                    ((u, v) in expected_edges) or ((v, u) in expected_edges)
                )
            self.assertEqual(sep_sets, expected_sepsets)

            # Generate independencies from a model.
            model = BayesianModel([("A", "C"), ("B", "C"), ("B", "D"), ("C", "E")])
            estimator = PC(independencies=model.get_independencies())
            skel, sep_sets = estimator.estimate(
                variant=variant, ci_test="independence_match", return_type="skeleton"
            )

            expected_edges = model.edges()
            expected_sepsets1 = {
                frozenset(("D", "C")): ("B",),
                frozenset(("E", "B")): ("C",),
                frozenset(("A", "D")): tuple(),
                frozenset(("E", "D")): ("C",),
                frozenset(("E", "A")): ("C",),
                frozenset(("A", "B")): tuple(),
            }
            expected_sepsets2 = {
                frozenset(("D", "C")): ("B",),
                frozenset(("E", "B")): ("C",),
                frozenset(("A", "D")): tuple(),
                frozenset(("E", "D")): ("B",),
                frozenset(("E", "A")): ("C",),
                frozenset(("A", "B")): tuple(),
            }
            for u, v in skel.edges():
                self.assertTrue(
                    ((u, v) in expected_edges) or ((v, u) in expected_edges)
                )

            self.assertTrue(
                (sep_sets == expected_sepsets1) or (sep_sets == expected_sepsets2)
            )

    def test_skeleton_to_pdag(self):
        skel = nx.Graph([("A", "D"), ("A", "C"), ("B", "C")])
        sep_sets = {
            frozenset({"D", "C"}): ("A",),
            frozenset({"A", "B"}): tuple(),
            frozenset({"D", "B"}): ("A",),
        }
        pdag = PC.skeleton_to_pdag(skeleton=skel, separating_sets=sep_sets)
        self.assertSetEqual(
            set(pdag.edges()), set([("B", "C"), ("A", "D"), ("A", "C"), ("D", "A")])
        )

        skel = nx.Graph([("A", "B"), ("A", "C")])
        sep_sets = {frozenset({"B", "C"}): ()}
        self.assertSetEqual(
            set(PC.skeleton_to_pdag(skeleton=skel, separating_sets=sep_sets).edges()),
            set([("B", "A"), ("C", "A")]),
        )

        sep_sets = {frozenset({"B", "C"}): ("A",)}
        pdag = PC.skeleton_to_pdag(skeleton=skel, separating_sets=sep_sets)
        self.assertSetEqual(
            set(PC.skeleton_to_pdag(skel, sep_sets).edges()),
            set([("A", "B"), ("B", "A"), ("A", "C"), ("C", "A")]),
        )

        skel = nx.Graph([("A", "C"), ("B", "C"), ("C", "D")])
        sep_sets = {
            frozenset({"A", "B"}): tuple(),
            frozenset({"A", "D"}): ("C",),
            frozenset({"B", "D"}): ("C",),
        }
        pdag = PC.skeleton_to_pdag(skeleton=skel, separating_sets=sep_sets)
        self.assertSetEqual(
            set(pdag.edges()), set([("A", "C"), ("B", "C"), ("C", "D")])
        )

        skel = nx.Graph([("A", "B"), ("A", "C"), ("B", "C"), ("B", "D")])
        sep_sets = {frozenset({"A", "D"}): tuple(), frozenset({"C", "D"}): ("A", "B")}
        pdag = PC.skeleton_to_pdag(skeleton=skel, separating_sets=sep_sets)
        self.assertSetEqual(
            set(pdag.edges()), set([("A", "B"), ("B", "C"), ("A", "C"), ("D", "B")])
        )

        skel = nx.Graph([("A", "B"), ("B", "C"), ("A", "D"), ("B", "D"), ("C", "D")])
        sep_sets = {frozenset({"A", "C"}): ("B",)}
        pdag = PC.skeleton_to_pdag(skeleton=skel, separating_sets=sep_sets)
        self.assertSetEqual(
            set(pdag.edges()),
            set(
                [
                    ("A", "B"),
                    ("B", "A"),
                    ("B", "C"),
                    ("C", "B"),
                    ("A", "D"),
                    ("B", "D"),
                    ("C", "D"),
                ]
            ),
        )

    def test_estimate_dag(self):
        for variant in ["orig", "stable", "parallel"]:
            ind = Independencies(["B", "C"], ["A", ["B", "C"], "D"])
            ind = ind.closure()
            estimator = PC(independencies=ind)
            model = estimator.estimate(
                variant="orig", ci_test="independence_match", return_type="dag"
            )
            expected_edges = {("B", "D"), ("A", "D"), ("C", "D")}
            self.assertEqual(model.edges(), expected_edges)

            model = BayesianModel([("A", "C"), ("B", "C"), ("B", "D"), ("C", "E")])
            estimator = PC(independencies=model.get_independencies())
            estimated_model = estimator.estimate(
                variant="orig", ci_test="independence_match", return_type="dag"
            )
            expected_edges_1 = set(model.edges())
            expected_edges_2 = {("B", "C"), ("A", "C"), ("C", "E"), ("D", "B")}
            self.assertTrue(
                (set(estimated_model.edges()) == expected_edges_1)
                or (set(estimated_model.edges()) == expected_edges_2)
            )


class TestPCEstimatorFromDiscreteData(unittest.TestCase):
    def test_build_skeleton(self):
        for variant in ["orig", "stable", "parallel"]:
            # Fake dataset no: 1
            np.random.seed(42)
            data = pd.DataFrame(
                np.random.randint(0, 2, size=(10000, 5)), columns=list("ABCDE")
            )
            data["F"] = data["A"] + data["B"] + data["C"]
            est = PC(data=data)
            skel, sep_sets = est.estimate(
                variant=variant,
                ci_test="chi_square",
                return_type="skeleton",
                significance_level=0.005,
            )
            expected_edges = {("A", "F"), ("B", "F"), ("C", "F")}
            expected_sepsets = {
                frozenset(("D", "F")): tuple(),
                frozenset(("D", "B")): tuple(),
                frozenset(("A", "C")): tuple(),
                frozenset(("D", "E")): tuple(),
                frozenset(("E", "F")): tuple(),
                frozenset(("E", "C")): tuple(),
                frozenset(("E", "B")): tuple(),
                frozenset(("D", "C")): tuple(),
                frozenset(("A", "B")): tuple(),
                frozenset(("A", "E")): tuple(),
                frozenset(("B", "C")): tuple(),
                frozenset(("A", "D")): tuple(),
            }
            for u, v in skel.edges():
                self.assertTrue(
                    ((u, v) in expected_edges) or ((v, u) in expected_edges)
                )
            self.assertEqual(sep_sets, expected_sepsets)

            # Fake dataset no: 2 Expected structure X <- Z -> Y
            def fake_ci(X, Y, Z=tuple(), **kwargs):
                if X == "X" and Y == "Y" and Z == ("Z",):
                    return True
                elif X == "Y" and Y == "X" and Z == ("Z",):
                    return True
                else:
                    return False

            np.random.seed(42)
            fake_data = pd.DataFrame(
                np.random.randint(low=0, high=2, size=(10000, 3)),
                columns=["X", "Y", "Z"],
            )
            est = PC(data=fake_data)
            skel, sep_sets = est.estimate(
                variant=variant, ci_test=fake_ci, return_type="skeleton"
            )
            expected_edges = {("X", "Z"), ("Y", "Z")}
            expected_sepsets = {frozenset(("X", "Y")): ("Z",)}
            for u, v in skel.edges():
                self.assertTrue(
                    ((u, v) in expected_edges) or ((v, u) in expected_edges)
                )
            self.assertEqual(sep_sets, expected_sepsets)

    def test_build_dag(self):
        for variant in ["orig", "stable", "parallel"]:
            np.random.seed(42)
            data = pd.DataFrame(
                np.random.randint(0, 3, size=(10000, 3)), columns=list("XYZ")
            )
            data["sum"] = data.sum(axis=1)
            est = PC(data=data)
            dag = est.estimate(
                variant=variant,
                ci_test="chi_square",
                return_type="dag",
                significance_level=0.001,
            )
            expected_edges = {("Z", "sum"), ("X", "sum"), ("Y", "sum")}
            self.assertEqual(set(dag.edges()), expected_edges)


class TestPCEstimatorFromContinuousData(unittest.TestCase):
    def test_build_skeleton(self):
        for variant in ["orig", "stable", "parallel"]:
            # Fake dataset no: 1
            np.random.seed(42)
            data = pd.DataFrame(np.random.randn(10000, 5), columns=list("ABCDE"))
            data["F"] = data["A"] + data["B"] + data["C"]
            est = PC(data=data)
            skel, sep_sets = est.estimate(
                variant=variant, ci_test="pearsonr", return_type="skeleton"
            )
            expected_edges = {("A", "F"), ("B", "F"), ("C", "F")}
            expected_edges_stable = {("A", "F"), ("B", "C"), ("B", "F"), ("C", "F")}
            expected_sepsets = {
                frozenset(("D", "F")): tuple(),
                frozenset(("D", "B")): tuple(),
                frozenset(("A", "C")): tuple(),
                frozenset(("D", "E")): tuple(),
                frozenset(("E", "F")): tuple(),
                frozenset(("E", "C")): tuple(),
                frozenset(("E", "B")): tuple(),
                frozenset(("D", "C")): tuple(),
                frozenset(("A", "B")): tuple(),
                frozenset(("A", "E")): tuple(),
                frozenset(("B", "C")): tuple(),
                frozenset(("A", "D")): tuple(),
                # This one is only for stable version.
                frozenset(("C", "B")): tuple(),
            }
            for u, v in skel.edges():
                self.assertTrue(
                    ((u, v) in expected_edges_stable)
                    or ((v, u) in expected_edges_stable)
                )

            for key, value in sep_sets.items():
                self.assertEqual(sep_sets[key], expected_sepsets[key])

            # Fake dataset no: 2. Expected model structure X <- Z -> Y
            def fake_ci(X, Y, Z=tuple(), **kwargs):
                if X == "X" and Y == "Y" and Z == ("Z",):
                    return True
                elif X == "Y" and Y == "X" and Z == ("Z",):
                    return True
                else:
                    return False

            np.random.seed(42)
            data = pd.DataFrame(np.random.randn(10000, 3), columns=list("XYZ"))
            est = PC(data=data)
            skel, sep_sets = est.estimate(
                variant=variant, ci_test=fake_ci, return_type="skeleton"
            )
            expected_edges = {("X", "Z"), ("Y", "Z")}
            expected_sepsets = {frozenset(("X", "Y")): ("Z",)}

            for u, v in skel.edges():
                self.assertTrue(
                    ((u, v) in expected_edges) or ((v, u) in expected_edges)
                )
            self.assertEqual(sep_sets, expected_sepsets)

    def test_build_dag(self):
        for variant in ["orig", "stable", "parallel"]:
            np.random.seed(42)
            data = pd.DataFrame(np.random.randn(10000, 3), columns=list("XYZ"))
            data["sum"] = data.sum(axis=1)
            est = PC(data=data)
            dag = est.estimate(variant=variant, ci_test="pearsonr", return_type="dag")

            expected_edges = {("Z", "sum"), ("X", "sum"), ("Y", "sum")}
            self.assertEqual(set(dag.edges()), expected_edges)


class TestPCRealModels(unittest.TestCase):
    def test_pc_alarm(self):
        alarm_model = get_example_model("alarm")
        data = BayesianModelSampling(alarm_model).forward_sample(size=int(1e3), seed=42)
        est = PC(data)
        dag = est.estimate(variant="stable", max_cond_vars=1)

    def test_pc_asia(self):
        asia_model = get_example_model("asia")
        data = BayesianModelSampling(asia_model).forward_sample(size=int(1e3), seed=42)
        est = PC(data)
        dag = est.estimate(variant="stable", max_cond_vars=1)
