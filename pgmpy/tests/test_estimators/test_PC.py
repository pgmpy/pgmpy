import unittest

import pandas as pd
import numpy as np
import networkx as nx

from pgmpy.estimators import PC
from pgmpy.models import BayesianModel
from pgmpy.independencies import Independencies


# This class tests examples from: Le, Thuc, et al. "A fast PC algorithm for
# high dimensional causal discovery with multi-core PCs." IEEE/ACM transactions
# on computational biology and bioinformatics (2016).
class TestPCFakeCITest(unittest.TestCase):
    def setUp(self):
        self.fake_data = pd.DataFrame(
            np.random.random((1000, 4)), columns=["A", "B", "C", "D"]
        )
        self.estimator = PC(self.fake_data)

    @staticmethod
    def fake_ci_t(X, Y, Z=[], **kwargs):
        """
        A mock CI testing function which gives False for every condition
        except for the following:
            1. B _|_ C
            2. B _|_ D
            3. C _|_ D
            4. A _|_ B | C
            5. A _|_ C | B
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
        expected_edges = {("A", "D")}
        for u, v in skel.edges():
            self.assertTrue(((u, v) in expected_edges) or ((v, u) in expected_edges))

        skel, sep_set = self.estimator.build_skeleton(
            ci_test=TestPCFakeCITest.fake_ci_t, max_cond_vars=0, variant="stable"
        )
        expected_edges = {("A", "B"), ("A", "C"), ("A", "D")}
        for u, v in skel.edges():
            self.assertTrue(((u, v) in expected_edges) or ((v, u) in expected_edges))


class TestPCEstimatorFromIndependencies(unittest.TestCase):
    def test_build_skeleton(self):
        # Specify a set of independencies
        ind = Independencies(["B", "C"], ["A", ["B", "C"], "D"])
        ind = ind.closure()
        estimator = PC(independencies=ind)
        skel, sep_sets = estimator.estimate(
            variant="orig", ci_test="independence_match", return_type="skeleton"
        )

        expected_edges = {("A", "D"), ("B", "D"), ("C", "D")}
        expected_sepsets = {frozenset(("A", "C")): tuple(),
                            frozenset(("A", "B")): tuple(),
                            frozenset(("C", "B")): tuple()}
        for u, v in skel.edges():
            self.assertTrue(((u, v) in expected_edges) or ((v, u) in expected_edges))
        self.assertEqual(sep_sets, expected_sepsets)

        # Generate independencies from a model.
        model = BayesianModel([("A", "C"), ("B", "C"), ("B", "D"), ("C", "E")])
        estimator = PC(independencies=model.get_independencies())
        skel, sep_sets = estimator.estimate(
            variant="orig", ci_test="independence_match", return_type="skeleton"
        )

        expected_edges = model.edges()
        expected_sepsets = {
            frozenset(("D", "C")): ("B", ),
            frozenset(("E", "B")): ("C", ),
            frozenset(("A", "D")): tuple(),
            frozenset(("E", "D")): ("C", ),
            frozenset(("E", "A")): ("C", ),
            frozenset(("A", "B")): tuple(),
        }
        for u, v in skel.edges():
            self.assertTrue(((u, v) in expected_edges) or ((v, u) in expected_edges))
        self.assertEqual(sep_sets, expected_sepsets)

    def test_skeleton_to_pdag(self):
        data = pd.DataFrame(
            np.random.randint(0, 3, size=(1000, 3)), columns=list("ABD")
        )
        data["C"] = data["A"] - data["B"]
        data["D"] += data["A"]
        estimator = PC(data=data)
        skel, sep_set = estimator.estimate(
            variant="orig", ci_test="chi_square", return_type="skeleton"
        )

        pdag = PC.skeleton_to_pdag(skeleton=skel, separating_sets=sep_set)
        self.assertSetEqual(
            set(pdag.edges()), set([("B", "C"), ("A", "D"), ("A", "C"), ("D", "A")])
        )

        skel = nx.UndirectedGraph([("A", "B"), ("A", "C")])
        sep_sets1 = {frozenset({"B", "C"}): ()}
        self.assertSetEqual(
            set(PC.skeleton_to_pdag(skeleton=skel, separating_set=sep_sets1).edges()),
            set([("B", "A"), ("C", "A")]),
        )

        sep_sets2 = {frozenset({"B", "C"}): ("A",)}
        pdag2 = PC.skeleton_to_pdag(skeleton=skel, separating_set=sep_sets2)
        self.assertSetEqual(
            set(c.skeleton_to_pdag(skel, sep_sets2).edges()),
            set([("A", "B"), ("B", "A"), ("A", "C"), ("C", "A")]),
        )

    def test_estimate_dag(self):
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
        # Fake dataset no: 1
        data = pd.DataFrame(
            np.random.randint(0, 2, size=(1000, 5)), columns=list("ABCDE")
        )
        data["F"] = data["A"] + data["B"] + data["C"]
        est = PC(data=data)
        skel, sep_sets = est.estimate(ci_test="chi_square", return_type="skeleton")
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
            self.assertTrue(((u, v) in expected_edges) or ((v, u) in expected_edges))
        self.assertEqual(sep_sets, expected_sepsets)

        # Fake dataset no: 2
        data = pd.DataFrame(
            np.random.randint(0, 2, size=(1000, 3)), columns=list("XYZ")
        )
        data["X"] += data["Z"]
        data["Y"] += data["Z"]
        est = PC(data=data)
        skel, sep_sets = est.estimate(
            variant="orig", ci_test="chi_square", return_type="skeleton"
        )
        expected_edges = {("X", "Z"), ("Y", "Z")}
        expected_sepsets = {frozenset(("X", "Y")): ("Z",)}

        for u, v in skel.edges():
            self.assertTrue(((u, v) in expected_edges) or ((v, u) in expected_edges))
        self.assertEqual(sep_sets, expected_sepsets)

    def test_build_dag(self):
        data = pd.DataFrame(
            np.random.randint(0, 3, size=(1000, 3)), columns=list("XYZ")
        )
        data["sum"] = data.sum(axis=1)
        est = PC(data=data)
        dag = est.estimate(variant="orig", ci_test="chi_square", return_type="dag")

        expected_edges = {("Z", "sum"), ("X", "sum"), ("Y", "sum")}
        self.assertEqual(set(dag.edges()), expected_edges)


class TestPCEstimatorFromContinuousData(unittest.TestCase):
    def test_build_skeleton(self):
        # Fake dataset no: 1
        data = pd.DataFrame(np.random.randn(1000, 5), columns=list("ABCDE"))
        data["F"] = data["A"] + data["B"] + data["C"]
        est = PC(data=data)
        skel, sep_sets = est.estimate(
            variant="orig", ci_test="pearsonr", return_type="skeleton"
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
            self.assertTrue(((u, v) in expected_edges) or ((v, u) in expected_edges))
        self.assertEqual(sep_sets, expected_sepsets)

        # Fake dataset no: 2
        data = pd.DataFrame(np.random.randn(1000, 3), columns=list("XYZ"))
        data["X"] += data["Z"]
        data["Y"] += data["Z"]
        est = PC(data=data)
        skel, sep_sets = est.estimate_skeleton(
            variant="orig", ci_test="pearsonr", return_type="skeleton"
        )
        expected_edges = {("X", "Z"), ("Y", "Z")}
        expected_sepsets = {frozenset(("X", "Y")): ("Z",)}

        for u, v in skel.edges():
            self.assertTrue(((u, v) in expected_edges) or ((v, u) in expected_edges))
        self.assertEqual(sep_sets, expected_sepsets)

    def test_build_dag(self):
        data = pd.DataFrame(np.random.randn(1000, 3), columns=list("XYZ"))
        data["sum"] = data.sum(axis=1)
        est = PC(data=data)
        dag = est.estimate(variant="orig", ci_test="pearsonr", return_type="dag")

        expected_edges = {("Z", "sum"), ("X", "sum"), ("Y", "sum")}
        self.assertEqual(set(dag.edges()), expected_edges)
