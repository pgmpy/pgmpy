import unittest

import networkx as nx
import numpy as np
import pandas as pd
from joblib.externals.loky import get_reusable_executor

from pgmpy.base import PDAG
from pgmpy.estimators import PC, ExpertKnowledge
from pgmpy.independencies import Independencies
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.utils import get_example_model


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
            1. B \u27c2 C
            2. B \u27c2 D
            3. C \u27c2 D
            4. A \u27c2 B | C
            5. A \u27c2 C | B
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
            ci_test=TestPCFakeCITest.fake_ci_t,
            max_cond_vars=0,
            variant="orig",
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
            ci_test=TestPCFakeCITest.fake_ci_t,
            max_cond_vars=0,
            variant="stable",
        )
        expected_edges = {("A", "B"), ("A", "C"), ("A", "D")}
        for u, v in skel.edges():
            self.assertTrue(((u, v) in expected_edges) or ((v, u) in expected_edges))


class TestPCEstimatorFromIndependences(unittest.TestCase):
    def test_build_skeleton_from_ind(self):
        # Specify a set of independencies
        for variant in ["orig", "stable", "parallel"]:
            ind = Independencies(["B", "C"], ["A", ["B", "C"], "D"])
            ind = ind.closure()
            estimator = PC(independencies=ind)
            skel, sep_sets = estimator.estimate(
                variant=variant,
                ci_test="independence_match",
                return_type="skeleton",
                n_jobs=2,
                show_progress=False,
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
            model = DiscreteBayesianNetwork(
                [("A", "C"), ("B", "C"), ("B", "D"), ("C", "E")]
            )
            estimator = PC(independencies=model.get_independencies())
            skel, sep_sets = estimator.estimate(
                variant=variant,
                ci_test="independence_match",
                return_type="skeleton",
                n_jobs=2,
                show_progress=False,
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
        # D - A - C - B  ==> D - A -> C <- B
        skel = nx.Graph([("A", "D"), ("A", "C"), ("B", "C")])
        sep_sets = {
            frozenset({"D", "C"}): ("A",),
            frozenset({"A", "B"}): tuple(),
            frozenset({"D", "B"}): ("A",),
        }
        pdag = PC.orient_colliders(skel, sep_sets)
        pdag = PC.apply_orientation_rules(pdag)
        self.assertSetEqual(
            set(pdag.edges()), set([("B", "C"), ("A", "D"), ("A", "C"), ("D", "A")])
        )

        # C - A - B  ==> C -> A <- B
        skel = nx.Graph([("A", "B"), ("A", "C")])
        sep_sets = {frozenset({"B", "C"}): ()}
        pdag = PC.orient_colliders(skeleton=skel, separating_sets=sep_sets)
        pdag = PC.apply_orientation_rules(pdag)
        self.assertSetEqual(
            set(pdag.edges()),
            set([("B", "A"), ("C", "A")]),
        )

        # C - A - B ==> C - A - B
        sep_sets = {frozenset({"B", "C"}): ("A",)}
        pdag = PC.orient_colliders(skeleton=skel, separating_sets=sep_sets)
        pdag = PC.apply_orientation_rules(pdag)
        self.assertSetEqual(
            set(pdag.edges()),
            set([("A", "B"), ("B", "A"), ("A", "C"), ("C", "A")]),
        )

        # {A, B} - C - D ==> {A, B} -> C -> D
        skel = nx.Graph([("A", "C"), ("B", "C"), ("C", "D")])
        sep_sets = {
            frozenset({"A", "B"}): tuple(),
            frozenset({"A", "D"}): ("C",),
            frozenset({"B", "D"}): ("C",),
        }
        pdag = PC.orient_colliders(skeleton=skel, separating_sets=sep_sets)
        pdag = PC.apply_orientation_rules(pdag)
        self.assertSetEqual(
            set(pdag.edges()), set([("A", "C"), ("B", "C"), ("C", "D")])
        )

        # C - A - B - {C, D} ==> C <- A -> B <- D; B -> C
        skel = nx.Graph([("A", "B"), ("A", "C"), ("B", "C"), ("B", "D")])
        sep_sets = {frozenset({"A", "D"}): tuple(), frozenset({"C", "D"}): ("A", "B")}
        pdag = PC.orient_colliders(skeleton=skel, separating_sets=sep_sets)
        pdag = PC.apply_orientation_rules(pdag)
        self.assertSetEqual(
            set(pdag.edges()), set([("A", "B"), ("B", "C"), ("A", "C"), ("D", "B")])
        )

        skel = nx.Graph([("A", "B"), ("B", "C"), ("A", "D"), ("B", "D"), ("C", "D")])
        sep_sets = {frozenset({"A", "C"}): ("B",)}
        pdag = PC.orient_colliders(skeleton=skel, separating_sets=sep_sets)
        pdag = PC.apply_orientation_rules(pdag)
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

    def test_pdag_to_cpdag(self):
        pdag = PDAG(directed_ebunch=[("A", "B")], undirected_ebunch=[("B", "C")])
        cpdag = PC.apply_orientation_rules(pdag, apply_r4=True)
        self.assertSetEqual(set(cpdag.edges()), {("A", "B"), ("B", "C")})

        pdag = PDAG(
            directed_ebunch=[("A", "B")], undirected_ebunch=[("B", "C"), ("C", "D")]
        )
        cpdag = PC.apply_orientation_rules(pdag, apply_r4=True)
        self.assertSetEqual(set(cpdag.edges()), {("A", "B"), ("B", "C"), ("C", "D")})

        pdag = PDAG(
            directed_ebunch=[("A", "B"), ("D", "C")], undirected_ebunch=[("B", "C")]
        )
        cpdag = PC.apply_orientation_rules(pdag, apply_r4=True)
        self.assertSetEqual(
            set(cpdag.edges()), {("A", "B"), ("D", "C"), ("B", "C"), ("C", "B")}
        )

        pdag = PDAG(
            directed_ebunch=[("A", "B"), ("D", "C"), ("D", "B")],
            undirected_ebunch=[("B", "C")],
        )
        cpdag = PC.apply_orientation_rules(pdag, apply_r4=True)
        self.assertSetEqual(
            set(cpdag.edges()), {("A", "B"), ("D", "C"), ("D", "B"), ("B", "C")}
        )

        pdag = PDAG(
            directed_ebunch=[("A", "B"), ("B", "C")], undirected_ebunch=[("A", "C")]
        )
        cpdag = PC.apply_orientation_rules(pdag, apply_r4=True)
        self.assertSetEqual(set(cpdag.edges()), {("A", "B"), ("B", "C"), ("A", "C")})

        pdag = PDAG(
            directed_ebunch=[("A", "B"), ("B", "C"), ("D", "C")],
            undirected_ebunch=[("A", "C")],
        )
        cpdag = PC.apply_orientation_rules(pdag, apply_r4=True)
        self.assertSetEqual(
            set(cpdag.edges()), {("A", "B"), ("B", "C"), ("A", "C"), ("D", "C")}
        )

        # Examples taken from Perkovi\`c 2017.
        pdag = PDAG(
            directed_ebunch=[("V1", "X")],
            undirected_ebunch=[("X", "V2"), ("V2", "Y"), ("X", "Y")],
        )
        cpdag = PC.apply_orientation_rules(pdag, apply_r4=True)
        self.assertEqual(
            set(cpdag.edges()),
            {("V1", "X"), ("X", "V2"), ("X", "Y"), ("V2", "Y"), ("Y", "V2")},
        )

        pdag = PDAG(
            directed_ebunch=[("Y", "X")],
            undirected_ebunch=[("V1", "X"), ("X", "V2"), ("V2", "Y")],
        )
        cpdag = PC.apply_orientation_rules(pdag, apply_r4=True)
        self.assertEqual(
            set(cpdag.edges()),
            {
                ("X", "V1"),
                ("Y", "X"),
                ("X", "V2"),
                ("V2", "X"),
                ("V2", "Y"),
                ("Y", "V2"),
            },
        )

        # Exmaples from Bang 2024
        pdag = PDAG(
            directed_ebunch=[("B", "D"), ("C", "D")],
            undirected_ebunch=[("A", "D"), ("A", "C")],
        )
        cpdag = PC.apply_orientation_rules(pdag, apply_r4=True)
        self.assertEqual(
            set(cpdag.edges()), {("B", "D"), ("D", "A"), ("C", "A"), ("C", "D")}
        )

        pdag = PDAG(
            directed_ebunch=[("A", "B"), ("C", "B")],
            undirected_ebunch=[("D", "B"), ("D", "A"), ("D", "C")],
        )
        cpdag = PC.apply_orientation_rules(pdag, apply_r4=True)
        self.assertSetEqual(
            set(cpdag.edges()),
            {
                ("A", "B"),
                ("C", "B"),
                ("D", "B"),
                ("D", "A"),
                ("A", "D"),
                ("D", "C"),
                ("C", "D"),
            },
        )

        undirected_edges = [("A", "C"), ("B", "C"), ("D", "C")]
        directed_edges = [("B", "D"), ("D", "A")]

        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)
        mpdag = PC.apply_orientation_rules(pdag, apply_r4=True)
        self.assertSetEqual(
            set(mpdag.edges()),
            set(
                [
                    ("C", "A"),
                    ("C", "B"),
                    ("B", "C"),
                    ("B", "D"),
                    ("D", "A"),
                    ("D", "C"),
                    ("C", "D"),
                ]
            ),
        )

        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)
        pdag = PC.apply_orientation_rules(pdag)
        self.assertSetEqual(
            set(pdag.edges()),
            set(
                [
                    ("A", "C"),
                    ("C", "A"),
                    ("C", "B"),
                    ("B", "C"),
                    ("B", "D"),
                    ("D", "A"),
                    ("D", "C"),
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
                variant="orig",
                ci_test="independence_match",
                return_type="dag",
                n_jobs=2,
                show_progress=False,
            )
            expected_edges = {("B", "D"), ("A", "D"), ("C", "D")}
            self.assertEqual(model.edges(), expected_edges)

            model = DiscreteBayesianNetwork(
                [("A", "C"), ("B", "C"), ("B", "D"), ("C", "E")]
            )
            estimator = PC(independencies=model.get_independencies())
            estimated_model = estimator.estimate(
                variant="orig",
                ci_test="independence_match",
                return_type="dag",
                n_jobs=2,
                show_progress=False,
            )
            expected_edges_1 = set(model.edges())
            expected_edges_2 = {("B", "C"), ("A", "C"), ("C", "E"), ("D", "B")}
            self.assertTrue(
                (set(estimated_model.edges()) == expected_edges_1)
                or (set(estimated_model.edges()) == expected_edges_2)
            )

    def tearDown(self):
        get_reusable_executor().shutdown(wait=True)


class TestPCEstimatorFromDiscreteData(unittest.TestCase):
    def test_build_skeleton_chi_square(self):
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
                show_progress=False,
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
                variant=variant,
                ci_test=fake_ci,
                return_type="skeleton",
                show_progress=False,
            )
            expected_edges = {("X", "Z"), ("Y", "Z")}
            expected_sepsets = {frozenset(("X", "Y")): ("Z",)}
            for u, v in skel.edges():
                self.assertTrue(
                    ((u, v) in expected_edges) or ((v, u) in expected_edges)
                )
            self.assertEqual(sep_sets, expected_sepsets)

    def test_build_skeleton(self):
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randint(0, 2, size=(10000, 5)), columns=list("ABCDE")
        )
        data["F"] = data["A"] + data["B"] + data["C"]
        est = PC(data=data)
        for variant in ["orig", "stable", "parallel"]:
            for test in [
                "g_sq",
                "log_likelihood",
                "modified_log_likelihood",
                "power_divergence",
            ]:
                skel, sep_sets = est.estimate(
                    variant=variant,
                    ci_test=test,
                    return_type="skeleton",
                    significance_level=0.005,
                    n_jobs=2,
                    show_progress=False,
                )

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
                n_jobs=2,
                show_progress=False,
            )
            expected_edges = {("Z", "sum"), ("X", "sum"), ("Y", "sum")}
            self.assertEqual(set(dag.edges()), expected_edges)

    def tearDown(self):
        get_reusable_executor().shutdown(wait=True)


class TestPCEstimatorFromContinuousData(unittest.TestCase):
    def test_build_skeleton(self):
        for ci_test in ["pearsonr", "pillai", "gcm"]:
            for variant in ["orig", "stable", "parallel"]:
                # Fake dataset no: 1
                np.random.seed(42)
                data = pd.DataFrame(np.random.randn(10000, 5), columns=list("ABCDE"))
                data["F"] = data["A"] + data["B"] + data["C"]
                est = PC(data=data)
                skel, sep_sets = est.estimate(
                    variant=variant,
                    ci_test=ci_test,
                    return_type="skeleton",
                    n_jobs=2,
                    show_progress=False,
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
                    variant=variant,
                    ci_test=fake_ci,
                    return_type="skeleton",
                    n_jobs=2,
                    show_progress=False,
                )
                expected_edges = {("X", "Z"), ("Y", "Z")}
                expected_sepsets = {frozenset(("X", "Y")): ("Z",)}

                for u, v in skel.edges():
                    self.assertTrue(
                        ((u, v) in expected_edges) or ((v, u) in expected_edges)
                    )
                self.assertEqual(sep_sets, expected_sepsets)

    def test_build_dag(self):
        for ci_test in ["pearsonr", "pillai", "gcm"]:
            for variant in ["orig", "stable", "parallel"]:
                np.random.seed(42)
                data = pd.DataFrame(np.random.randn(10000, 3), columns=list("XYZ"))
                data["sum"] = data.sum(axis=1)
                est = PC(data=data)
                dag = est.estimate(
                    variant=variant,
                    ci_test=ci_test,
                    return_type="dag",
                    n_jobs=2,
                    show_progress=False,
                )

                expected_edges = {("Z", "sum"), ("X", "sum"), ("Y", "sum")}
                self.assertEqual(set(dag.edges()), expected_edges)

    def tearDown(self):
        get_reusable_executor().shutdown(wait=True)


class TestPCRealModels(unittest.TestCase):
    def test_pc_alarm(self):
        alarm_model = get_example_model("alarm")
        data = BayesianModelSampling(alarm_model).forward_sample(size=int(1e4), seed=42)
        est = PC(data)
        dag = est.estimate(
            variant="stable", max_cond_vars=5, n_jobs=2, show_progress=False
        )

    def test_pc_asia(self):
        asia_model = get_example_model("asia")
        data = BayesianModelSampling(asia_model).forward_sample(size=int(1e5), seed=42)
        est = PC(data)
        req_edges = [("xray", "either")]
        background = ExpertKnowledge(required_edges=req_edges)
        with self.assertLogs(level="WARNING") as cm:
            dag = est.estimate(
                variant="stable",
                max_cond_vars=4,
                expert_knowledge=background,
                n_jobs=2,
                show_progress=False,
            )
        self.assertEqual(
            cm.output,
            [
                "WARNING:pgmpy:Specified expert knowledge conflicts with learned structure. Ignoring edge xray->either from required edges"
            ],
        )

    def test_temporal_pc_cancer(self):
        cancer_model = get_example_model("cancer")
        data = BayesianModelSampling(cancer_model).forward_sample(
            size=int(5e4), seed=42
        )
        est = PC(data)
        background = ExpertKnowledge(  # e.g. we only know "Pollution", "Smoker", "Cancer" can be the causes of others
            temporal_order=[["Pollution", "Smoker", "Cancer"], ["Dyspnoea", "Xray"]],
            max_cond_vars=4,
        )
        pdag = est.estimate(
            variant="stable",
            expert_knowledge=background,
            n_jobs=2,
            show_progress=False,
        )
        self.assertSetEqual(
            set(pdag.edges()),
            set(
                [
                    ("Cancer", "Xray"),
                    ("Cancer", "Dyspnoea"),
                    ("Smoker", "Cancer"),
                    ("Pollution", "Cancer"),
                ]
            ),
        )

    def test_temporal_pc_sachs(self):
        temporal_order = [
            ["PKC", "Plcg"],
            [
                "PKA",
                "Raf",
                "Jnk",
                "P38",
                "PIP3",
                "PIP2",
                "Mek",
                "Erk",
            ],
            ["Akt"],
        ]
        temporal_forbidden_edges = set(
            [
                ("PKA", "PKC"),
                ("PKA", "Plcg"),
                ("Raf", "PKC"),
                ("Raf", "Plcg"),
                ("Jnk", "PKC"),
                ("Jnk", "Plcg"),
                ("P38", "PKC"),
                ("P38", "Plcg"),
                ("PIP3", "PKC"),
                ("PIP3", "Plcg"),
                ("PIP2", "PKC"),
                ("PIP2", "Plcg"),
                ("Mek", "PKC"),
                ("Mek", "Plcg"),
                ("Erk", "PKC"),
                ("Erk", "Plcg"),
                ("Akt", "PKC"),
                ("Akt", "Plcg"),
                ("Akt", "PKA"),
                ("Akt", "Raf"),
                ("Akt", "Jnk"),
                ("Akt", "P38"),
                ("Akt", "PIP3"),
                ("Akt", "PIP2"),
                ("Akt", "Mek"),
                ("Akt", "Erk"),
            ]
        )

        model = get_example_model("sachs")
        df = model.simulate(int(1e3))
        expert = ExpertKnowledge(temporal_order=temporal_order)
        pdag = PC(df).estimate(ci_test="chi_square", expert_knowledge=expert)
        self.assertTrue(temporal_forbidden_edges.isdisjoint(set(pdag.edges())))
