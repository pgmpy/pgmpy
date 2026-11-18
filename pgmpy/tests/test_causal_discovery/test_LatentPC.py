import networkx as nx
import numpy as np
import pandas as pd
import pytest

from pgmpy.base import PDAG
from pgmpy.causal_discovery import ExpertKnowledge, LatentPC
from pgmpy.example_models import load_model


def test_skeleton_to_pdag():
    latent_pc = LatentPC()

    # D - A - C - B  ==> D - A -> C <- B
    latent_pc.skeleton_ = nx.Graph([("A", "D"), ("A", "C"), ("B", "C")])
    latent_pc.separating_sets_ = {
        frozenset({"D", "C"}): ("A",),
        frozenset({"A", "B"}): tuple(),
        frozenset({"D", "B"}): ("A",),
    }
    pdag = latent_pc._orient_colliders()
    pdag = pdag.apply_meeks_rules(apply_r4=False)
    assert set(pdag.get_edges(data=True)) == {
        ("A", "C", "->"),
        ("A", "D", "--"),
        ("B", "C", "->"),
    }

    # C - A - B  ==> C -> A <- B
    latent_pc.skeleton_ = nx.Graph([("A", "B"), ("A", "C")])
    latent_pc.separating_sets_ = {frozenset({"B", "C"}): ()}
    pdag = latent_pc._orient_colliders()
    pdag = pdag.apply_meeks_rules(apply_r4=False)
    assert set(pdag.get_edges(data=True)) == {("B", "A", "->"), ("C", "A", "->")}

    # C - A - B ==> C - A - B
    latent_pc.skeleton_ = nx.Graph([("A", "B"), ("A", "C")])
    latent_pc.separating_sets_ = {frozenset({"B", "C"}): ("A",)}
    pdag = latent_pc._orient_colliders()
    pdag = pdag.apply_meeks_rules(apply_r4=False)
    assert set(pdag.get_edges(data=True)) == {("A", "B", "--"), ("A", "C", "--")}

    # {A, B} - C - D ==> {A, B} -> C -> D
    latent_pc.skeleton_ = nx.Graph([("A", "C"), ("B", "C"), ("C", "D")])
    latent_pc.separating_sets_ = {
        frozenset({"A", "B"}): tuple(),
        frozenset({"A", "D"}): ("C",),
        frozenset({"B", "D"}): ("C",),
    }
    pdag = latent_pc._orient_colliders()
    pdag = pdag.apply_meeks_rules(apply_r4=False)
    assert set(pdag.get_edges(data=True)) == {
        ("A", "C", "->"),
        ("B", "C", "->"),
        ("C", "D", "->"),
    }

    # C - A - B - {C, D} ==> C <- A -> B <- D; B -> C
    latent_pc.skeleton_ = nx.Graph([("A", "B"), ("A", "C"), ("B", "C"), ("B", "D")])
    latent_pc.separating_sets_ = {
        frozenset({"A", "D"}): tuple(),
        frozenset({"C", "D"}): ("A", "B"),
    }
    pdag = latent_pc._orient_colliders()
    pdag = pdag.apply_meeks_rules(apply_r4=False)
    assert set(pdag.get_edges(data=True)) == {
        ("A", "B", "->"),
        ("A", "C", "->"),
        ("B", "C", "->"),
        ("D", "B", "->"),
    }

    latent_pc.skeleton_ = nx.Graph([("A", "B"), ("B", "C"), ("A", "D"), ("B", "D"), ("C", "D")])
    latent_pc.separating_sets_ = {frozenset({"A", "C"}): ("B",)}
    pdag = latent_pc._orient_colliders()
    pdag = pdag.apply_meeks_rules(apply_r4=False)
    assert pdag == PDAG(
        edge_list=[
            ("A", "D", "->"),
            ("B", "D", "->"),
            ("C", "D", "->"),
            ("A", "B", "--"),
            ("B", "C", "--"),
        ]
    )

    # A - B - C - D: two conflicting colliders at B and C.
    # A->B<-C and B->C<-D conflict on the B-C edge. The second
    # collider should be skipped so the edge is not deleted.
    latent_pc.skeleton_ = nx.Graph([("A", "B"), ("B", "C"), ("C", "D")])
    latent_pc.separating_sets_ = {
        frozenset({"A", "C"}): tuple(),
        frozenset({"A", "D"}): ("B",),
        frozenset({"B", "D"}): tuple(),
    }
    pdag = latent_pc._orient_colliders()
    assert set(pdag.get_edges(data=True)) == {
        ("A", "B", "->"),
        ("C", "B", "->"),
        ("C", "D", "--"),
    }

    # Three colliders A->B<-E, B->C<-F, C->A<-G would orient A->B->C->A into a directed
    # cycle. The third collider closes the cycle, so it is skipped entirely and both of
    # its edges (C-A and G-A) are left undirected instead of crashing PDAG construction.
    latent_pc.skeleton_ = nx.Graph([("A", "B"), ("E", "B"), ("B", "C"), ("F", "C"), ("C", "A"), ("G", "A")])
    latent_pc.separating_sets_ = {
        frozenset({"A", "E"}): tuple(),
        frozenset({"B", "F"}): tuple(),
        frozenset({"C", "G"}): tuple(),
    }
    pdag = latent_pc._orient_colliders()
    assert pdag == PDAG(
        edge_list=[
            ("A", "B", "->"),
            ("E", "B", "->"),
            ("B", "C", "->"),
            ("F", "C", "->"),
            ("C", "A", "--"),
            ("G", "A", "--"),
        ]
    )


def test_build_skeleton_chi_square():

    # Fake dataset no: 1
    np.random.seed(42)
    data = pd.DataFrame(np.random.randint(0, 2, size=(10000, 5)), columns=list("ABCDE"))
    data["F"] = data["A"] + data["B"] + data["C"]
    est = LatentPC(
        ci_test="chi_square",
        significance_level=0.005,
        show_progress=False,
    )
    est.fit(X=[data])
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
    for u, v in est.skeleton_.edges():
        assert ((u, v) in expected_edges) or ((v, u) in expected_edges)
    assert est.separating_sets_ == expected_sepsets

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
    est = LatentPC(
        ci_test=fake_ci,
        show_progress=False,
    )
    est.fit(X=[fake_data])
    expected_edges = {("X", "Z"), ("Y", "Z")}
    expected_sepsets = {frozenset(("X", "Y")): ("Z",)}
    for u, v in est.skeleton_.edges():
        assert ((u, v) in expected_edges) or ((v, u) in expected_edges)
    assert est.separating_sets_ == expected_sepsets


def test_build_skeleton_discrete():
    np.random.seed(42)
    data = pd.DataFrame(np.random.randint(0, 2, size=(10000, 5)), columns=list("ABCDE"))
    data["F"] = data["A"] + data["B"] + data["C"]

    for test in [
        "g_sq",
        "log_likelihood",
        "modified_log_likelihood",
        "power_divergence",
    ]:
        est = LatentPC(
            ci_test=test,
            significance_level=0.005,
            n_jobs=1,
            show_progress=False,
        )
        est.fit(X=[data])


def test_search_space():
    adult_data = pd.read_csv("pgmpy/tests/test_estimators/testdata/adult.csv")

    search_space = [
        ("Age", "Education"),
        ("Education", "HoursPerWeek"),
        ("Education", "Income"),
        ("HoursPerWeek", "Income"),
        ("Age", "Income"),
    ]

    allowed_adjacencies = {frozenset(edge) for edge in search_space}

    expert_knowledge = ExpertKnowledge(search_space=search_space)

    est = LatentPC(
        expert_knowledge=expert_knowledge,
        show_progress=False,
    )

    est.fit(X=[adult_data])
    # The learned adjacencies are restricted to the search space (no flag needed).
    for u, v in est.causal_graph_.edges():
        assert frozenset((u, v)) in allowed_adjacencies


def test_required_edge_enforced_against_independent_data():
    # A, B, C are mutually independent, so PC would normally drop every edge.
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "A": rng.integers(0, 2, size=2000),
            "B": rng.integers(0, 2, size=2000),
            "C": rng.integers(0, 2, size=2000),
        }
    )
    expert = ExpertKnowledge(required_edges=[("A", "B")])
    est = LatentPC(
        ci_test="chi_square",
        expert_knowledge=expert,
        n_jobs=1,
        show_progress=False,
    )
    est.fit(X=[data])
    pdag = est.causal_graph_
    # The required edge is retained and oriented A->B despite A, B being independent.
    print(pdag.directed_edges)
    assert ("A", "B") in pdag.directed_edges


@pytest.mark.parametrize("ci_test", ["pearsonr", "pillai", "gcm"])
def test_build_skeleton_continuous(ci_test):

    # Fake dataset no: 1
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(10000, 5), columns=list("ABCDE"))
    data["F"] = data["A"] + data["B"] + data["C"]
    est = LatentPC(
        ci_test=ci_test,
        n_jobs=1,
        show_progress=False,
    )
    est.fit(X=[data])
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
    }
    for u, v in est.skeleton_.edges():
        assert ((u, v) in expected_edges_stable) or ((v, u) in expected_edges_stable)

    for key, value in est.separating_sets_.items():
        assert est.separating_sets_[key] == expected_sepsets[key]

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
    est = LatentPC(
        ci_test=fake_ci,
        n_jobs=1,
        show_progress=False,
    )
    est.fit(X=[data])
    expected_edges = {("X", "Z"), ("Y", "Z")}
    expected_sepsets = {frozenset(("X", "Y")): ("Z",)}

    for u, v in est.skeleton_.edges():
        assert ((u, v) in expected_edges) or ((v, u) in expected_edges)
    assert est.separating_sets_ == expected_sepsets


def test_temporal_pc_cancer():
    cancer_model = load_model("bnlearn/cancer")
    data = cancer_model.simulate(n_samples=int(5e4), seed=42)

    background = ExpertKnowledge(  # e.g. we know only "Pollution", "Smoker", "Cancer" can be the causes of others
        temporal_order=[["Pollution", "Smoker", "Cancer"], ["Dyspnoea", "Xray"]],
        max_cond_vars=4,
    )
    est = LatentPC(
        expert_knowledge=background,
        n_jobs=1,
        show_progress=False,
    )
    est.fit(X=[data])
    pdag = est.causal_graph_
    assert set(pdag.get_edges(data=True)) == {
        ("Cancer", "Xray", "->"),
        ("Cancer", "Dyspnoea", "->"),
        ("Smoker", "Cancer", "->"),
        ("Pollution", "Cancer", "->"),
    }


def test_temporal_pc_sachs():
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
    temporal_forbidden_edges = {
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
    }

    model = load_model("bnlearn/sachs")
    df = model.simulate(int(1e3))

    expert = ExpertKnowledge(temporal_order=temporal_order)
    pdag = LatentPC(ci_test="chi_square", expert_knowledge=expert).fit(X=[df]).causal_graph_
    learned_edges = set()
    for u, v, edge_type in pdag.get_edges(data=True):
        learned_edges.add((u, v))
        if edge_type == "--":
            learned_edges.add((v, u))
    assert temporal_forbidden_edges.isdisjoint(learned_edges)


@pytest.mark.parametrize("orient_rule", ["pvalue", "effect"])
def test_orient_rule(orient_rule):
    cancer_model = load_model("bnlearn/cancer")
    data = cancer_model.simulate(n_samples=int(5e4), seed=42)

    est = LatentPC(
        ci_test="chi_square",
        orient_rule=orient_rule,
        show_progress=False,
    )
    est.fit(X=[data])

    pdag = est.causal_graph_
    assert ("Pollution", "Cancer") in pdag.edges() or (
        "Cancer",
        "Pollution",
    ) in pdag.edges()
    assert ("Smoker", "Cancer") in pdag.edges() or ("Cancer", "Smoker") in pdag.edges()


def test_latent_model_binomial():
    def generate_binomial_df(n, p_x, seed):
        rng = np.random.default_rng(seed)

        # Independent roots.
        X = rng.binomial(1, p_x, size=n)
        Z = rng.binomial(1, 0.45, size=n)

        # Y depends on both X and Z.
        p_y = np.select(
            [
                (X == 0) & (Z == 0),
                (X == 1) & (Z == 0),
                (X == 0) & (Z == 1),
                (X == 1) & (Z == 1),
            ],
            [0.03, 0.80, 0.75, 0.97],
        )
        Y = rng.binomial(1, p_y)

        p_w = np.where(Y == 1, 0.90, 0.10)
        W = rng.binomial(1, p_w)

        return pd.DataFrame({"X": X, "Z": Z, "Y": Y, "W": W}).astype("category")

    datasets = [
        generate_binomial_df(n=5000, p_x=0.20, seed=42),
        generate_binomial_df(n=5000, p_x=0.80, seed=0),
    ]

    model = LatentPC(significance_level=0.01, show_progress=False)
    model.fit(datasets)
    assert sorted([("X", "Y", "->"), ("f_1", "X", "->"), ("Z", "Y", "->"), ("Y", "W", "->")]) == sorted(
        model.causal_graph_.get_edges()
    )

    # Model with no significant change in mechanism should not have an F-node (latent) edge
    datasets_2 = [
        generate_binomial_df(n=5000, p_x=0.15, seed=42),
        generate_binomial_df(n=5000, p_x=0.17, seed=0),
    ]
    model = LatentPC(significance_level=0.01, show_progress=False)
    model.fit(datasets_2)
    assert sorted([("X", "Y", "->"), ("Z", "Y", "->"), ("Y", "W", "->")]) == sorted(model.causal_graph_.get_edges())


def test_latent_model_gaussian():
    def generate_gaussian_df(n, mean_s, seed):
        rng = np.random.default_rng(seed)

        S = rng.normal(loc=mean_s, scale=1.0, size=n)

        A = 1.1 * S + rng.normal(scale=0.6, size=n)
        B = -0.9 * S + rng.normal(scale=0.6, size=n)

        C = 1.2 * A + rng.normal(scale=0.5, size=n)
        D = 0.8 * B + rng.normal(scale=0.5, size=n)

        E = 1.0 * C - 1.1 * D + rng.normal(scale=0.5, size=n)

        return pd.DataFrame(
            {
                "S": S,
                "A": A,
                "B": B,
                "C": C,
                "D": D,
                "E": E,
            }
        )

    datasets = [
        generate_gaussian_df(n=5000, mean_s=-1.5, seed=42),
        generate_gaussian_df(n=5000, mean_s=1.5, seed=0),
    ]

    model = LatentPC(significance_level=0.005, max_cond_vars=4, show_progress=False)

    model.fit(datasets)
    assert sorted(
        [
            ("f_1", "S", "->"),
            ("S", "A", "->"),
            ("S", "B", "->"),
            ("A", "C", "->"),
            ("B", "D", "->"),
            ("C", "E", "->"),
            ("D", "E", "->"),
        ]
    ) == sorted(model.causal_graph_.get_edges())
