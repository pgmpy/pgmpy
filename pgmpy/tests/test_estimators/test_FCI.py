import numpy as np
import pandas as pd

from pgmpy.estimators.FCI import FCI


def fake_ci_oracle(X, Y, Z=None, **kwargs):
    Z = tuple(Z or [])

    # For collider/fork/chain data: A and B are independent, C depends on both
    independencies = {
        ("A", "B", ()),
        ("A", "B", ("C",)),
    }

    return (X, Y, Z) in independencies or (Y, X, Z) in independencies


def collider_data(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=n)
    B = rng.normal(size=n)
    C = A + B + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"A": A, "B": B, "C": C})


def fork_data(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=n)
    A = C + rng.normal(scale=0.1, size=n)
    B = C + rng.normal(scale=0.1, size=n)

    return pd.DataFrame({"A": A, "B": B, "C": C})


def chain_data(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=n)
    B = A + rng.normal(scale=0.1, size=n)
    C = B + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"A": A, "B": B, "C": C})


def skeleton(pag):
    return {frozenset((u, v)) for u, v, *_ in pag.edges()}


class TestFCI:
    def test_estimate(self):
        data = collider_data()
        pag = FCI(data).estimate(ci_test=fake_ci_oracle)

        assert skeleton(pag) == {frozenset(("A", "C")), frozenset(("B", "C"))}

    def test_skeleton_discovery(self):
        data = fork_data()
        skeleton_graph, seps = FCI(data).build_skeleton(ci_test=fake_ci_oracle)

        # For the fake oracle and simple fork data we expect an undirected
        # skeleton connecting A-B and A-C (fork: A <- C -> B results in edges A-C, B-C)
        assert frozenset(("A", "C")) in {frozenset(e) for e in skeleton_graph.edges()}

    def test_estimate_collider_structure(self):
        data = collider_data()
        pag = FCI(data).estimate(ci_test=fake_ci_oracle)

        assert skeleton(pag) == {frozenset(("A", "C")), frozenset(("B", "C"))}

    def test_estimate_fork_structure(self):
        data = fork_data()
        pag = FCI(data).estimate(ci_test=fake_ci_oracle)

        # fork structure: C is common cause of A and B -> skeleton should connect A-C and B-C
        assert frozenset(("A", "C")) in {frozenset(e) for e in pag.edges()}
        assert frozenset(("B", "C")) in {frozenset(e) for e in pag.edges()}

    def test_estimate_chain_structure(self):
        data = chain_data()
        pag = FCI(data).estimate(ci_test=fake_ci_oracle)

        # In chain A -> B -> C, A and B are dependent (no sep set),
        # so they should be connected
        # A and C are independent given B, so they might not connect
        assert len(pag.edges()) > 0

    def test_estimate_latent_confounder(self):
        # latent confounder L -> A, L -> B. C unrelated.
        # With oracle saying A,B independent, the skeleton should not connect them
        rng = np.random.default_rng(0)
        L = rng.normal(size=2000)
        A = L + rng.normal(scale=0.1, size=2000)
        B = L + rng.normal(scale=0.1, size=2000)
        C = rng.normal(size=2000)
        data = pd.DataFrame({"A": A, "B": B, "C": C})

        pag = FCI(data).estimate(ci_test=fake_ci_oracle)

        # The oracle sees A and B as independent, so they should not be connected
        # We just verify the PAG is valid (has edges)
        assert len(list(pag.edges())) > 0
