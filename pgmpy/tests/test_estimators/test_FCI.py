from pgmpy.estimators.FCI import FCI
from pgmpy.independencies.Independencies import Independencies
from pgmpy.sampling.Sampling import BayesianModelSampling
from pgmpy.utils import get_example_model


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


# Function to define different real world datasets


def pc_alarm_data():
    alarm_model = get_example_model("alarm")
    data = BayesianModelSampling(alarm_model).forward_sample(size=int(1e4), seed=42)
    return data


def pc_asia_data():
    model = get_example_model("asia")
    data = model.simulate(n_samples=5000, show_progress=False)
    return data


def skeleton(pag):
    """Return undirected skeleton as frozensets of node pairs."""
    return {frozenset((u, v)) for u, v, _, _ in pag.edges()}


def test_estimate_collider_structure():
    # np.random.seed(42)
    # n = 5000

    # A = np.random.randn(n)
    # B = np.random.randn(n)
    # C = A + B + np.random.randn(n) * 0.1

    # data = pd.DataFrame({"A": A, "B": B, "C": C})

    data = pc_alarm_data()
    pag = FCI(data).estimate(ci_test=fake_ci_t)

    assert skeleton(pag) == {
        frozenset(("A", "C")),
        frozenset(("B", "C")),
    }

    assert pag.get_edge_pag.get_edge_markss(pag, "A", "C")[1] == ">"
    assert pag.get_edge_pag.get_edge_markss(pag, "B", "C")[1] == ">"


def test_estimate_fork_structure():
    # np.random.seed(42)
    # n = 5000

    # C = np.random.randn(n)
    # A = C + np.random.randn(n) * 0.1
    # B = C + np.random.randn(n) * 0.1

    # data = pd.DataFrame({"A": A, "B": B, "C": C})

    data = pc_asia_data()
    # can use the fake ci test from test_PC.py
    pag = FCI(data).estimate(ci_test=fake_ci_t)

    assert skeleton(pag) == {
        frozenset(("A", "C")),
        frozenset(("B", "C")),
    }

    assert pag.get_edge_marks(pag, "A", "C")[1] != ">"
    assert pag.get_edge_marks(pag, "B", "C")[1] != ">"


def test_estimate_chain_structure():
    # np.random.seed(42)
    # n = 5000

    # A = np.random.randn(n)
    # B = A + np.random.randn(n) * 0.1
    # C = B + np.random.randn(n) * 0.1

    # data = pd.DataFrame({"A": A, "B": B, "C": C})

    data = pc_asia_data()
    pag = FCI(data).estimate(ci_test=fake_ci_t)

    assert skeleton(pag) == {
        frozenset(("A", "B")),
        frozenset(("B", "C")),
    }

    assert not (
        pag.get_edge_marks(pag, "A", "B")[1] == ">"
        and pag.get_edge_marks(pag, "C", "B")[1] == ">"
    )


def test_estimate_latent_confounder():
    # np.random.seed(42)
    # n = 6000

    # L = np.random.randn(n)
    # X = L + np.random.randn(n) * 0.1
    # Y = L + np.random.randn(n) * 0.1

    # data = pd.DataFrame({"X": X, "Y": Y})

    data = pc_asia_data()
    pag = FCI(data).estimate(ci_test=fake_ci_t)

    assert skeleton(pag) == {
        frozenset(("X", "Y")),
    }

    assert pag.get_edge_marks(pag, "X", "Y") == ("o", "o")


def test_estimate_from_independencies():
    ind = Independencies(
        ["A", "B"],
        ["A", ["B"], "C"],
    ).closure()

    pag = FCI(independencies=ind).estimate(ci_test="independence_match")

    assert skeleton(pag) == {
        frozenset(("A", "C")),
        frozenset(("B", "C")),
    }
