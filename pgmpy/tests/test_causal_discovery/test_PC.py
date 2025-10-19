import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.causal_discovery import PC


def make_estimator():
    return PC()


@parametrize_with_checks([make_estimator()])
def test_pc_compatibility(estimator, check):
    check(estimator)


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


@pytest.fixture
def fake_data():
    np.random.seed(42)
    return pd.DataFrame(np.random.random((1000, 4)), columns=["A", "B", "C", "D"])


@pytest.mark.parametrize("variant", ["orig", "stable"])
def test_build_skeleton(fake_data, variant):
    skel, sep_set = PC()._build_skeleton(fake_data, ci_test=fake_ci_t, variant=variant)
    expected_edges = {("A", "C"), ("A", "D")}
    for u, v in skel.edges():
        assert ((u, v) in expected_edges) or ((v, u) in expected_edges)

    # Test with 0 conditional vars
    skel, sep_set = PC()._build_skeleton(
        fake_data,
        ci_test=fake_ci_t,
        max_cond_vars=0,
        variant=variant,
    )
    expected_edges = {("A", "B"), ("A", "C"), ("A", "D")}
    for u, v in skel.edges():
        assert ((u, v) in expected_edges) or ((v, u) in expected_edges)
