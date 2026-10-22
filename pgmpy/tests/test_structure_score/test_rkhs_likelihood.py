import numpy as np
import pandas as pd
import pytest

from pgmpy.base import DAG
from pgmpy.structure_score import RKHSLikelihood


@pytest.fixture
def rkhs_case1_score():
    """
    Test data from Huang et al. (KDD 2018), Section 2, Case 1.

    Reference
    ---------
    https://dl.acm.org/doi/abs/10.1145/3219819.3220104  
    """
    np.random.seed(42)
    n = 500
    E1 = np.random.normal(0, 0.5, n)
    E2 = np.random.normal(0, 0.5, n)
    E3 = np.random.normal(0, 0.5, n)
    X1 = E1
    X2 = 0.8 * (X1 + X1**2) + E2
    X3 = 0.8 * (X2 + X2**2) + E3
    data = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})
    return RKHSLikelihood(data)


@pytest.fixture
def rkhs_case2_score():
    """
    Test data from Huang et al. (KDD 2018), Section 2, Case 2.

    Reference
    ---------
    https://dl.acm.org/doi/abs/10.1145/3219819.3220104
    """
    np.random.seed(42)
    n = 500
    E1 = np.random.normal(0, 0.5, n)
    E2 = np.random.normal(0, 0.5, n)
    X1 = E1
    X2 = (np.sin(X1) + E2) ** 2
    data = pd.DataFrame({"X1": X1, "X2": X2})
    return RKHSLikelihood(data)


class TestRKHSLikelihood:
    def test_local_score_no_parents(self, rkhs_case1_score):
        print(rkhs_case1_score.local_score("X1", ()))
        print(rkhs_case1_score.local_score("X2", ()))
        print(rkhs_case1_score.local_score("X3", ()))
        assert rkhs_case1_score.local_score(variable="X1", parents=()) == pytest.approx(4174481.9044, abs=1e-3)
        assert rkhs_case1_score.local_score(variable="X2", parents=()) == pytest.approx(4115699.3699, abs=1e-3)
        assert rkhs_case1_score.local_score(variable="X3", parents=()) == pytest.approx(3974301.0309, abs=1e-3)

    def test_local_score_with_parents(self, rkhs_case1_score):
        print(rkhs_case1_score.local_score("X2", ("X1",)))
        print(rkhs_case1_score.local_score("X3", ("X1", "X2")))
        assert rkhs_case1_score.local_score(variable="X2", parents=("X1",)) == pytest.approx(4088057.4804, abs=1e-3)
        assert rkhs_case1_score.local_score(variable="X3", parents=("X1", "X2",)) == pytest.approx(3949536.9518, abs=1e-3)

    def test_detects_nonlinear_dependence(self, rkhs_case2_score):
        print(rkhs_case2_score.local_score("X1", ()))
        print(rkhs_case2_score.local_score("X2", ("X1",)))
        assert rkhs_case2_score.local_score(variable="X1", parents=()) == pytest.approx(4139245.0009, abs=1e-3)
        assert rkhs_case2_score.local_score(variable="X2", parents=("X1",)) == pytest.approx(4192936.6696, abs=1e-3)
