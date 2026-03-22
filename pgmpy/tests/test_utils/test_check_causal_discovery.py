import pandas as pd
import pytest

from pgmpy.causal_discovery import GES, PC, HillClimbSearch
from pgmpy.causal_discovery._base import _BaseCausalDiscovery
from pgmpy.tests.test_causal_discovery import check_causal_discovery


@pytest.mark.parametrize(
    ("estimator", "data_type"),
    [
        (PC(ci_test="chi_square", return_type="dag", show_progress=False), "discrete"),
        (HillClimbSearch(return_type="dag", show_progress=False), "discrete"),
        (GES(return_type="dag"), "discrete"),
    ],
    ids=[
        "PC-discrete",
        "HillClimbSearch-discrete",
        "GES-discrete",
    ],
)
def test_existing_algorithms_pass(estimator, data_type):
    check_causal_discovery(estimator, data_type=data_type)


def test_fails_without_inheritance():
    class BadEstimator:
        pass

    with pytest.raises(TypeError):
        check_causal_discovery(BadEstimator(), data_type="discrete")


def test_fails_without_fit():
    class NoFitEstimator(_BaseCausalDiscovery):
        pass

    with pytest.raises(AssertionError):
        check_causal_discovery(NoFitEstimator(), data_type="discrete")


def test_fails_without_causal_graph():
    class NoCausalGraphEstimator(_BaseCausalDiscovery):
        def _fit(self, X: pd.DataFrame):
            return self

    with pytest.raises(AssertionError):
        check_causal_discovery(NoCausalGraphEstimator(), data_type="discrete")
