import pandas as pd
import pytest

from pgmpy.causal_discovery._base import _BaseCausalDiscovery
from pgmpy.tests.test_causal_discovery import check_causal_discovery


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
