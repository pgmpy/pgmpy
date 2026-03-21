import unittest

import pandas as pd

from pgmpy.causal_discovery import GES, PC, HillClimbSearch
from pgmpy.causal_discovery._base import _BaseCausalDiscovery
from pgmpy.utils.check_causal_discovery import check_causal_discovery


class TestCheckCausalDiscovery(unittest.TestCase):
    """Tests for the check_causal_discovery compliance checker."""

    def test_existing_algorithms_pass(self):
        for estimator in [PC(), GES(), HillClimbSearch()]:
            with self.subTest(estimator=type(estimator).__name__):
                check_causal_discovery(estimator)

    def test_fails_without_inheritance(self):
        class BadEstimator:
            pass

        with self.assertRaises(TypeError):
            check_causal_discovery(BadEstimator())

    def test_fails_without_fit(self):
        class NoFitEstimator(_BaseCausalDiscovery):
            pass

        with self.assertRaises(AssertionError):
            check_causal_discovery(NoFitEstimator())

    def test_fails_without_causal_graph(self):
        class NoCausalGraphEstimator(_BaseCausalDiscovery):
            def _fit(self, X: pd.DataFrame):
                return self

        with self.assertRaises(AssertionError):
            check_causal_discovery(NoCausalGraphEstimator())
