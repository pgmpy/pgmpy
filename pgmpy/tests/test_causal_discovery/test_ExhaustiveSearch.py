import unittest

import numpy as np
import pandas as pd

from pgmpy.causal_discovery import ExhaustiveSearch


class TestExhaustiveSearch(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.data = pd.DataFrame(
            np.random.randint(low=0, high=2, size=(100, 3)),
            columns=["A", "B", "C"],
        )
        self.data = self.data.astype("category")

    def test_fit_returns_self(self):
        est = ExhaustiveSearch(scoring_method="bic-d")
        result = est.fit(self.data)
        self.assertEqual(result, est)

    def test_causal_graph_attribute(self):
        est = ExhaustiveSearch(scoring_method="bic-d")
        est.fit(self.data)
        self.assertTrue(hasattr(est, "causal_graph_"))

    def test_adjacency_matrix_attribute(self):
        est = ExhaustiveSearch(scoring_method="bic-d")
        est.fit(self.data)
        self.assertTrue(hasattr(est, "adjacency_matrix_"))
        self.assertEqual(est.adjacency_matrix_.shape, (3, 3))

    def test_variables_attribute(self):
        est = ExhaustiveSearch(scoring_method="bic-d")
        est.fit(self.data)
        self.assertEqual(est.variables_, ["A", "B", "C"])

    def test_all_dags_generates_dags(self):
        est = ExhaustiveSearch()
        est.fit(self.data)
        dags = list(est._all_dags())
        self.assertEqual(len(dags), 25)

    def test_return_type_dag(self):
        est = ExhaustiveSearch(scoring_method="bic-d", return_type="dag")
        est.fit(self.data)
        self.assertTrue(hasattr(est, "causal_graph_"))

    def test_return_type_pdag(self):
        est = ExhaustiveSearch(scoring_method="bic-d", return_type="pdag")
        est.fit(self.data)
        self.assertTrue(hasattr(est, "causal_graph_"))

    def test_invalid_return_type(self):
        est = ExhaustiveSearch(scoring_method="bic-d", return_type="invalid")
        with self.assertRaises(ValueError):
            est.fit(self.data)

    def test_empty_dag_in_all_dags(self):
        est = ExhaustiveSearch()
        est.fit(self.data)
        dags = list(est._all_dags())
        self.assertEqual(len(list(dags[0].edges())), 0)

    def test_refit_on_different_data(self):
        est = ExhaustiveSearch(scoring_method="bic-d")
        est.fit(self.data)

        new_data = pd.DataFrame(
            np.random.randint(low=0, high=2, size=(100, 2)),
            columns=["X", "Y"],
        )
        new_data = new_data.astype("category")
        est.fit(new_data)
        self.assertEqual(est.variables_, ["X", "Y"])


if __name__ == "__main__":
    unittest.main()