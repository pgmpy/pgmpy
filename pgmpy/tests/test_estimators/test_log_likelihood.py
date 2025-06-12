import unittest
import numpy as np
import pandas as pd
import os
from pgmpy.estimators import LogLikelihoodScore
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD


test_data_path = os.path.join(
    os.path.dirname(__file__), "testdata", "mixed_testdata.csv"
)
mixed_testdata = pd.read_csv(test_data_path)


class TestLogLikelihoodScore(unittest.TestCase):
    def setUp(self):
        self.model = DiscreteBayesianNetwork([("A", "B"), ("B", "C")])

        # Create CPDs
        cpd_a = TabularCPD("A", 2, [[0.5], [0.5]])
        cpd_b = TabularCPD("B", 2, [[0.5, 0.5], [0.5, 0.5]], ["A"], [2])
        cpd_c = TabularCPD("C", 2, [[0.5, 0.5], [0.5, 0.5]], ["B"], [2])

        self.model.add_cpds(cpd_a, cpd_b, cpd_c)

        self.data = pd.DataFrame(
            {"A": [0, 0, 1, 1], "B": [0, 1, 0, 1], "C": [0, 1, 1, 0]}
        )

        self.score = LogLikelihoodScore(self.data)

    def test_initialization(self):
        """Test initialization of LogLikelihoodScore"""
        self.assertIsInstance(self.score, LogLikelihoodScore)
        self.assertEqual(self.score.data.shape, (4, 3))
        self.assertListEqual(list(self.score.data.columns), ["A", "B", "C"])

    def test_local_score(self):
        """Test local score computation for a single node"""
        local_score = self.score.local_score("B", ["A"])
        self.assertIsInstance(local_score, float)
        self.assertLess(local_score, 0)

    def test_score(self):
        """Test overall score computation for the model"""
        score = self.score.score(self.model)
        self.assertIsInstance(score, float)
        self.assertLess(score, 0)

    def test_score_with_missing_data(self):
        """Test score computation with missing data"""
        data_with_missing = self.data.copy()
        data_with_missing.loc[0, "A"] = np.nan
        score_with_missing = LogLikelihoodScore(data_with_missing)
        score = score_with_missing.score(self.model)
        self.assertIsInstance(score, float)
        self.assertLess(score, 0)

    def test_score_with_mixed_data(self):
        """Test score computation with mixed data types"""
        score = LogLikelihoodScore(mixed_testdata)
        model = DiscreteBayesianNetwork([("A", "B"), ("B", "C")])

        cpd_a = TabularCPD("A", 2, [[0.5], [0.5]])
        cpd_b = TabularCPD("B", 2, [[0.5, 0.5], [0.5, 0.5]], ["A"], [2])
        cpd_c = TabularCPD("C", 2, [[0.5, 0.5], [0.5, 0.5]], ["B"], [2])
        model.add_cpds(cpd_a, cpd_b, cpd_c)
        score_value = score.score(model)
        self.assertIsInstance(score_value, float)
        self.assertLess(score_value, 0)

    def test_score_consistency(self):
        """Test that score is consistent across multiple calls"""
        score1 = self.score.score(self.model)
        score2 = self.score.score(self.model)
        self.assertEqual(score1, score2)

    def test_score_with_different_models(self):

        model2 = DiscreteBayesianNetwork([("A", "C"), ("B", "C")])
        cpd_a = TabularCPD("A", 2, [[0.5], [0.5]])
        cpd_b = TabularCPD("B", 2, [[0.5], [0.5]])
        cpd_c = TabularCPD(
            "C", 2, [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]], ["A", "B"], [2, 2]
        )
        model2.add_cpds(cpd_a, cpd_b, cpd_c)

        score1 = self.score.score(self.model)
        score2 = self.score.score(model2)
        self.assertNotEqual(score1, score2)

    def test_score_with_empty_data(self):
        """Test score computation with empty data"""
        empty_data = pd.DataFrame(columns=["A", "B", "C"])
        score = LogLikelihoodScore(empty_data)
        with self.assertRaises(ValueError):
            score.score(self.model)

    def test_score_with_invalid_model(self):
        """Test score computation with an invalid model"""
        invalid_model = DiscreteBayesianNetwork([("A", "B"), ("B", "C")])

        with self.assertRaises(ValueError):
            self.score.score(invalid_model)
