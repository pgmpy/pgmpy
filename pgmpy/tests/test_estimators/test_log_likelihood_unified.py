import unittest
import numpy as np
import pandas as pd
from pgmpy.estimators import LogLikelihoodScore
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.metrics.bn_inference import BayesianModelProbability


class TestLogLikelihoodUnified(unittest.TestCase):
    def setUp(self):
        self.model = DiscreteBayesianNetwork([('A', 'B'), ('B', 'C')])
        
        cpd_a = TabularCPD('A', 2, [[0.5], [0.5]])
        cpd_b = TabularCPD('B', 2, [[0.7, 0.3], [0.3, 0.7]], ['A'], [2])
        cpd_c = TabularCPD('C', 2, [[0.8, 0.2], [0.2, 0.8]], ['B'], [2])
        
        self.model.add_cpds(cpd_a, cpd_b, cpd_c)
        
        self.data = pd.DataFrame({
            'A': [0, 0, 1, 1],
            'B': [0, 1, 0, 1],
            'C': [0, 1, 0, 1]
        })
        
        self.score = LogLikelihoodScore(self.data)

    def test_basic_functionality(self):
        
        self.assertIsInstance(self.score, LogLikelihoodScore)
        self.assertEqual(self.score.data.shape, (4, 3))
        self.assertListEqual(list(self.score.data.columns), ['A', 'B', 'C'])
        
        local_score = self.score.local_score('B', ['A'], model=self.model)
        self.assertIsInstance(local_score, float)
        self.assertTrue(np.isfinite(local_score))
        
        score = self.score.score(self.model)
        self.assertIsInstance(score, float)
        self.assertTrue(np.isfinite(score))

    def test_frequency_vs_cpd_scoring(self):
        """Test that both frequency-based and CPD-based scores are finite floats, but not necessarily equal."""
        freq_score = LogLikelihoodScore(self.data, use_cpd=False)
        freq_result = freq_score.score(
            self.model
        )
        cpd_score = LogLikelihoodScore(self.data, use_cpd=True)
        cpd_result = cpd_score.score(self.model)
        self.assertIsInstance(freq_result, float)
        self.assertIsInstance(cpd_result, float)
        self.assertTrue(np.isfinite(freq_result))
        self.assertTrue(np.isfinite(cpd_result))

    def test_consistency_with_old_implementations(self):
        """Test consistency with old implementations (CPD-based)."""
        
        new_score = LogLikelihoodScore(self.data, use_cpd=True).score(
            self.model
        )

        old_score = BayesianModelProbability(self.model).score(
            self.data
        )

        self.assertAlmostEqual(new_score, old_score, places=2)

    def test_mixed_data_types(self):
        """Test scoring with mixed data types"""
        mixed_data = pd.DataFrame({
            'A': [0, 1, 0, 1], 
            'B': ['low', 'high', 'low', 'high'],
            'C': [0.1, 0.2, 0.3, 0.4],
        })
        
        mixed_model = DiscreteBayesianNetwork([('A', 'B'), ('B', 'C')])
        
        cpd_a = TabularCPD('A', 2, [[0.5], [0.5]])
        cpd_b = TabularCPD('B', 2, [[0.7, 0.3], [0.3, 0.7]], ['A'], [2])
        cpd_c = TabularCPD('C', 2, [[0.8, 0.2], [0.2, 0.8]], ['B'], [2])
        
        mixed_model.add_cpds(cpd_a, cpd_b, cpd_c)
        
        # Test scoring
        score = LogLikelihoodScore(mixed_data)
        result = score.score(mixed_model)
        self.assertIsInstance(result, float)

    def test_missing_data(self):
        data_with_missing = self.data.copy()
        data_with_missing.loc[0, 'A'] = np.nan
        
        score = LogLikelihoodScore(data_with_missing)
        result = score.score(self.model)
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))

    def test_invalid_inputs(self):
        empty_data = pd.DataFrame(columns=['A', 'B', 'C'])
        with self.assertRaises(ValueError):
            LogLikelihoodScore(empty_data).score(self.model)
        
        incomplete_data = pd.DataFrame({'A': [0, 1]})
        with self.assertRaises(ValueError):
            LogLikelihoodScore(incomplete_data).score(self.model)
        
        invalid_model = DiscreteBayesianNetwork([('A', 'B')])
        with self.assertRaises(ValueError):
            self.score.score(invalid_model)

    def test_numeric_scoring(self):
        
        numeric_data = pd.DataFrame({
            'A': [0.1, 0.2, 0.3, 0.4],
            'B': [0.5, 0.6, 0.7, 0.8],
            'C': [0.9, 1.0, 1.1, 1.2]
        })
        
        score = LogLikelihoodScore(numeric_data)
        result = score.score(self.model)
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))

    def test_categorical_scoring(self):
        categorical_data = pd.DataFrame({
            'A': ['low', 'low', 'high', 'high'],
            'B': ['low', 'high', 'low', 'high'],
            'C': ['low', 'high', 'low', 'high']
        })
        
        score = LogLikelihoodScore(categorical_data)
        result = score.score(self.model)
        self.assertIsInstance(result, float)
        self.assertLess(result, 0) 