import unittest
import pandas as pd
import numpy as np
from pgmpy.estimators import IGCI


class TestIGCI(unittest.TestCase):
    def setUp(self):
        # Generate synthetic cause-effect data for testing
        np.random.seed(42)
        self.x = np.random.uniform(0, 5, 1000)
        self.y = np.square(self.x) + 0.1 * np.random.normal(0, 1, 1000)
        self.data = pd.DataFrame({'X': self.x, 'Y': self.y})
        
        self.z = np.random.uniform(-2, 2, 1000)
        self.w = np.exp(self.z) + 0.2 * np.random.normal(0, 1, 1000)
        self.data['Z'] = self.z
        self.data['W'] = self.w

    def test_initialization(self):
        """Test IGCI initialization"""
        igci = IGCI(self.data)
        self.assertEqual(igci.data.shape, self.data.shape)
        self.assertFalse(igci.assume_normalized)
        
        igci_normalized = IGCI(self.data, assume_normalized=True)
        self.assertTrue(igci_normalized.assume_normalized)
        
    def test_normalize_data(self):
        """Test data normalization function"""
        igci = IGCI(self.data)
        norm_data = igci._normalize_data(self.x)
        self.assertTrue(np.all(norm_data >= 0) and np.all(norm_data <= 1))
        self.assertAlmostEqual(np.min(norm_data), 0, places=5)
        self.assertAlmostEqual(np.max(norm_data), 1, places=5)
        
        const_data = np.ones(100)
        norm_const = igci._normalize_data(const_data)
        self.assertTrue(np.all(norm_const == 0))
        
    def test_direction_estimation_entropy(self):
        """Test causal direction estimation using entropy method"""
        igci = IGCI(self.data)
        
        direction_xy = igci.estimate_direction('X', 'Y', method='entropy')
        self.assertEqual(direction_xy, 1)
        
        direction_zw = igci.estimate_direction('Z', 'W', method='entropy')
        self.assertEqual(direction_zw, 1)
        
        direction_yx = igci.estimate_direction('Y', 'X', method='entropy')
        self.assertEqual(direction_yx, -1)
        
    def test_direction_estimation_slope(self):
        """Test causal direction estimation using slope method"""
        igci = IGCI(self.data)
        
        direction_xy = igci.estimate_direction('X', 'Y', method='slope')
        self.assertEqual(direction_xy, 1)
        
        direction_zw = igci.estimate_direction('Z', 'W', method='slope')
        self.assertEqual(direction_zw, 1)
        
    def test_invalid_method(self):
        """Test that invalid methods raise ValueError"""
        igci = IGCI(self.data)
        with self.assertRaises(ValueError):
            igci.estimate_direction('X', 'Y', method='invalid_method')
            
    def test_estimate_all_pairs(self):
        """Test estimating causal directions for all variable pairs"""
        igci = IGCI(self.data)
        results = igci.estimate(method='entropy')
        
        expected_pairs = [('X', 'Y'), ('X', 'Z'), ('X', 'W'), ('Y', 'Z'), ('Y', 'W'), ('Z', 'W')]
        self.assertEqual(set(results.keys()), set(expected_pairs))
        
        self.assertEqual(results[('X', 'Y')], 1)  # X -> Y
        self.assertEqual(results[('Z', 'W')], 1)  # Z -> W
        
    def test_estimate_specified_pairs(self):
        """Test estimating causal directions for specified variable pairs"""
        igci = IGCI(self.data)
        specified_pairs = [('X', 'Y'), ('Z', 'W')]
        results = igci.estimate(variables=specified_pairs, method='entropy')
        
        self.assertEqual(set(results.keys()), set(specified_pairs))
        
        self.assertEqual(results[('X', 'Y')], 1)  # X -> Y
        self.assertEqual(results[('Z', 'W')], 1)  # Z -> W
        
    def test_with_nan_values(self):
        """Test handling of NaN values"""
        data_with_nan = self.data.copy()
        data_with_nan.iloc[0:10, 0] = np.nan  # Set some X values to NaN
        data_with_nan.iloc[20:30, 1] = np.nan  # Set some Y values to NaN
        
        igci = IGCI(data_with_nan)
        direction = igci.estimate_direction('X', 'Y', method='entropy')
        self.assertEqual(direction, 1)

if __name__ == "__main__":
    unittest.main()