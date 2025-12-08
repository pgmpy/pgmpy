import unittest

import pandas as pd

from pgmpy.base import DAG
from pgmpy.datasets import load_dataset
from pgmpy.datasets.registry import DATASETS


class TestDatasets(unittest.TestCase):
    def test_registry(self):
        """Test if datasets are registered correctly."""
        datasets = DATASETS.list_all()
        self.assertIn("abalone", datasets)
        self.assertIn("sachs", datasets)

    def test_load_dataset_return_types(self):
        """Test the conditional return types of load_dataset."""
        # 1. load_ground_truth=True (Default) -> returns Tuple
        df, gt = load_dataset("sachs")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(gt, DAG)

        # 2. load_ground_truth=False -> returns DataFrame only
        df_only = load_dataset("sachs", load_ground_truth=False)
        self.assertIsInstance(df_only, pd.DataFrame)
        self.assertNotIsInstance(df_only, tuple)

    def test_sachs_jittered_variant(self):
        """Test specific variant logic where nodes mismatch."""
        df, gt = load_dataset("sachs", variant="jittered_experimental")

        # This variant has 20 vars (11 proteins + 9 interventions)
        # But ground truth only defines edges for proteins
        self.assertEqual(df.shape[1], 20)
        self.assertEqual(len(gt.nodes()), 20)

        # Verify edge count is still 20 (standard Sachs edges)
        self.assertEqual(len(gt.edges()), 20)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            load_dataset("non_existent_dataset")

        with self.assertRaises(KeyError):
            load_dataset("sachs", variant="bad_variant")
