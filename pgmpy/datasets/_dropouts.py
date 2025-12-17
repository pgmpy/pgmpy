import pandas as pd

from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class Dropouts(_BaseDataset):
    name = "dropouts"
    tags = {
        "n_variables": 8,
        "n_samples": 159,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/dropouts/"

    data_url = base_url + "data/dropouts.cov.txt"
    ground_truth_url = None
    expert_knowledge_url = None

    @classmethod
    def load_dataframe(cls) -> pd.DataFrame:
        """Custom parser for dropouts covariance matrix format."""
        raw_data = cls._get_raw_data("data", cls.data_url)
        text = raw_data.decode("utf-8-sig", errors="ignore")
        lines = text.strip().split('\n')

        # Parse the covariance matrix format
        n_samples = int(lines[0])  # First line is sample size
        variable_names = lines[1].split('\t')  # Second line is variable names
        n_vars = len(variable_names)

        # Parse correlation/covariance matrix (lower triangular)
        corr_matrix = []
        for i in range(2, 2 + n_vars):
            if i < len(lines):
                row_values = lines[i].split('\t')
                # Pad with zeros to make full row
                full_row = ['0.0'] * n_vars
                for j, val in enumerate(row_values):
                    if j < len(full_row):
                        full_row[j] = val
                corr_matrix.append([float(x) for x in full_row])

        # Convert to symmetric matrix
        import numpy as np
        matrix = np.array(corr_matrix)

        # Make symmetric (copy lower triangle to upper)
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                matrix[i, j] = matrix[j, i]

        # Generate synthetic data from correlation matrix
        # Using Cholesky decomposition for correlated normal data
        np.random.seed(42)  # For reproducibility
        L = np.linalg.cholesky(matrix)
        uncorr_data = np.random.randn(n_samples, n_vars)
        corr_data = uncorr_data @ L.T

        # Create DataFrame
        df = pd.DataFrame(corr_data, columns=variable_names)
        return df