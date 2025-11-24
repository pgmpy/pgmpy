import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class _BaseCausalPrediction(RegressorMixin, BaseEstimator):
    """
    Base class for causal prediction algorithms in pgmpy. Provides common
    functionality for preparing and validating feature dataframes.
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        return tags

    def _prepare_feature_df(self, X, required_features) -> pd.DataFrame:
        """
        Convert input (either numpy array or dataframe) to a DataFrame and
        validate that column names exactly match DAG variables.

        If a numpy array is provided, it is converted to a DataFrame with
        range index column names (0, 1, ..., n_features-1).

        Parameters
        ----------
        X : array-like or DataFrame
            Input features.

        required_features : list[int]
            Column indices expected from the DAG.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only required columns.
        """

        # Step 1: Check required features is provided
        if required_features is None:
            raise ValueError("required_features must be provided.")

        # Step 2: Convert input to DataFrame format
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            # For numpy arrays, use range index as column names
            X_arr = np.asarray(X)
            if X_arr.ndim == 1:
                raise ValueError(
                    "Reshape your data: X must be 2D. If using a 1D array, reshape it to (n_samples, 1)."
                )
            X_df = pd.DataFrame(X_arr, columns=range(X_arr.shape[1]))

        # Step 3: Validation: column names must exactly match DAG variables
        missing_columns = set(required_features) - set(X_df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns in input data: {list(missing_columns)}. "
                f"DAG expects columns: {required_features}, but got: {list(X_df.columns)}"
            )

        return X_df[required_features]
