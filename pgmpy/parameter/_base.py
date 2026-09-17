import numpy as np
import pandas as pd
from skbase.base import BaseEstimator as _BaseEstimator


class BaseParameter(_BaseEstimator):
    """Base class for all parameter classes in pgmpy."""

    _tags = {
        "parameter_type": str,  # classifier, regressor
        "produces_factor": bool,  # Only for TabularCPD
        "is_linear_gaussian": bool,  # Only for LinearGaussianCPD
        "supports_fit_joint": bool,  # Only for FunctionalCPD
        "missing": bool,
        "can_be_root": bool,
    }

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, sample_weight=None):
        """Fit parameter to training data.

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pandas DataFrame, must be same length as X
            labels to fit regressor to

        Returns
        -------
        self : reference to self
        """
        X, y, sample_weight = self._check_data(X, y, sample_weight)
        self._fit(X, y, sample_weight)
        self._is_fitted = True
        return self

    def _fit(self, X, y=None, sample_weight=None):
        raise NotImplementedError

    def predict_proba(self, X):
        """Predict distribution over labels for data from features.

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y : skpro's Distribution class
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' before calling 'predict_proba'."
            )
        X, _, _ = self._check_data(X)
        y_pred = self._predict_proba(X)
        return y_pred

    def _predict_proba(self, X):
        raise NotImplementedError

    def _check_data(self, X, y=None, sample_weight=None):
        """check train data with tag"""
        # Check X
        if isinstance(X, pd.Series):
            X = X.to_frame()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if X.shape[0] == 0:
            raise ValueError("X must have at least one row.")

        if X.shape[1] == 0:
            raise ValueError("X must have at least one column.")

        if not self.get_tag("missing"):
            if X.isna().any().any():
                raise ValueError(f"{self.__class__.__name__} cannot deal with missing values.")

        if y is None:
            n_samples = len(X)
        else:
            # Check y
            if isinstance(y, pd.Series):
                y = y.to_frame()

            if not isinstance(y, pd.DataFrame):
                raise TypeError("y must be a pandas DataFrame.")

            if y.shape[0] == 0:
                raise ValueError("y must have at least one row.")

            if y.shape[1] == 0:
                raise ValueError("y must have at least one column.")

            if self.get_tag("missing") is not True:
                if y.isna().any().any():
                    raise ValueError(f"{self.__class__.__name__} cannot deal with missing values.")

            if y.shape[1] != 1:
                raise ValueError(f"y must contain exactly one target column. Got {y.shape[1]} columns.")

            if len(X) != len(y):
                raise ValueError(f"X and y must have the same number of rows. Got len(X)={len(X)}, len(y)={len(y)}.")

            if not X.index.equals(y.index):
                raise ValueError("X and y must have the same index.")

            n_samples = len(y)

        # Check sample_weight
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)

            if len(sample_weight) != n_samples:
                raise ValueError(f"sample_weight must have length {n_samples}. Got {len(sample_weight)}.")

            if np.any(sample_weight < 0):
                raise ValueError("sample_weight cannot contain negative values.")

        return X, y, sample_weight
