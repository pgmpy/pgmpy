from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd
from skbase.base import BaseEstimator
from sklearn.base import clone


class CovarianceRegression(BaseEstimator):
    r"""
    Linear least-squares regressions between the columns of a dataset, computed from its covariance matrix.

    ``fit`` estimates the mean and the covariance matrix of the data once. The coefficients and the residual covariance
    of the linear regression of any set of columns :math:`y` on any other set of columns :math:`X` then follow from the
    covariance matrix alone, without another pass over the data:

    .. math::
        B = \Sigma_{yX} \Sigma_{XX}^{-1}, \qquad
        \Sigma_{y \mid X} = \Sigma_{yy} - B \Sigma_{Xy}.

    The regressors are standardized before solving, which keeps the system well conditioned when the variables differ
    in scale. When the regressors are singular (collinear or constant), the pseudo-inverse gives the minimum-norm fit,
    as in :class:`sklearn.linear_model.LinearRegression`.

    Parameters
    ----------
    covariance_estimator : sklearn covariance estimator, optional
        Estimator of the mean and the covariance matrix, for example :class:`sklearn.covariance.LedoitWolf` for a
        shrinkage estimate or :class:`sklearn.covariance.MinCovDet` for a robust one. It is cloned and fitted on the
        data, and its ``location_`` and ``covariance_`` are used. If None, the sample mean and the maximum likelihood
        (``ddof=0``) covariance are used, so that the regressions are ordinary least squares fits with an intercept.

    Attributes
    ----------
    covariance_ : numpy.ndarray of shape (n_variables, n_variables)
        Estimated covariance matrix of the variables.

    location_ : numpy.ndarray of shape (n_variables,)
        Estimated mean of the variables.

    n_samples_ : int
        Number of rows in the fitted data.

    feature_names_in_ : list
        Labels of the variables: the column labels of a fitted DataFrame, or the positions ``0, ..., n_variables - 1``
        of a fitted array.

    covariance_estimator_ : sklearn covariance estimator
        The fitted clone of ``covariance_estimator``. Only set when ``covariance_estimator`` is given.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.utils import CovarianceRegression
    >>> rng = np.random.default_rng(0)
    >>> data = pd.DataFrame({"A": rng.normal(size=1000), "B": rng.normal(size=1000)})
    >>> data["C"] = 2 * data["A"] - data["B"] + rng.normal(scale=0.5, size=1000)
    >>> regression = CovarianceRegression().fit(X=data)
    >>> regression.coefficients(X=["A", "B"], y=["C"]).round(1)
    array([[ 2., -1.]])
    >>> regression.residual_covariance(X=["A", "B"], y=["C"]).round(2)
    array([[0.24]])

    A shrinkage estimate of the covariance matrix:

    >>> from sklearn.covariance import LedoitWolf
    >>> shrunk = CovarianceRegression(covariance_estimator=LedoitWolf()).fit(X=data)
    >>> shrunk.coefficients(X=["A", "B"], y=["C"]).round(1)
    array([[ 2., -1.]])
    """

    def __init__(self, covariance_estimator=None):
        self.covariance_estimator = covariance_estimator
        super().__init__()

    def fit(self, X: pd.DataFrame | np.ndarray) -> CovarianceRegression:
        """
        Estimate the mean and the covariance matrix of `X`.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray of shape (n_samples, n_variables)
            Numeric data without missing values. The variables are referred to by the column labels of a DataFrame and
            by the column positions of an array.

        Returns
        -------
        self : CovarianceRegression
            The fitted instance.
        """
        # Step 1: Collect the variable labels and the values.
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            values = X.to_numpy(dtype=float)
        else:
            values = np.asarray(X, dtype=float)
            self.feature_names_in_ = list(range(values.shape[1]))
        self._positions = {variable: position for position, variable in enumerate(self.feature_names_in_)}

        # Step 2: Estimate the mean and the covariance matrix.
        if self.covariance_estimator is None:
            self.location_ = values.mean(axis=0)
            self.covariance_ = np.atleast_2d(np.cov(values, rowvar=False, ddof=0))
        else:
            self.covariance_estimator_ = clone(self.covariance_estimator).fit(values)
            self.location_ = self.covariance_estimator_.location_
            self.covariance_ = self.covariance_estimator_.covariance_
        self.n_samples_ = values.shape[0]

        self._is_fitted = True
        return self

    def coefficients(self, X: Sequence[Hashable], y: Sequence[Hashable]) -> np.ndarray:
        """
        Coefficients of the linear regression of the variables `y` on the variables `X`.

        Parameters
        ----------
        X : list
            Labels of the regressors.

        y : list
            Labels of the regressed variables.

        Returns
        -------
        numpy.ndarray
            Coefficients of shape ``(len(y), len(X))``, as ``coef_`` of :class:`sklearn.linear_model.LinearRegression`.
            The intercepts are the means of `y` minus the coefficients multiplied by the means of `X`.
        """
        x_positions = [self._positions[variable] for variable in X]
        y_positions = [self._positions[variable] for variable in y]

        scale = np.sqrt(self.covariance_[x_positions, x_positions])
        scale[scale == 0] = 1.0
        inverse_scale = 1.0 / scale[:, None]
        standardized = self.covariance_[np.ix_(x_positions, x_positions)] * inverse_scale * inverse_scale.T
        rhs = self.covariance_[np.ix_(x_positions, y_positions)] * inverse_scale
        try:
            return (np.linalg.solve(standardized, rhs) * inverse_scale).T
        except np.linalg.LinAlgError:
            return (np.linalg.pinv(standardized) @ rhs * inverse_scale).T

    def residual_covariance(self, X: Sequence[Hashable], y: Sequence[Hashable]) -> np.ndarray:
        """
        Covariance matrix of the residuals of the linear regression of the variables `y` on the variables `X`.

        Parameters
        ----------
        X : list
            Labels of the regressors. If empty, the covariance matrix of `y` is returned.

        y : list
            Labels of the regressed variables.

        Returns
        -------
        numpy.ndarray
            Residual covariance matrix of shape ``(len(y), len(y))``. With the default ``covariance_estimator``, it is
            the residual sum of squares and cross-products divided by ``n_samples_``.
        """
        y_positions = [self._positions[variable] for variable in y]
        total = self.covariance_[np.ix_(y_positions, y_positions)]
        if len(X) == 0:
            return total

        x_positions = [self._positions[variable] for variable in X]
        return total - self.covariance_[np.ix_(y_positions, x_positions)] @ self.coefficients(X=X, y=y).T
