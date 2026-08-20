from skpro.regression.base._delegate import _DelegatedProbaRegressor

from pgmpy.parameter._base import BaseParameter


class SkproAdapter(_DelegatedProbaRegressor, BaseParameter):
    """
    Parameter adapter class for skpro regression models.

    Parameters
    ----------
    estimator : object

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.parameter.adapter.SkproAdapter import SkproAdapter
    >>> from skpro.regression.linear import GLMRegressor
    >>> rng = np.random.default_rng(seed=42)
    >>> n = 1000
    >>> A = rng.normal(size=n)
    >>> B = rng.normal(size=n)
    >>> C = rng.normal(size=n)
    >>> mean_y = 1 + 2 * A + 5 * B + 7 * C
    >>> X = pd.DataFrame(
    ...     {
    ...         "A": A,
    ...         "B": B,
    ...         "C": C,
    ...     }
    ... )
    >>> y = pd.DataFrame({"y": rng.normal(loc=mean_y, scale=3)})
    >>> cpd = SkproAdapter(GLMRegressor())
    >>> cpd.fit(X, y)
    SkproAdapter(estimator=GLMRegressor())
    >>> normal = cpd.predict_proba(X[:5])
    >>> normal # doctest: +SKIP
    Normal

    """

    _tags = {
        "parameter_type": "regressor",
        "produces_factor": False,
        "is_linear_gaussian": False,
        "supports_fit_joint": False,
        "missing": False,
        "can_be_root": False,
    }
    _delegate_name = "estimator"

    def __init__(self, estimator):
        self.estimator = estimator
        super().__init__()

    def _fit(self, X, y, sample_weight=None):

        estimator = self._get_delegate()
        estimator.fit(X=X, y=y, C=sample_weight)
        return self

    def _predict_proba(self, X):
        estimator = self._get_delegate()
        return estimator.predict_proba(X=X)
