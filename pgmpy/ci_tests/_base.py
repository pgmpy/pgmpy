from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from skbase.base import BaseObject
from skbase.lookup import all_objects
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class _ResidualMixin:
    """
    Mixin that provides a predict-then-residualize strategy for CI tests.

    Requires the using class to expose the following attributes before calling
    any method of this mixin:

    Attributes
    ----------
    data : pd.DataFrame
        The (preprocessed) dataset.
    dtypes : dict
        Mapping from column name to dtype code: ``"N"`` (numerical),
        ``"C"`` (categorical unordered), or ``"O"`` (categorical ordered),
        as returned by :func:`pgmpy.utils.preprocess_data`.
    estimator : sklearn-compatible estimator or None
        Estimator used to predict each variable from the conditioning set.
        Must support ``fit`` / ``predict``, and ``predict_proba`` for
        categorical targets. When ``None``, defaults to
        :class:`~sklearn.ensemble.RandomForestClassifier` for categorical /
        ordinal variables and :class:`~sklearn.ensemble.RandomForestRegressor`
        for numerical variables.
    """

    def _fit_predict(self, target_col: str, Z_data: pd.DataFrame):
        """
        Fit an estimator on ``Z_data`` to predict ``target_col``.

        Parameters
        ----------
        target_col : str
            Name of the column in ``self.data`` to predict.
        Z_data : pd.DataFrame
            Feature matrix (already one-hot encoded) used for fitting.

        Returns
        -------
        pred : np.ndarray
            Predicted values (1-D for numerical, 2-D probability matrix for
            categorical / ordinal targets).
        cat_index : pd.CategoricalIndex or None
            Category index returned by :func:`pandas.factorize`; ``None`` for
            numerical targets.
        """
        is_cat = self.dtypes[target_col] in ("C", "O")
        target_data = self.data.loc[:, target_col]
        cat_index = None

        if self.estimator is not None:
            model = clone(self.estimator)
            if is_cat and not hasattr(model, "predict_proba"):
                raise ValueError(
                    f"The provided estimator ({type(model).__name__}) must have a "
                    f"`predict_proba` method for discrete variable '{target_col}'."
                )
        else:
            model_cls = RandomForestClassifier if is_cat else RandomForestRegressor
            model = model_cls(random_state=0)

        if is_cat:
            y_encoded, cat_index = pd.factorize(target_data)
            model.fit(Z_data, y_encoded)
            pred = model.predict_proba(Z_data)
        else:
            model.fit(Z_data, target_data)
            pred = model.predict(Z_data)

        return pred, cat_index, model

    def get_residuals(self, X: str, Z: list) -> pd.DataFrame | pd.Series:
        """
        Compute residuals of ``X`` after regressing out ``Z``.

        For categorical / ordinal ``X``: returns a :class:`~pandas.DataFrame`
        of shape ``(n, K-1)`` — one-hot encoded dummy matrix minus predicted
        class probabilities with the last column dropped to avoid
        multicollinearity.

        For numerical ``X``: returns a :class:`~pandas.Series` of length ``n``
        — observed values minus predicted values.

        An intercept column of ones is always appended to ``Z`` so the
        estimator always has at least one feature and learns the baseline
        of the target variable.

        Parameters
        ----------
        X : str
            Name of the variable to residualize.
        Z : list of str
            Conditioning variables. May be empty.

        Returns
        -------
        pd.DataFrame or pd.Series
            Residuals of ``X`` given ``Z``.
        """
        z_cols = list(Z) + ["_intercept_Z"]
        z_data_source = self.data.assign(_intercept_Z=np.ones(self.data.shape[0]))

        Z_data = pd.get_dummies(z_data_source.loc[:, z_cols])
        pred, cat_index, estimator = self._fit_predict(X, Z_data)

        if self.dtypes[X] in ("C", "O"):
            dummies = pd.get_dummies(self.data.loc[:, X]).loc[:, cat_index.categories[cat_index.codes]]
            residual = (dummies - pred).iloc[:, :-1]
        else:
            residual = self.data.loc[:, X] - pred

        return residual, estimator


@dataclass(frozen=True)
class _CITestResult:
    statistic: float | None
    p_value: float
    effect_size: float | None = None
    attributes: dict[str, object] = field(default_factory=dict)


class _BaseCITest(BaseObject):
    """
    Base class for all Conditional Independence (CI) tests.

    Subclasses must implement ``_compute_result``.
    """

    _tags = {
        "name": None,
        "data_types": (),
        "default_for": None,
        "requires_data": True,
        "is_symmetric": True,
    }

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._result_cache = {}
        super().__init__()

    def __call__(
        self,
        X: str,
        Y: str,
        Z: list | tuple = (),
        significance_level: float = 0.05,
    ):
        return self.is_independent(X=X, Y=Y, Z=Z, significance_level=significance_level)

    def is_independent(
        self,
        X: str,
        Y: str,
        Z: list | tuple = (),
        significance_level: float = 0.05,
    ) -> bool:
        """
        Perform the conditional independence test and return a boolean result.

        Parameters
        ----------
        X : str
            The first variable for testing the independence condition X ⊥⊥ Y | Z.
        Y : str
            The second variable for testing the independence condition X ⊥⊥ Y | Z.
        Z : list or tuple
            A list of conditional variables for testing the condition X ⊥⊥ Y | Z.
            Default is an empty tuple.
        significance_level : float, default=0.05
            The significance level for the test.

        Returns
        -------
        bool
            True if X _|_ Y | Z (p_value_ >= significance_level), else False.

        Raises
        ------
        ValueError
            If inputs are invalid.

        Notes
        -----
        Always sets ``self.statistic_``, ``self.p_value_``, and ``self.effect_size_``
        as side effects, regardless of the return value. Access these attributes to
        inspect raw results.
        CI test instances are not thread-safe; use a separate instance per thread
        for parallel computation.
        """
        self.run_test(X=X, Y=Y, Z=list(Z))

        return self.p_value_ >= significance_level

    def run_test(self, X, Y, Z):
        """
        Run the statistical test and return the test statistic and p-value.

        Uses the subclass-provided ``_compute_result`` hook and updates the
        instance result attributes from the returned payload.

        Parameters
        ----------
        X : str
            The first variable for testing the independence condition X ⊥⊥ Y | Z.
        Y : str
            The second variable for testing the independence condition X ⊥⊥ Y | Z.
        Z : list
            A list of conditional variables for testing the condition X ⊥⊥ Y | Z.

        Returns
        -------
        statistic : float
            The test statistic.
        p_value : float
            The p-value for the test.
        """
        self._validate_inputs(X, Y, Z)
        Z = list(Z)

        if self.use_cache:
            if self.get_tag("is_symmetric", tag_value_default=True):
                x_key, y_key = sorted((X, Y), key=repr)
            else:
                x_key, y_key = X, Y
            cache_key = (x_key, y_key, tuple(sorted(Z, key=repr)))

            result = self._result_cache.get(cache_key)
            if result is None:
                result = self._compute_result(X=X, Y=Y, Z=Z)
                self._result_cache[cache_key] = result
        else:
            result = self._compute_result(X=X, Y=Y, Z=Z)

        self.statistic_ = result.statistic
        self.p_value_ = result.p_value
        self.effect_size_ = result.effect_size
        for attr_name, attr_value in result.attributes.items():
            setattr(self, attr_name, attr_value)

        return self.statistic_, self.p_value_

    def _compute_result(self, X, Y, Z):
        """
        Compute the CI test result for a single query.

        Subclasses must implement this method and return a ``_CITestResult``.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _compute_result.")

    def _validate_inputs(self, X, Y, Z):
        if X == Y:
            raise ValueError("X and Y must be different variables.")

        if not isinstance(Z, (list, tuple, set)):
            raise ValueError(f"Z must be a list or tuple. Got {type(Z)}.")

        if X in Z or Y in Z:
            raise ValueError(f"X and Y cannot appear in Z. Found {X if X in Z else Y} in Z.")


def get_ci_test(test=None, data=None, use_cache=True):
    """
    Return an instantiated CI test object given a test name, instance, or data.

    This is the recommended factory for obtaining a CI test. It supports four
    calling patterns:

    1. **Pass-through**: if ``test`` is already a :class:`_BaseCITest` instance, it is
       returned as-is.
    2. **Callable**: if ``test`` is any other callable (e.g. a custom function), it is
       returned as-is.
    3. **By name**: if ``test`` is a string, the registered CI test whose ``name`` tag
       matches (case-insensitive) is instantiated and returned.
    4. **Auto-detect**: if ``test`` is ``None``, the default CI test for the data type
       inferred from ``data`` is instantiated and returned.

    Parameters
    ----------
    test : str, _BaseCITest instance, callable, or None
        The CI test to retrieve. If a string, must match the ``name`` tag of a
        registered CI test (e.g. ``"chi_square"``, ``"pearsonr"``). If ``None``,
        the default test for the data type of ``data`` is used.
    data : pandas.DataFrame or None
        The dataset to pass to the CI test constructor. Required when ``test`` is
        ``None`` or when the resolved test has ``requires_data=True``.

    Returns
    -------
    _BaseCITest or callable
        An instantiated CI test object ready to call, or the original callable if
        ``test`` was already callable.

    Raises
    ------
    ValueError
        If ``test`` is ``None`` and ``data`` is also ``None``.
    ValueError
        If ``test`` is a string that does not match any registered CI test name.
    ValueError
        If the resolved CI test requires data but ``data`` is ``None``.
    ValueError
        If ``test`` is not a string, ``_BaseCITest`` instance, callable, or ``None``.

    Examples
    --------
    Get the default CI test for a continuous dataset (returns :class:`Pearsonr`):

    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.ci_tests import ChiSquare, Pearsonr
    >>> rng = np.random.default_rng(seed=42)
    >>> data = pd.DataFrame(data=rng.standard_normal(size=(100, 3)), columns=["X", "Y", "Z"])
    >>> test = get_ci_test(data=data)
    >>> isinstance(test, Pearsonr)
    True

    Get a CI test by name:

    >>> test = get_ci_test(test="chi_square", data=data)
    >>> isinstance(test, ChiSquare)
    True

    Pass an already-instantiated CI test (returned unchanged):

    >>> existing = Pearsonr(data=data)
    >>> get_ci_test(test=existing) is existing
    True

    Pass any callable (e.g. a custom function) and it is returned unchanged:

    >>> def my_ci_test(X, Y, Z, significance_level=0.05):
    ...     return True
    ...
    >>> get_ci_test(test=my_ci_test) is my_ci_test
    True
    """

    from pgmpy.utils import get_dataset_type

    if isinstance(test, _BaseCITest):
        return test

    if callable(test):
        return test

    if test is None:
        if data is None:
            raise ValueError("Cannot determine CI test: both `test` and `data` are None.")

        var_type = get_dataset_type(data)
        filter_tags = {"default_for": var_type}

    elif isinstance(test, str):
        filter_tags = {"name": test.lower()}
    else:
        raise ValueError(f"Invalid `test` argument: {test!r}")

    tests = all_objects(
        object_types=_BaseCITest,
        package_name="pgmpy.ci_tests",
        return_names=False,
        filter_tags=filter_tags,
    )

    if tests:
        cls = tests[0]
        if cls.get_class_tag("requires_data", tag_value_default=True):
            if data is None:
                raise ValueError(f"CI test '{cls.__name__}' requires data, but data is None.")
            return cls(data=data, use_cache=use_cache)
        else:
            return cls(use_cache=use_cache)
    raise ValueError(f"Unknown CI test: {test!r}")
