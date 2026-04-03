from skbase.base import BaseObject
from skbase.lookup import all_objects


class _BaseCITest(BaseObject):
    """
    Base class for all Conditional Independence (CI) tests. Subclasses must implement `run_test`.
    """

    _tags = {
        "name": None,
        "data_types": (),
        "default_for": None,
        "requires_data": True,
    }

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
        Always sets ``self.statistic_`` and ``self.p_value_`` as side effects,
        regardless of the return value. Access these attributes to inspect raw results.
        CI test instances are not thread-safe; use a separate instance per thread
        for parallel computation.
        """
        self._validate_inputs(X, Y, Z)
        self.run_test(X=X, Y=Y, Z=list(Z))

        return self.p_value_ >= significance_level

    def run_test(self, X, Y, Z):
        """
        Run the statistical test and return the test statistic and p-value.

        Subclasses must implement this method. It should set ``self.statistic_``
        and ``self.p_value_`` as attributes, and may set additional attributes
        (e.g. ``self.dof_``).

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
        raise NotImplementedError(f"{self.__class__.__name__} must implement run_test.")

    def _validate_inputs(self, X, Y, Z):
        if X == Y:
            raise ValueError("X and Y must be different variables.")

        if not isinstance(Z, (list, tuple, set)):
            raise ValueError(f"Z must be a list or tuple. Got {type(Z)}.")

        if X in Z or Y in Z:
            raise ValueError(f"X and Y cannot appear in Z. Found {X if X in Z else Y} in Z.")


def get_ci_test(test=None, data=None):
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
            return cls(data=data)
        return cls()
    raise ValueError(f"Unknown CI test: {test!r}")
