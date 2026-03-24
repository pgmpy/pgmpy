# This extension template provides instructions to add new conditional independence (CI)
# tests to pgmpy.
#
# Please follow the following steps:
# 1. Copy this file to `pgmpy/estimators/` and rename it as `your_ci_test_name.py`
#    (e.g., `my_ci_test.py`).
#    Note: Do NOT start the filename with an underscore `_`, otherwise it won't be discovered.
# 2. Go through the file and address all the TODOs.
# 3. Register your test by importing and calling `ci_registry.register` in
#    `pgmpy/estimators/CITests.py`, or add the `@ci_registry.register` decorator directly
#    in that file if your test is small enough to live there.
# 4. Add an import of your function in `pgmpy/estimators/__init__.py`.
# 5. Add the function name to the `__all__` list in `pgmpy/estimators/__init__.py`.
# 6. If you would like to contribute the CI test to pgmpy, please add tests in
#    `pgmpy/tests/test_estimators/test_CITests.py`.


# TODO: Add any other necessary imports here.
import numpy as np  # noqa: F401
import pandas as pd
from scipy import stats  # noqa: F401

from pgmpy.estimators.CITests import ci_registry


# TODO: Fill in the decorator arguments:
#   - `name`       : A unique string identifier for your test (lowercase, underscores allowed).
#                    This is the string users will pass as `ci_test="your_name"`.
#   - `data_types` : A list of data types your test supports.
#                    Choose from: "discrete", "continuous", "mixed".
#   - `is_default` : Set to True ONLY if this test should become the default for one of
#                    the listed data types. Leave False unless you have a strong reason.
@ci_registry.register(
    name="my_ci_test",  # TODO: Replace with your test's unique name.
    data_types=["discrete"],  # TODO: Replace with the appropriate data type(s).
    is_default=False,
)
def my_ci_test(X, Y, Z, data, boolean=True, **kwargs):
    """
    [One line description of the CI test.]

    [Detailed description of the test. Explain the null hypothesis, the statistic
    being computed, and any important assumptions (e.g., sample size, data type).]

    The null hypothesis for this test is: X is independent of Y given Z.

    Parameters
    ----------
    X : int, str, or any hashable object
        A variable name contained in the dataset.

    Y : int, str, or any hashable object
        A variable name contained in the dataset, different from X.

    Z : list or array-like
        A list of variable names contained in the dataset, different from X and Y.
        This is the conditioning set that (potentially) makes X and Y independent.
        Pass an empty list `[]` for an unconditional independence test.

    data : pandas.DataFrame
        The dataset on which to test the independence condition. Each column
        corresponds to a variable; rows are observations.

    boolean : bool, default=True
        If True, an additional keyword argument `significance_level` must be
        provided via `**kwargs`. Returns True if the p-value is greater than or
        equal to `significance_level` (i.e., fail to reject independence),
        otherwise returns False.

        If False, returns the raw test statistic, p-value, and degrees of freedom.

    **kwargs
        significance_level : float
            Required when `boolean=True`. Threshold for rejecting the null
            hypothesis (e.g., 0.05).
        # TODO: Document any additional keyword arguments your test requires.

    Returns
    -------
    result : bool or tuple
        If boolean=True  : returns True if p_value >= significance_level, else False.
        If boolean=False : returns a tuple (statistic, p_value, dof).
            - statistic : float -- the computed test statistic.
            - p_value   : float -- the p-value for the test.
            - dof       : int   -- the degrees of freedom used in the test.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(
    ...     np.random.randint(0, 2, size=(10000, 3)), columns=["A", "B", "C"]
    ... )
    >>> # TODO: Replace with a realistic working example for your test.
    >>> my_ci_test(X="A", Y="B", Z=[], data=data, boolean=True, significance_level=0.05)
    True
    >>> my_ci_test(X="A", Y="B", Z=["C"], data=data, boolean=False)
    (statistic_value, p_value, dof)

    References
    ----------
    .. [1] TODO: Add the primary citation for this test.
    .. [2] TODO: Add additional references if applicable.
    """
    # ------------------------------------------------------------------
    # Step 1: Validate and normalise inputs.
    # ------------------------------------------------------------------
    if not hasattr(Z, "__iter__"):
        raise ValueError(f"Z must be an iterable (e.g., a list). Got type: {type(Z)}")
    Z = list(Z)

    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"data must be a pandas.DataFrame. Got type: {type(data)}")

    if (X in Z) or (Y in Z):
        raise ValueError(f"X and Y must not appear in Z. Found {X if X in Z else Y} in Z.")

    # TODO: Add any additional input validation specific to your test here.
    # For example, checking that the data is of the expected type (discrete/continuous).

    # ------------------------------------------------------------------
    # Step 2: Compute the test statistic.
    # ------------------------------------------------------------------

    # TODO: Replace the block below with your actual test implementation.
    #
    # Common patterns:
    #
    #   Unconditional test (len(Z) == 0):
    #       Compute the statistic directly on data[X] and data[Y].
    #
    #   Conditional test (len(Z) > 0):
    #       Iterate over the unique states of Z (for discrete data), or
    #       partial out the effect of Z via regression (for continuous data),
    #       then aggregate the per-stratum statistics and degrees of freedom.
    #
    # See `chi_square` or `pearsonr` in pgmpy/estimators/CITests.py for reference.

    if len(Z) == 0:
        # TODO: Implement the unconditional independence test here.
        statistic = 0.0
        dof = 1
        p_value = 1.0
    else:
        # TODO: Implement the conditional independence test here.
        statistic = 0.0
        dof = 1
        p_value = 1.0

    # ------------------------------------------------------------------
    # Step 3: Return the result in the requested format.
    # ------------------------------------------------------------------
    if boolean:
        return p_value >= kwargs["significance_level"]
    else:
        return statistic, p_value, dof
