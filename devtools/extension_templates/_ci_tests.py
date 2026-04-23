# This extension template provides instructions to add new Conditional Independence (CI) tests to pgmpy.

# Please follow the following steps:
# 1. Copy this file to `pgmpy/ci_tests` and rename it as `your_ci_test.py`.
# 2. Go through this file and address all the TODOs.
# 3. Add an import in `pgmpy/ci_tests/__init__.py`.
# 4. Add tests in `pgmpy/tests/test_ci_tests/test_your_ci_test.py`.

# TODO: Add necessary imports here (e.g., numpy, scipy, sklearn, etc.)
import pandas as pd

from pgmpy.ci_tests._base import _BaseCITest, _CITestResult


class YourCITest(_BaseCITest):
    """
    [One-line description of the CI test]

    [Detailed description of the test. Please include test statistic and p-value equations if possible]

    Parameters
    ----------
    data : pandas.DataFrame, optional
        Dataset used for CI testing. Required if `requires_data=True`.

    param1 : type, optional
        Description of hyperparameter.

    param2 : type, optional
        Description of hyperparameter.

    Attributes
    ----------
    statistic_ : float
        Test statistic computed during the test.

    p_value_ : float
        p-value corresponding to the test statistic.

    Examples
    --------
    >>> import pandas as pd
    >>> from pgmpy.ci_tests import YourCITest
    >>> data = pd.DataFrame(...)
    >>> test = YourCITest(data=data)
    >>> test("X", "Y", ["Z"], significance_level=0.05)
    # Show the output

    References
    ----------
    .. [1] Add reference for the CI test
    """

    # TODO: Required: Metadata used for registering the CI test.
    _tags = {
        "name": "your_ci_test",  # Unique name allowing users to find this test by name.
        "data_types": ("continuous",),  # Can be combination of ("discrete", "continuous", "mixed").
        "default_for": None,  # If specified, this test becomes the default for the specified data type.
        "requires_data": True,  # False for tests that don’t use data.
        "is_symmetric": True,  # Set to False only if swapping X and Y can change the result.
    }

    def __init__(self, data: pd.DataFrame = None, param1=None, param2=None, use_cache: bool = True):
        # TODO: Store all inputs as attributes. Add any validation if required by parameters.
        self.data = data
        self.param1 = param1
        self.param2 = param2

        super().__init__(use_cache=use_cache)

    def _compute_result(
        self,
        X: str,
        Y: str,
        Z: list,
    ):
        """
        Compute the test statistic and p-value payload for a single query.

        Parameters
        ----------
        X : str
            First variable
        Y : str
            Second variable
        Z : list
            Conditioning variables

        Returns
        -------
        _CITestResult
            Result payload consumed by `_BaseCITest.run_test`.
        """
        # TODO: Add logic for computing the test statistic and p-value. Can use self.data, and other params.
        statistic = None
        p_value = None

        # TODO: If you need to expose extra result attributes (for example `dof_`),
        # return them in the `attributes` dict. The base class will project them onto
        # the instance after cache lookup / execution.
        return _CITestResult(statistic=statistic, p_value=p_value, attributes={})
