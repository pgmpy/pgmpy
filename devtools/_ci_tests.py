# This extension template provides instructions to add new Conditional Independence (CI) tests to pgmpy.

# Please follow the following steps:
# 1. Copy this file to `pgmpy/ci_tests` and rename it as `your_ci_test.py`.
# 2. Go through the file and address all the TODOs.
# 3. Add an import in `pgmpy/ci_tests/__init__.py`.
# 4. Add tests in `pgmpy/tests/test_ci_tests/test_your_ci_test.py`.

# TODO: Add necessary imports (e.g., numpy, scipy, sklearn, etc.)
import pandas as pd

from pgmpy.ci_tests._base import _BaseCITest


class YourCITest(_BaseCITest):
    """
    [One-line description of the CI test]

    [Detailed description of the test]

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
        P-value corresponding to the test statistic.

    Examples
    --------
    >>> import pandas as pd
    >>> from pgmpy.estimators.CITests import YourCITest
    >>> data = pd.DataFrame(...)
    >>> test = YourCITest(data=data)
    >>> test("X", "Y", ["Z"], significance_level=0.05)

    References
    ----------
    .. [1] Add reference for the CI test
    """

    # TODO: Required: Metadata used for registering the CI test
    _tags = {
        "name": "your_ci_test",
        "data_types": ("continuous",),  # ("discrete", "continuous", "mixed")
        "default_for": None,  # Set if this should be default for a data type
        "requires_data": True,  # False for tests that don’t use data
    }

    def __init__(self, data: pd.DataFrame = None, param1=None, param2=None):
        """
        Initialize the CI test.

        TODO:
        - Store all parameters as attributes
        - Validate inputs if required
        """
        self.data = data
        self.param1 = param1
        self.param2 = param2

        super().__init__()

    def run_test(
        self,
        X: str,
        Y: str,
        Z: list,
    ):
        """
        Compute the test statistic and p-value.

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
        statistic : float
        p_value : float
        """

        # TODO: Access data if required and do required augmentations
        # data = self.data

        # TODO: Implement the CI test logic here
        # This may include:
        # - preprocessing
        # - computing a test statistic
        # - computing a p-value

        statistic = None
        p_value = None

        # Required: Store results
        self.statistic_ = statistic
        self.p_value_ = p_value

        return self.statistic_, self.p_value_
