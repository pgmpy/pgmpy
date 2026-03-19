from typing import Iterable, Optional, Tuple, Union

import pandas as pd
from skbase.base import BaseObject
from skbase.lookup import all_objects


class _BaseCITest(BaseObject):
    """
    Base class for all Conditional Independence (CI) tests.
    Subclasses must implement `_compute_statistic`.
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
        Z: Optional[Iterable] = None,
        boolean: bool = True,
        significance_level: float = 0.05,
        **kwargs,
    ):
        return self.test(
            X=X,
            Y=Y,
            Z=Z,
            boolean=boolean,
            significance_level=significance_level,
            **kwargs,
        )

    def test(
        self,
        X: str,
        Y: str,
        Z: Optional[Iterable] = None,
        boolean: bool = True,
        significance_level: float = 0.05,
        **kwargs,
    ) -> Union[bool, Tuple[float, ...]]:
        """
        Perform the conditional independence test.

        Parameters
        ----------
        X : str
            The first variable for testing the independence condition X ⊥⊥ Y | Z.
        Y : str
            The second variable for testing the independence condition X ⊥⊥ Y | Z.
        Z : list or array-like, optional
            A list of conditional variables for testing the condition X ⊥⊥ Y | Z.
            Default is None, which is treated as an empty list.
        boolean : bool, default=True
            If True, returns a boolean indicating independence (p-value >= significance_level).
            If False, returns the test statistic and p-value.
        significance_level : float, default=0.05
            The significance level for the test. Only used if boolean=True.
        **kwargs
            Additional arguments specific to the CI test implementation.

        Returns
        -------
        bool or tuple
            If boolean=True, returns True if p-value >= significance_level, else False.
            If boolean=False, returns a tuple of (test_statistic, p-value).

        Raises
        ------
        ValueError
            If inputs are invalid.
        """

        Z = [] if Z is None else list(Z)
        self._validate_inputs(X, Y, Z)

        result = self._compute_statistic(X=X, Y=Y, Z=Z, **kwargs)

        # p_value is always second element
        p_value = result[1]

        if boolean:
            return p_value >= significance_level

        return result

    def _validate_inputs(self, X, Y, Z):
        if X == Y:
            raise ValueError("X and Y must be different variables.")

        if self.get_tag("requires_data"):
            if not hasattr(self, "data") or not isinstance(self.data, pd.DataFrame):
                raise ValueError(
                    f"self.data must be a pandas.DataFrame. Got {type(getattr(self, 'data', None))}."
                )

            missing = ({X, Y} | set(Z)) - set(self.data.columns)
            if missing:
                raise ValueError(f"Missing columns in data: {missing}")

    def _compute_statistic(
        self,
        X: str,
        Y: str,
        Z: list,
        **kwargs,
    ) -> Tuple[float, ...]:
        raise NotImplementedError


def _instantiate_ci_test(cls, data=None):
    """
    Helper to instantiate a CI test class, respecting whether it requires data.
    """
    requires_data = cls.get_class_tag("requires_data", tag_value_default=True)
    if requires_data:
        if data is None:
            raise ValueError(
                f"CI test '{cls.__name__}' requires data, but data is None."
            )
        return cls(data=data)
    return cls()


def get_ci_test(test=None, data=None):
    """
    Retrieve CI test instance by name or infer default from data.
    """

    from pgmpy.utils import get_dataset_type

    if isinstance(test, _BaseCITest):
        return test

    if callable(test):
        return test

    if test is None:
        if data is None:
            raise ValueError(
                "Cannot determine CI test: both `test` and `data` are None."
            )

        var_type = get_dataset_type(data)

        tests = all_objects(
            object_types=_BaseCITest,
            package_name="pgmpy.ci_tests",
            return_names=False,
            filter_tags={"default_for": var_type},
        )

        if tests:
            return _instantiate_ci_test(tests[0], data=data)

        raise ValueError(f"No default CI test found for data type '{var_type}'.")

    if isinstance(test, str):
        tests = all_objects(
            object_types=_BaseCITest,
            package_name="pgmpy.ci_tests",
            return_names=False,
            filter_tags={"name": test.lower()},
        )

        if tests:
            return _instantiate_ci_test(tests[0], data=data)

        raise ValueError(f"Unknown CI test: {test!r}")

    raise ValueError(f"Invalid `test` argument: {test!r}")
