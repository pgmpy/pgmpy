from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.special import psi
from scipy.stats import differential_entropy
from skbase.base import BaseObject
from skbase.lookup import all_objects

from pgmpy.ci_tests import BaseCITest, get_ci_test


def _spacing_entropy(values: np.typing.ArrayLike, base: float | None = None) -> float:
    """Estimate differential entropy using consecutive sorted spacings."""
    if base is not None and (base <= 0 or base == 1):
        raise ValueError("base must be positive and not equal to 1.")

    values = np.sort(np.asarray(values))
    if values.size < 2:
        raise ValueError("Entropy estimation requires at least two observations.")

    deltas = np.diff(values)
    if np.any(deltas == 0):
        raise ValueError("Spacing entropy requires distinct observations.")

    entropy = psi(values.size) - psi(1) + np.log(deltas).sum() / (values.size - 1)
    if base is not None:
        entropy /= np.log(base)
    return entropy


class BaseBivariateScore(BaseObject):
    """Base class for scores that compare two one-dimensional samples.

    Subclasses are called as ``score(x, y)`` and return a float. A smaller score indicates the
    preferred direction. Each subclass defines ``name`` and ``supported_algorithms`` tags for
    built-in lookup.
    """

    _tags = {"name": None, "supported_algorithms": []}

    def __call__(self, x: np.typing.ArrayLike, y: np.typing.ArrayLike) -> float:
        raise NotImplementedError


class IndependenceScore(BaseBivariateScore):
    """
    Dependence score from a conditional-independence test.

    Runs an unconditional CI test between ``x`` and ``y``. A smaller score means weaker dependence.

    Parameters
    ----------
    ci_test : str or pgmpy.ci_tests.BaseCITest, default="pearsonr"
        The independence test, resolved via :func:`pgmpy.ci_tests.get_ci_test`.
        The test must provide the output selected by ``criterion``.

    criterion : {"effect_size", "statistic", "p_value"}, default="effect_size"
        Which CI-test output to use. Each option is transformed so that a smaller value means
        weaker dependence:

        - ``"effect_size"`` returns the test's non-negative dependence magnitude.
        - ``"statistic"`` returns the absolute test statistic.
        - ``"p_value"`` returns the negative p-value.
    """

    _tags = {"name": "independence", "supported_algorithms": ["anm", "pnl"]}

    def __init__(self, ci_test: str | BaseCITest = "pearsonr", criterion: str = "effect_size") -> None:
        self.ci_test = ci_test
        self.criterion = criterion
        super().__init__()

    def __call__(self, x: np.typing.ArrayLike, y: np.typing.ArrayLike) -> float:
        data = pd.DataFrame({"_x": np.asarray(x), "_y": np.asarray(y)})
        if isinstance(self.ci_test, BaseCITest):
            test = self.ci_test.clone().set_params(data=data)
        else:
            test = get_ci_test(test=self.ci_test, data=data)
        test.run_test("_x", "_y", Z=[])

        if self.criterion == "effect_size":
            score = test.effect_size_
        elif self.criterion == "statistic":
            score = abs(test.statistic_)
        elif self.criterion == "p_value":
            score = -test.p_value_
        else:
            raise ValueError(
                f"Unknown criterion: {self.criterion!r}. Must be one of 'effect_size', 'statistic', 'p_value'."
            )
        return score


class EntropyScore(BaseBivariateScore):
    """
    Differential-entropy score, ``H(x) + H(y)`` :cite:p:`mooij_2016`.

    A smaller value means a better-fitting direction. The parameters are forwarded to
    :func:`scipy.stats.differential_entropy`.

    Parameters
    ----------
    method : {"auto", "vasicek", "van es", "ebrahimi", "correa"}, default="auto"
        Differential-entropy estimator.

    window_length : int, optional
        Window length for the spacing-based estimators. The default is chosen by SciPy.

    base : float, optional
        Logarithm base for the entropy. The default uses the natural logarithm.
    """

    _tags = {"name": "entropy", "supported_algorithms": ["anm"]}

    def __init__(
        self,
        method: str = "auto",
        window_length: int | None = None,
        base: float | None = None,
    ) -> None:
        self.method = method
        self.window_length = window_length
        self.base = base
        super().__init__()

    def __call__(self, x: np.typing.ArrayLike, y: np.typing.ArrayLike) -> float:
        entropy_kwargs = {
            "method": self.method,
            "base": self.base,
        }
        if self.window_length is not None:
            entropy_kwargs["window_length"] = self.window_length
        return differential_entropy(x, **entropy_kwargs) + differential_entropy(y, **entropy_kwargs)


class EntropyDifferenceScore(BaseBivariateScore):
    """
    Difference of marginal entropies, ``H(y) - H(x)`` :cite:p:`mooij_2016`.

    Parameters
    ----------
    method : {"spacing", "auto", "vasicek", "van es", "ebrahimi", "correa"}, default="spacing"
        Entropy estimator. ``"spacing"`` uses consecutive sorted spacings; other values are passed
        to :func:`scipy.stats.differential_entropy`.

    window_length : int, optional
        Window length for SciPy estimators. Must be ``None`` when ``method="spacing"``.

    base : float, optional
        Logarithm base for the entropy. The default uses the natural logarithm.
    """

    _tags = {"name": "entropy", "supported_algorithms": ["igci"]}

    def __init__(
        self,
        method: str = "spacing",
        window_length: int | None = None,
        base: float | None = None,
    ) -> None:
        self.method = method
        self.window_length = window_length
        self.base = base
        super().__init__()

    def __call__(self, x: np.typing.ArrayLike, y: np.typing.ArrayLike) -> float:
        if self.method == "spacing":
            if self.window_length is not None:
                raise ValueError("window_length is not supported by the spacing estimator.")
            return _spacing_entropy(y, base=self.base) - _spacing_entropy(x, base=self.base)

        entropy_kwargs = {
            "method": self.method,
            "base": self.base,
        }
        if self.window_length is not None:
            entropy_kwargs["window_length"] = self.window_length
        return differential_entropy(y, **entropy_kwargs) - differential_entropy(x, **entropy_kwargs)


class GaussScore(BaseBivariateScore):
    """
    Gaussian (log-variance) score, ``log Var(x) + log Var(y)`` :cite:p:`mooij_2016`.

    The Gaussian special case of :class:`EntropyScore`. A smaller value means a better-fitting
    direction. This score is unreliable when identifiability depends on non-Gaussian noise; prefer
    :class:`EntropyScore` or :class:`IndependenceScore` in that case.
    """

    _tags = {"name": "gauss", "supported_algorithms": ["anm"]}

    def __call__(self, x: np.typing.ArrayLike, y: np.typing.ArrayLike) -> float:
        return np.log(np.var(x)) + np.log(np.var(y))


class SlopeScore(BaseBivariateScore):
    """
    Log-slope score for IGCI :cite:p:`mooij_2016`.

    Assumes continuous numeric inputs. Sorts observations by ``x`` and then ``y``, and averages ``y``
    within groups of repeated ``x`` values. Following Equation 21, each neighboring log slope is
    weighted by the multiplicity of its left ``x`` value. Zero ``y`` spacings are ignored.
    """

    _tags = {"name": "slope", "supported_algorithms": ["igci"]}

    def __call__(self, x: np.typing.ArrayLike, y: np.typing.ArrayLike) -> float:
        x = np.asarray(x)
        y = np.asarray(y)

        order = np.lexsort((y, x))
        x = x[order]
        y = y[order]

        x, run_start, multiplicities = np.unique(x, return_index=True, return_counts=True)
        if x.size < 2:
            raise ValueError("SlopeScore requires at least two distinct x values.")
        y = np.add.reduceat(y, run_start) / multiplicities

        x_diff = np.diff(x)
        y_diff = np.diff(y)
        valid = y_diff != 0
        if not valid.any():
            raise ValueError("SlopeScore requires at least one pair with non-zero x and y spacings.")

        return np.average(
            np.log(np.abs(y_diff[valid] / x_diff[valid])),
            weights=multiplicities[:-1][valid],
        )


def get_bivariate_score(
    score: str | BaseBivariateScore | Callable[[np.typing.ArrayLike, np.typing.ArrayLike], float],
    algorithm: str,
) -> BaseBivariateScore | Callable[[np.typing.ArrayLike, np.typing.ArrayLike], float]:
    """Return a score selected by name or supplied by the user.

    Parameters
    ----------
    score : str or callable
        Built-in score name, configured score object, or custom callable.
    algorithm : {"anm", "igci", "pnl"}
        Causal discovery algorithm that will use the score.

    Returns
    -------
    callable
        Resolved score callable.

    Raises
    ------
    ValueError
        If ``score`` is an unknown name or is not callable.
    """
    algorithm = algorithm.lower()

    if isinstance(score, BaseBivariateScore):
        if algorithm not in score.get_tag("supported_algorithms"):
            raise ValueError(f"{type(score).__name__} does not support {algorithm.upper()}.")
        return score

    if isinstance(score, str):
        score_classes = all_objects(
            object_types=BaseBivariateScore,
            package_name="pgmpy.causal_discovery",
            return_names=False,
            filter_tags={
                "name": score.lower(),
                "supported_algorithms": algorithm,
            },
        )
        if score_classes:
            return score_classes[0]()
        raise ValueError(f"Unknown {algorithm.upper()} score: {score!r}.")

    if callable(score) and not isinstance(score, type):
        return score

    raise ValueError(f"Invalid {algorithm.upper()} score: {score!r}. Pass a built-in name or callable.")
