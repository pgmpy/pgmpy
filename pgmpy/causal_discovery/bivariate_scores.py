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
    preferred direction. Every bivariate score sets all of the tags listed by
    ``pgmpy.registry.all_tags("bivariate_score")``. See :doc:`/api/tags` for what each tag means.
    """

    _tags = {
        "object_type": "bivariate_score",
        "name": None,
        "input_type": None,
        "data_types": [],
        "assumption:linearity": None,
        "assumption:additive_noise": None,
        "assumption:gaussian_noise": None,
    }

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

    _tags = {
        "name": "independence",
        "input_type": "cause_residual",
        "data_types": ["continuous"],
        "assumption:linearity": False,
        "assumption:additive_noise": False,
        "assumption:gaussian_noise": False,
    }

    def __init__(self, ci_test: str | BaseCITest = "pearsonr", criterion: str = "effect_size") -> None:
        self.ci_test = ci_test
        self.criterion = criterion
        super().__init__()

        if isinstance(ci_test, BaseCITest):
            test_tags = ci_test.get_tags()
        else:
            test_classes = all_objects(
                object_types=BaseCITest,
                package_name="pgmpy.ci_tests",
                return_names=False,
                filter_tags={"name": ci_test.lower()},
            )
            test_tags = test_classes[0].get_class_tags() if test_classes else {}
        self.set_tags(
            **{
                key: test_tags[key]
                for key in self.get_tags()
                if (key == "data_types" or key.startswith("assumption:")) and key in test_tags
            }
        )

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

    _tags = {
        "name": "entropy",
        "input_type": "cause_residual",
        "data_types": ["continuous"],
        "assumption:linearity": False,
        "assumption:additive_noise": False,
        "assumption:gaussian_noise": False,
    }

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

    _tags = {
        "name": "entropy_difference",
        "input_type": "cause_effect",
        "data_types": ["continuous"],
        "assumption:linearity": False,
        "assumption:additive_noise": False,
        "assumption:gaussian_noise": False,
    }

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

    _tags = {
        "name": "gauss",
        "input_type": "cause_residual",
        "data_types": ["continuous"],
        "assumption:linearity": False,
        "assumption:additive_noise": False,
        "assumption:gaussian_noise": True,
    }

    def __call__(self, x: np.typing.ArrayLike, y: np.typing.ArrayLike) -> float:
        return np.log(np.var(x)) + np.log(np.var(y))


class SlopeScore(BaseBivariateScore):
    """
    Log-slope score for IGCI :cite:p:`mooij_2016`.

    Assumes continuous numeric inputs. Sorts observations by ``x`` and then ``y``, and averages ``y``
    within groups of repeated ``x`` values. Following Equation 21, each neighboring log slope is
    weighted by the multiplicity of its left ``x`` value. Zero ``y`` spacings are ignored.
    """

    _tags = {
        "name": "slope",
        "input_type": "cause_effect",
        "data_types": ["continuous"],
        "assumption:linearity": False,
        "assumption:additive_noise": False,
        "assumption:gaussian_noise": False,
    }

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
    input_type: str,
) -> BaseBivariateScore | Callable[[np.typing.ArrayLike, np.typing.ArrayLike], float]:
    """Return a score selected by name or supplied by the user.

    Parameters
    ----------
    score : str, BaseBivariateScore, or callable
        Built-in score name, configured score object, or custom callable. Custom callables are returned unchanged.

    input_type : {"cause_residual", "cause_effect"}
        What the caller passes to the score. Built-in scores must have a matching ``input_type`` tag.

    Returns
    -------
    callable
        Resolved score callable.

    Raises
    ------
    ValueError
        If ``score`` is an unknown name, a built-in score of a different input type, or not callable.
    """
    input_descr = {"cause_residual": "(cause, residual)", "cause_effect": "(cause, effect)"}
    valid_scores = all_objects(
        object_types=BaseBivariateScore,
        package_name="pgmpy.causal_discovery",
        return_names=False,
        filter_tags={"input_type": input_type},
    )
    valid_names = ", ".join(sorted(cls.get_class_tag("name") for cls in valid_scores))

    if isinstance(score, BaseBivariateScore):
        score_input_type = score.get_tag("input_type")
        if score_input_type != input_type:
            raise ValueError(
                f"{type(score).__name__} takes {input_descr[score_input_type]} inputs, but {input_descr[input_type]} "
                f"inputs are required. Use one of: {valid_names}."
            )
        return score

    if isinstance(score, str):
        for cls in valid_scores:
            if cls.get_class_tag("name") == score.lower():
                return cls()
        raise ValueError(f"Unknown score for {input_descr[input_type]} inputs: {score!r}. Use one of: {valid_names}.")

    if callable(score) and not isinstance(score, type):
        return score

    raise ValueError(f"Invalid score: {score!r}. Pass a built-in name, a BaseBivariateScore instance, or a callable.")
