import numpy as np
import pandas as pd
from scipy.stats import differential_entropy
from skbase.base import BaseObject
from skbase.lookup import all_objects

from pgmpy.ci_tests import get_ci_test


class BaseANMScore(BaseObject):
    """
    Base class for ANM direction scores.

    Subclasses implement :meth:`__call__` with signature ``(x, y) -> float`` where a smaller value
    indicates that ``x`` and ``y`` are more independent. Hyperparameters are configured through the
    subclass ``__init__``, and the ``name`` tag is the string key used to select the score via
    :func:`get_anm_score`.
    """

    _tags = {
        "name": None,
    }

    def __call__(self, x, y) -> float:
        raise NotImplementedError


class IndependenceScore(BaseANMScore):
    """
    Dependence score from a conditional-independence test.

    Runs an unconditional CI test between ``x`` and ``y`` and returns one of its outputs as the
    score. The score is always oriented so that **a smaller value means the variables are more
    independent**.

    Parameters
    ----------
    ci_test : str or pgmpy.ci_tests.BaseCITest, default="pearsonr"
        The independence test, resolved via :func:`pgmpy.ci_tests.get_ci_test`.

    criterion : {"effect_size", "statistic", "p_value"}, default="effect_size"
        Which output of the CI test to use as the score, transformed so that a smaller score means
        more independent:

        - ``"effect_size"`` -- the test's ``effect_size_`` (a non-negative dependence magnitude);
        - ``"statistic"`` -- the absolute test statistic ``|statistic_|``;
        - ``"p_value"`` -- the negative p-value ``-p_value_`` (a larger p-value, i.e. more evidence
          of independence, gives a smaller score).
    """

    _tags = {
        "name": "independence",
    }

    def __init__(self, ci_test="pearsonr", criterion="effect_size"):
        self.ci_test = ci_test
        self.criterion = criterion
        super().__init__()

    def __call__(self, x, y) -> float:
        data = pd.DataFrame({"_x": np.asarray(x), "_y": np.asarray(y)})
        test = get_ci_test(test=self.ci_test, data=data)
        test.run_test("_x", "_y", Z=[])

        if self.criterion == "effect_size":
            return test.effect_size_
        if self.criterion == "statistic":
            return abs(test.statistic_)
        if self.criterion == "p_value":
            return -test.p_value_
        raise ValueError(
            f"Unknown criterion: {self.criterion!r}. Must be one of 'effect_size', 'statistic', 'p_value'."
        )


class EntropyScore(BaseANMScore):
    """
    Differential-entropy score, ``H(x) + H(y)`` :cite:p:`mooij_2016`.

    A smaller value means a better-fitting direction. The parameters are forwarded to
    :func:`scipy.stats.differential_entropy`.

    Parameters
    ----------
    method : {"auto", "vasicek", "van es", "ebrahimi", "correa"}, default="auto"
        Differential-entropy estimator.

    window_length : int, optional
        Window length for the spacing-based estimators (default chosen by scipy).

    base : float, optional
        Logarithm base for the entropy (default: natural log).
    """

    _tags = {
        "name": "entropy",
    }

    def __init__(self, method="auto", window_length=None, base=None):
        self.method = method
        self.window_length = window_length
        self.base = base
        super().__init__()

    def __call__(self, x, y) -> float:
        kwargs = {"method": self.method, "base": self.base}

        # This is required due to API changes in scipy 1.16.
        if self.window_length is not None:
            kwargs["window_length"] = self.window_length

        return float(differential_entropy(np.asarray(x), **kwargs) + differential_entropy(np.asarray(y), **kwargs))


class GaussScore(BaseANMScore):
    """
    Gaussian (log-variance) score, ``log Var(x) + log Var(y)`` :cite:p:`mooij_2016`.

    The Gaussian special case of :class:`EntropyScore` (the differential entropy of a variable is
    upper-bounded by ``0.5 * log(2 * pi * e * Var)``, with equality for a Gaussian). A smaller value
    means a better-fitting direction.

    Consistent only for additive-noise models with (near-)Gaussian noise. When identifiability stems
    from non-Gaussian noise -- e.g. a linear mechanism with non-Gaussian noise -- this score is
    unreliable; prefer :class:`EntropyScore` or :class:`IndependenceScore`.
    """

    _tags = {
        "name": "gauss",
    }

    def __call__(self, x, y) -> float:
        return float(np.log(np.var(np.asarray(x))) + np.log(np.var(np.asarray(y))))


def get_anm_score(score):
    """
    Resolve ``score`` to a callable scoring object for the ANM estimator.

    Parameters
    ----------
    score : str, BaseANMScore, or callable
        - a registered score name (the ``name`` tag of a :class:`BaseANMScore`, e.g.
          ``"independence"``, ``"entropy"``, ``"gauss"``), resolved to a default-configured instance;
        - a :class:`BaseANMScore` instance, returned as-is;
        - any callable ``fn(x, y) -> float``, returned as-is.

    Returns
    -------
    callable
        An object callable as ``score(x, y) -> float``.

    Raises
    ------
    ValueError
        If ``score`` is a string that does not name a registered score, or is neither a callable nor
        a registered name.
    """
    if isinstance(score, BaseANMScore):
        return score

    if isinstance(score, str):
        scores = all_objects(
            object_types=BaseANMScore,
            package_name="pgmpy.causal_discovery",
            return_names=False,
            filter_tags={"name": score.lower()},
        )
        if scores:
            return scores[0]()
        raise ValueError(
            f"Unknown score: {score!r}. Must be a registered score name, a BaseANMScore "
            "instance, or a callable of the form fn(x, y) -> float."
        )

    if callable(score) and not isinstance(score, type):
        return score

    raise ValueError(
        f"Invalid score: {score!r}. Must be a registered score name, a BaseANMScore "
        "instance, or a callable of the form fn(x, y) -> float."
    )
