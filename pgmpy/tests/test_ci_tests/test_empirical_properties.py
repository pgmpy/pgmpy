import numpy as np
import pandas as pd
import pytest

from pgmpy.ci_tests import (
    GCM,
    ChiSquare,
    FisherZ,
    GeneralizedCov,
    GSq,
    HotellingLawley,
    LogLikelihood,
    ModifiedLogLikelihood,
    Pearsonr,
    PearsonrEquivalence,
    PillaiTrace,
    PowerDivergence,
    RoysLargestRoot,
    WilksLambda,
)

ALPHA = 0.05
N_REPETITIONS = 20
MAX_FALSE_POSITIVE_RATE = 0.10
MIN_POWER = 0.85


def _empty_kwargs(seed: int) -> dict:
    return {}


def _generalized_cov_kwargs(seed: int) -> dict:
    return {
        "n_permutations": 30,
        "random_state": seed,
    }


def _simulate_discrete_conditional_independence(rng: np.random.Generator, dependent: bool) -> pd.DataFrame:
    n_samples = 800
    z = rng.binomial(1, 0.5, size=n_samples)
    x = np.where(z == 1, rng.binomial(1, 0.8, size=n_samples), rng.binomial(1, 0.2, size=n_samples))

    if dependent:
        logits = -1.5 + 3.0 * x + 2.0 * z
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))
    else:
        y = np.where(z == 1, rng.binomial(1, 0.8, size=n_samples), rng.binomial(1, 0.2, size=n_samples))

    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def _simulate_continuous_conditional_independence(rng: np.random.Generator, dependent: bool) -> pd.DataFrame:
    n_samples = 400
    z = rng.normal(size=n_samples)
    x = 0.9 * z + rng.normal(scale=1.0, size=n_samples)
    y = 0.9 * z + rng.normal(scale=1.0, size=n_samples)

    if dependent:
        y = y + 0.8 * x

    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def _simulate_mixed_dimensional_conditional_independence(rng: np.random.Generator, dependent: bool) -> pd.DataFrame:
    n_samples = 500
    z = rng.normal(size=n_samples)

    latent_x = 0.8 * z + rng.normal(scale=1.0, size=n_samples)
    x = pd.qcut(latent_x, 4, labels=["x1", "x2", "x3", "x4"])
    x_num = pd.Series(x).cat.codes.to_numpy() - 1.5

    y = 0.8 * z + rng.normal(scale=1.0, size=n_samples)
    if dependent:
        y = y + 0.9 * x_num

    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def _simulate_generalized_cov_conditional_independence(rng: np.random.Generator, dependent: bool) -> pd.DataFrame:
    n_samples = 500
    z = rng.normal(size=n_samples)

    x = 0.8 * z + rng.normal(scale=1.3, size=n_samples)
    latent_y = 0.8 * z + rng.normal(scale=1.3, size=n_samples)
    if dependent:
        latent_y = latent_y + 0.9 * x

    y = pd.qcut(latent_y, 3, labels=["y1", "y2", "y3"])
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def _simulate_equivalence_data(rng: np.random.Generator, practically_independent: bool) -> pd.DataFrame:
    n_samples = 1000
    z = rng.normal(size=n_samples)
    x = 0.9 * z + rng.normal(scale=1.0, size=n_samples)
    y = 0.9 * z + rng.normal(scale=1.0, size=n_samples)

    if not practically_independent:
        y = y + 0.3 * x

    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def _estimate_rejection_rates(ci_test_cls, simulate_data, kwargs_factory):
    false_rejections = 0
    true_rejections = 0

    for rep in range(N_REPETITIONS):
        null_seed = 1_000 + rep
        alt_seed = 5_000 + rep

        null_test = ci_test_cls(
            data=simulate_data(np.random.default_rng(null_seed), dependent=False),
            **kwargs_factory(null_seed),
        )
        false_rejections += not null_test("X", "Y", ["Z"], significance_level=ALPHA)

        alt_test = ci_test_cls(
            data=simulate_data(np.random.default_rng(alt_seed), dependent=True),
            **kwargs_factory(alt_seed),
        )
        true_rejections += not alt_test("X", "Y", ["Z"], significance_level=ALPHA)

    return false_rejections / N_REPETITIONS, true_rejections / N_REPETITIONS


STANDARD_CI_TESTS = [
    pytest.param(ChiSquare, _simulate_discrete_conditional_independence, _empty_kwargs, id="chi_square"),
    pytest.param(GSq, _simulate_discrete_conditional_independence, _empty_kwargs, id="g_sq"),
    pytest.param(LogLikelihood, _simulate_discrete_conditional_independence, _empty_kwargs, id="log_likelihood"),
    pytest.param(
        ModifiedLogLikelihood,
        _simulate_discrete_conditional_independence,
        _empty_kwargs,
        id="modified_log_likelihood",
    ),
    pytest.param(
        PowerDivergence,
        _simulate_discrete_conditional_independence,
        _empty_kwargs,
        id="power_divergence",
    ),
    pytest.param(Pearsonr, _simulate_continuous_conditional_independence, _empty_kwargs, id="pearsonr"),
    pytest.param(FisherZ, _simulate_continuous_conditional_independence, _empty_kwargs, id="fisher_z"),
    pytest.param(GCM, _simulate_continuous_conditional_independence, _empty_kwargs, id="gcm"),
    pytest.param(PillaiTrace, _simulate_mixed_dimensional_conditional_independence, _empty_kwargs, id="pillai"),
    pytest.param(
        WilksLambda,
        _simulate_mixed_dimensional_conditional_independence,
        _empty_kwargs,
        id="wilks_lambda",
    ),
    pytest.param(
        HotellingLawley,
        _simulate_mixed_dimensional_conditional_independence,
        _empty_kwargs,
        id="hotelling_lawley",
    ),
    pytest.param(
        RoysLargestRoot,
        _simulate_mixed_dimensional_conditional_independence,
        _empty_kwargs,
        id="roys_largest_root",
    ),
    pytest.param(
        GeneralizedCov,
        _simulate_generalized_cov_conditional_independence,
        _generalized_cov_kwargs,
        id="generalized_cov",
    ),
]


# IndependenceMatch is a deterministic lookup rather than a statistical test, so it
# is intentionally excluded from these empirical calibration/power checks.
@pytest.mark.parametrize(("ci_test_cls", "simulate_data", "kwargs_factory"), STANDARD_CI_TESTS)
def test_ci_tests_are_empirically_calibrated_and_powerful(ci_test_cls, simulate_data, kwargs_factory):
    false_positive_rate, power = _estimate_rejection_rates(ci_test_cls, simulate_data, kwargs_factory)

    assert false_positive_rate <= MAX_FALSE_POSITIVE_RATE
    assert power >= MIN_POWER


def test_pearsonr_equivalence_is_empirically_calibrated_and_powerful():
    false_positive_rate = 0
    power = 0

    for rep in range(N_REPETITIONS):
        null_seed = 9_000 + rep
        alt_seed = 13_000 + rep

        null_test = PearsonrEquivalence(
            data=_simulate_equivalence_data(np.random.default_rng(null_seed), practically_independent=False),
            delta_threshold=0.1,
        )
        false_positive_rate += null_test("X", "Y", ["Z"], significance_level=ALPHA)

        alt_test = PearsonrEquivalence(
            data=_simulate_equivalence_data(np.random.default_rng(alt_seed), practically_independent=True),
            delta_threshold=0.1,
        )
        power += alt_test("X", "Y", ["Z"], significance_level=ALPHA)

    assert false_positive_rate / N_REPETITIONS <= MAX_FALSE_POSITIVE_RATE
    assert power / N_REPETITIONS >= MIN_POWER
