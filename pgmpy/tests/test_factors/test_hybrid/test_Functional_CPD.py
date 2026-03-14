import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

from pgmpy import config
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.hybrid import FunctionalCPD
from pgmpy.models.LinearGaussianBayesianNetwork import LinearGaussianBayesianNetwork

dist = _safe_import("pyro.distributions", pkg_name="pyro-ppl")
torch = _safe_import("torch")

pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies("pyro-ppl", severity="none"),
    reason="execute only if required dependency present",
)


@pytest.fixture(autouse=True)
def set_torch():
    config.set_backend("torch")
    yield
    config.set_backend("numpy")


def test_class_init():
    """
    Test the initialization of the FunctionalCPD class.
    """
    cpd = FunctionalCPD(
        variable="x3",
        fn=lambda parent_sample: dist.Normal(
            1.0 + 0.2 * parent_sample["x1"] + 0.3 * parent_sample["x2"], 1
        ),
        parents=["x1", "x2"],
    )

    assert cpd.variable == "x3"
    assert cpd.parents == ["x1", "x2"]
    assert callable(cpd.fn), "The function passed to FunctionalCPD must be callable."


def test_linear_gaussian():
    """
    Test the equivalence of FunctionalCPD with LinearGaussianCPD sampling.
    """
    x1_cpd = LinearGaussianCPD("x1", [0], 1.0)

    x2_cpd = LinearGaussianCPD("x2", [0], 1.0)

    x3_cpd = LinearGaussianCPD(
        "x3",
        [1.0, 0.2, 0.3],
        1.0,
        evidence=["x1", "x2"],
    )
    num_samples = 10000

    lgbn = LinearGaussianBayesianNetwork([("x1", "x3"), ("x2", "x3")])
    lgbn.add_cpds(x1_cpd, x2_cpd, x3_cpd)

    linear_gaussian_samples = lgbn.simulate(num_samples, seed=42)

    functional_cpd = FunctionalCPD(
        variable="x3",
        fn=lambda parent_sample: dist.Normal(
            1.0 + 0.2 * parent_sample["x1"] + 0.3 * parent_sample["x2"], 1
        ),
        parents=["x1", "x2"],
    )

    functional_samples = functional_cpd.sample(
        num_samples, linear_gaussian_samples[["x1", "x2"]]
    )

    # functional_mean = functional_samples.mean()
    # functional_variance = functional_samples.var()
    # linear_gaussian_mean = linear_gaussian_samples["x3"].mean()
    # linear_gaussian_variance = linear_gaussian_samples["x3"].var()

    tolerance = 1e-1

    assert functional_samples.mean() == pytest.approx(
        linear_gaussian_samples["x3"].mean(), abs=tolerance
    )
    assert functional_samples.var() == pytest.approx(
        linear_gaussian_samples["x3"].var(), abs=tolerance
    )


def test_different_distributions():
    exp_cpd = FunctionalCPD("exponential", lambda _: dist.Exponential(rate=2.0))

    exp_samples = exp_cpd.sample(n_samples=5000)
    assert np.all(exp_samples >= 0)
    assert np.mean(exp_samples) == pytest.approx(0.5, abs=0.1)

    uni_cpd = FunctionalCPD(
        "uniform",
        lambda parent: dist.Uniform(
            low=parent["exponential"], high=parent["exponential"] + 5
        ),
        parents=["exponential"],
    )

    exp_samples = pd.DataFrame({"exponential": exp_samples})

    uni_samples = uni_cpd.sample(n_samples=5000, parent_sample=exp_samples)

    assert np.all(uni_samples >= exp_samples["exponential"])
    assert np.all(uni_samples <= exp_samples["exponential"] + 5)


def test_sample_vectorized():
    """
    Test FunctionalCPD with vectorized sampling.
    """

    def vectorized_fn(parent_sample):
        x1 = torch.tensor(parent_sample["x1"].values, dtype=torch.float32)
        x2 = torch.tensor(parent_sample["x2"].values, dtype=torch.float32)
        mean = 1.0 + 0.5 * x1 + 0.25 * x2
        return dist.Normal(mean, torch.ones_like(mean))

    cpd = FunctionalCPD(
        variable="x3", fn=vectorized_fn, parents=["x1", "x2"], vectorized=True
    )

    parent_samples = pd.DataFrame(
        {"x1": np.random.randn(1000), "x2": np.random.randn(1000)}
    )

    samples = cpd.sample(n_samples=1000, parent_sample=parent_samples)
    assert len(samples) == 1000
    assert np.isfinite(samples).all()


def test_sample_iterative():
    """
    Test FunctionalCPD with iterative sampling (vectorized=False).
    """

    def row_fn(row):
        mean = 1.0 + 0.5 * row["x1"] + 0.25 * row["x2"]
        return dist.Normal(mean, 1.0)

    cpd = FunctionalCPD(
        variable="x3", fn=row_fn, parents=["x1", "x2"], vectorized=False
    )

    parent_samples = pd.DataFrame(
        {"x1": np.random.randn(1000), "x2": np.random.randn(1000)}
    )

    samples = cpd.sample(n_samples=1000, parent_sample=parent_samples)
    assert len(samples) == 1000
    assert np.isfinite(samples).all()


def test_vectorized_without_parent():
    """
    Test FunctionalCPD with vectorized sampling without parents.
    """

    def vectorized_fn(_):
        return dist.Normal(torch.zeros(1000), torch.ones(1000))

    cpd = FunctionalCPD(variable="z", fn=vectorized_fn, parents=[], vectorized=True)
    samples = cpd.sample(n_samples=1000)
    assert len(samples) == 1000
    assert np.isfinite(samples).all()


def test_iterative_without_parent():
    """
    Test FunctionalCPD with iterative sampling (vectorized=False) without parents.
    """

    def iterative_fn(_):
        return dist.Normal(0.0, 1.0)

    cpd = FunctionalCPD(variable="z", fn=iterative_fn, parents=[], vectorized=False)
    samples = cpd.sample(n_samples=1000)
    assert len(samples) == 1000
    assert np.isfinite(samples).all()
