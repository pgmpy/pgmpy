import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

from pgmpy.parameterization import PyroAdapter, PyroNUTS, PyroSVI, TabularMLE

pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies(["pyro-ppl", "skpro"], severity="none"),
    reason="execute only if required dependency present",
)

pyro = _safe_import("pyro", pkg_name="pyro-ppl")
dist = _safe_import("pyro.distributions", pkg_name="pyro-ppl")


def unknown_mean(parents):
    return dist.Normal(pyro.sample("mu", dist.Normal(0.0, 10.0)), 1.0)


Y = pd.Series([0.5, 1.5, 1.0, 2.0], name="y")


def test_tags():
    # Local estimators of a PyroAdapter, which need Pyro.
    for cls, name in ((PyroSVI, "pyro_svi"), (PyroNUTS, "pyro_nuts")):
        assert (cls.get_class_tag("object_type"), cls.get_class_tag("name")) == ("local_estimator", name)
        assert cls.get_class_tag("parameterization") == ["pyro_adapter"]
        assert cls.get_class_tag("supports_weighted_data") is True
        assert cls.get_class_tag("python_dependencies") == "pyro-ppl"


def test_construction_errors():
    for cls, name in ((PyroSVI, "num_samples"), (PyroSVI, "num_steps"), (PyroNUTS, "num_samples")):
        with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
            cls(**{name: 0})
    with pytest.raises(ValueError, match="warmup_steps must be"):
        PyroNUTS(warmup_steps=-1)
    with pytest.raises(TypeError, match="Pyro estimator"):
        PyroAdapter(unknown_mean, estimator=TabularMLE())


def test_diagnostics():
    # Each estimator reports its own diagnostics: SVI the loss after each step, NUTS Pyro's MCMC diagnostics.
    svi = PyroAdapter(unknown_mean, estimator=PyroSVI(num_steps=20, num_samples=10, random_state=0)).fit(None, Y)
    assert svi.diagnostics_["losses"].shape == (20,)
    nuts = PyroAdapter(unknown_mean, estimator=PyroNUTS(num_samples=20, warmup_steps=20, random_state=0)).fit(None, Y)
    assert nuts.posterior_samples_["mu"].shape == (20,)
    assert {"n_eff", "r_hat"} <= set(nuts.diagnostics_["mu"])


def test_weights_follow_the_estimator():
    # A PyroAdapter takes sample weights only if its estimator does.
    class UnweightedSVI(PyroSVI):
        _tags = {"name": "unweighted_svi", "supports_weighted_data": False}

    adapter = PyroAdapter(unknown_mean, estimator=UnweightedSVI())
    assert adapter.get_tag("supports_weighted_data") is False
    with pytest.raises(ValueError, match="does not support sample_weight"):
        adapter.fit(None, Y, sample_weight=[1.0, 2.0, 1.0, 1.0])
