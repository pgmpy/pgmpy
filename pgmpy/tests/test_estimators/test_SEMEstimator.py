import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.estimators import IVEstimator, SEMEstimator
from pgmpy.models import SEM, SEMGraph

requires_torch = pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="execute only if required dependency present",
)


@pytest.fixture
def sem_models():
    config.set_backend("torch")
    custom = SEMGraph(
        ebunch=[("a", "b"), ("b", "c")], latents=[], err_corr=[], err_var={}
    )
    a = np.random.randn(10**3)
    b = a + np.random.normal(loc=0, scale=0.1, size=10**3)
    c = b + np.random.normal(loc=0, scale=0.2, size=10**3)
    custom_data = pd.DataFrame({"a": a, "b": b, "c": c})
    custom_data -= custom_data.mean(axis=0)

    demo = SEMGraph(
        ebunch=[
            ("xi1", "x1"),
            ("xi1", "x2"),
            ("xi1", "x3"),
            ("xi1", "eta1"),
            ("eta1", "y1"),
            ("eta1", "y2"),
            ("eta1", "y3"),
            ("eta1", "y4"),
            ("eta1", "eta2"),
            ("xi1", "eta2"),
            ("eta2", "y5"),
            ("eta2", "y6"),
            ("eta2", "y7"),
            ("eta2", "y8"),
        ],
        latents=["xi1", "eta1", "eta2"],
        err_corr=[
            ("y1", "y5"),
            ("y2", "y6"),
            ("y3", "y7"),
            ("y4", "y8"),
            ("y2", "y4"),
            ("y6", "y8"),
        ],
        err_var={},
    )

    demo_data = pd.read_csv(
        "pgmpy/tests/test_estimators/testdata/democracy1989a.csv",
        index_col=0,
        header=0,
    )

    union = SEMGraph(
        ebunch=[
            ("yrsmill", "unionsen"),
            ("age", "laboract"),
            ("age", "deferenc"),
            ("deferenc", "laboract"),
            ("deferenc", "unionsen"),
            ("laboract", "unionsen"),
        ],
        latents=[],
        err_corr=[("yrsmill", "age")],
        err_var={},
    )

    union_data = pd.read_csv(
        "pgmpy/tests/test_estimators/testdata/union1989b.csv", index_col=0, header=0
    )

    yield {
        "custom": custom,
        "custom_lisrel": custom.to_lisrel(),
        "custom_data": custom_data,
        "demo": demo,
        "demo_lisrel": demo.to_lisrel(),
        "demo_data": demo_data,
        "union": union,
        "union_lisrel": union.to_lisrel(),
        "union_data": union_data,
    }
    config.set_backend("numpy")


@pytest.fixture
def iv_models():
    model = SEM.from_graph(
        ebunch=[
            ("Z1", "X", 1.0),
            ("Z2", "X", 1.0),
            ("Z2", "W", 1.0),
            ("W", "U", 1.0),
            ("U", "X", 1.0),
            ("U", "Y", 1.0),
            ("X", "Y", 1.0),
        ],
        latents=["U"],
        err_var={"Z1": 1, "Z2": 1, "W": 1, "X": 1, "U": 1, "Y": 1},
    )
    return {
        "model": model,
        "generated_data": model.to_lisrel().generate_samples(100000),
    }


@requires_torch
def test_get_init_values(sem_models):
    demo_estimator = SEMEstimator(sem_models["demo"])
    for method in ["random", "std"]:
        B_init, zeta_init = demo_estimator.get_init_values(
            data=sem_models["demo_data"], method=method
        )

        m = len(sem_models["demo_lisrel"].eta)
        assert B_init.shape == (m, m)
        assert zeta_init.shape == (m, m)

        union_estimator = SEMEstimator(sem_models["union"])
        B_init, zeta_init = union_estimator.get_init_values(
            data=sem_models["union_data"], method=method
        )
        m = len(sem_models["union_lisrel"].eta)
        assert B_init.shape == (m, m)
        assert zeta_init.shape == (m, m)


@requires_torch
def test_union_estimator_random_init(sem_models):
    estimator = SEMEstimator(sem_models["union_lisrel"])
    estimator.fit(
        sem_models["union_data"],
        method="ml",
        opt="adam",
        max_iter=10**6,
        exit_delta=1e-1,
    )


@requires_torch
def test_custom_estimator_random_init(sem_models):
    estimator = SEMEstimator(sem_models["custom_lisrel"])
    estimator.fit(sem_models["custom_data"], method="ml", max_iter=10**6, opt="adam")
    estimator.fit(sem_models["custom_data"], method="uls", max_iter=10**6, opt="adam")
    estimator.fit(
        sem_models["custom_data"],
        method="gls",
        max_iter=10**6,
        opt="adam",
        W=np.ones((3, 3)),
    )


@requires_torch
def test_union_estimator_std_init(sem_models):
    estimator = SEMEstimator(sem_models["union_lisrel"])
    estimator.fit(
        sem_models["union_data"],
        method="ml",
        opt="adam",
        init_values="std",
        max_iter=10**6,
        exit_delta=1e-1,
    )


@requires_torch
def test_custom_estimator_std_init(sem_models):
    estimator = SEMEstimator(sem_models["custom_lisrel"])
    estimator.fit(
        sem_models["custom_data"],
        method="ml",
        init_values="std",
        max_iter=10**6,
        opt="adam",
    )


def test_iv_fit(iv_models):
    estimator = IVEstimator(iv_models["model"])
    param, summary = estimator.fit(X="X", Y="Y", data=iv_models["generated_data"])
    assert (param - 1) < 0.027
