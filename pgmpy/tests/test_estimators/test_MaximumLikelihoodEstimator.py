import numpy as np
import pandas as pd
import pytest
from joblib.externals.loky import get_reusable_executor
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import MaximumLikelihoodEstimator


def get_cpd(estimator, variable):
    return next(cpd for cpd in estimator.parameters_ if cpd.variable == variable)


@pytest.fixture(params=["numpy", "torch"])
def backend(request):
    if request.param == "torch":
        if not _check_soft_dependencies("torch", severity="none"):
            pytest.skip("torch not installed")
        config.set_backend("torch")
    yield request.param
    config.set_backend("numpy")


@pytest.fixture
def setup_data():
    m1 = DiscreteBayesianNetwork([("A", "C"), ("B", "C")])
    model_latents = DiscreteBayesianNetwork([("A", "C"), ("B", "C")], latents=["C"])

    data_latents = pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0]})

    d1 = pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0]})
    d2 = pd.DataFrame(
        data={
            "A": [0, np.nan, 1],
            "B": [0, 1, 0],
            "C": [1, 1, np.nan],
            "D": [np.nan, "Y", np.nan],
        }
    )
    cpds = [
        TabularCPD("A", 2, [[2.0 / 3], [1.0 / 3]]),
        TabularCPD("B", 2, [[2.0 / 3], [1.0 / 3]]),
        TabularCPD(
            "C",
            2,
            [[0.0, 0.0, 1.0, 0.5], [1.0, 1.0, 0.0, 0.5]],
            evidence=["A", "B"],
            evidence_card=[2, 2],
        ),
    ]

    mle1 = MaximumLikelihoodEstimator().fit(m1, d1)

    yield {
        "m1": m1,
        "model_latents": model_latents,
        "data_latents": data_latents,
        "d1": d1,
        "d2": d2,
        "cpds": cpds,
        "mle1": mle1,
    }

    get_reusable_executor().shutdown(wait=True)


def test_error_latent_model(setup_data, backend):
    data = setup_data
    with pytest.raises(ValueError):
        MaximumLikelihoodEstimator().fit(data["model_latents"], data["data_latents"])


def test_fit_sets_fitted_attributes(setup_data, backend):
    data = setup_data
    estimator = MaximumLikelihoodEstimator()

    assert estimator.fit(data["m1"], data["d1"]) is estimator
    assert estimator.state_names_ == {"A": [0, 1], "B": [0, 1], "C": [0, 1]}


def test_parameters_incomplete_data(setup_data, backend):
    data = setup_data
    assert get_cpd(data["mle1"], "A") == data["cpds"][0]
    assert get_cpd(data["mle1"], "B") == data["cpds"][1]
    assert get_cpd(data["mle1"], "C") == data["cpds"][2]
    assert len(data["mle1"].parameters_) == 3


def test_state_names1(backend):
    m = DiscreteBayesianNetwork([("A", "B")])
    d = pd.DataFrame(data={"A": [2, 3, 8, 8, 8], "B": ["X", "O", "X", "O", "X"]})
    cpd_b = TabularCPD(
        "B",
        2,
        [[0, 1, 1.0 / 3], [1, 0, 2.0 / 3]],
        evidence=["A"],
        evidence_card=[3],
        state_names={"A": [2, 3, 8], "B": ["O", "X"]},
    )
    mle2 = MaximumLikelihoodEstimator().fit(m, d)
    assert get_cpd(mle2, "B") == cpd_b


def test_state_names2(backend):
    m = DiscreteBayesianNetwork([("Light?", "Color"), ("Fruit", "Color")])
    d = pd.DataFrame(
        data={
            "Fruit": ["Apple", "Apple", "Apple", "Banana", "Banana"],
            "Light?": [True, True, False, False, True],
            "Color": ["red", "green", "black", "black", "yellow"],
        }
    )
    color_cpd = TabularCPD(
        "Color",
        4,
        [[1, 0, 1, 0], [0, 0.5, 0, 0], [0, 0.5, 0, 0], [0, 0, 0, 1]],
        evidence=["Fruit", "Light?"],
        evidence_card=[2, 2],
        state_names={
            "Color": ["black", "green", "red", "yellow"],
            "Light?": [False, True],
            "Fruit": ["Apple", "Banana"],
        },
    )
    mle2 = MaximumLikelihoodEstimator().fit(m, d)
    assert get_cpd(mle2, "Color") == color_cpd


def test_class_init(setup_data, backend):
    data = setup_data
    mle = MaximumLikelihoodEstimator(state_names={"A": [0, 1], "B": [0, 1], "C": [0, 1]})
    assert mle.fit(data["m1"], data["d1"]) is mle
    assert get_cpd(mle, "A") == data["cpds"][0]
    assert get_cpd(mle, "B") == data["cpds"][1]
    assert get_cpd(mle, "C") == data["cpds"][2]
    assert len(mle.parameters_) == 3


def test_fit_does_not_mutate_constructor_state_names(setup_data, backend):
    data = setup_data
    supplied_state_names = {"A": [0, 1], "B": [0, 1], "C": [0, 1]}

    estimator = MaximumLikelihoodEstimator(state_names=supplied_state_names).fit(data["m1"], data["d1"])
    estimator.state_names_["A"].append(2)

    assert supplied_state_names == {"A": [0, 1], "B": [0, 1], "C": [0, 1]}


def test_nonoccurring_values(setup_data, backend):
    data = setup_data
    mle = MaximumLikelihoodEstimator(
        state_names={"A": [0, 1, 23], "B": [0, 1], "C": [0, 42, 1], 1: [2]},
    ).fit(data["m1"], data["d1"])
    cpds = [
        TabularCPD("A", 3, [[2.0 / 3], [1.0 / 3], [0]], state_names={"A": [0, 1, 23]}),
        TabularCPD("B", 2, [[2.0 / 3], [1.0 / 3]], state_names={"B": [0, 1]}),
        TabularCPD(
            "C",
            3,
            [
                [0.0, 0.0, 1.0, 1.0 / 3, 1.0 / 3, 1.0 / 3],
                [0.0, 0.0, 0.0, 1.0 / 3, 1.0 / 3, 1.0 / 3],
                [1.0, 1.0, 0.0, 1.0 / 3, 1.0 / 3, 1.0 / 3],
            ],
            evidence=["A", "B"],
            evidence_card=[3, 2],
            state_names={"A": [0, 1, 23], "B": [0, 1], "C": [0, 42, 1]},
        ),
    ]
    assert get_cpd(mle, "A") == cpds[0]
    assert get_cpd(mle, "B") == cpds[1]
    assert get_cpd(mle, "C") == cpds[2]
    assert len(mle.parameters_) == 3


def test_missing_data(setup_data, backend):
    data = setup_data
    e1 = MaximumLikelihoodEstimator(state_names={"C": [0, 1]}).fit(data["m1"], data["d2"])
    cpds1 = [
        TabularCPD("A", 2, [[0.5], [0.5]]),
        TabularCPD("B", 2, [[2.0 / 3], [1.0 / 3]]),
        TabularCPD(
            "C",
            2,
            [[0, 0.5, 0.5, 0.5], [1, 0.5, 0.5, 0.5]],
            evidence=["A", "B"],
            evidence_card=[2, 2],
        ),
    ]
    assert get_cpd(e1, "A") == cpds1[0]
    assert get_cpd(e1, "B") == cpds1[1]
    assert get_cpd(e1, "C") == cpds1[2]
    assert len(e1.parameters_) == 3
