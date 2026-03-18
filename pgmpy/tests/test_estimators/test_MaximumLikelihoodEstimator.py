import numpy as np
import pandas as pd
import pytest
from joblib.externals.loky import get_reusable_executor
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors import FactorDict
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork, JunctionTree


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

    m2 = JunctionTree()
    m2.add_nodes_from([("A", "B")])
    m3 = JunctionTree()
    m3.add_edges_from([(("A", "C"), ("B", "C"))])

    d1 = pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0]})
    d2 = pd.DataFrame(
        data={
            "A": [0, np.nan, 1],
            "B": [0, 1, 0],
            "C": [1, 1, np.nan],
            "D": [np.nan, "Y", np.nan],
        }
    )
    # Use Example from ML Machine Learning - A Probabilistic Perspective
    # Section 19.5.7.1.
    d3 = pd.DataFrame(
        data={
            "A": [0] * 43 + [0] * 9 + [1] * 44 + [1] * 4,
            "B": [0] * 43 + [1] * 9 + [0] * 44 + [1] * 4,
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
    potentials1 = FactorDict.from_dataframe(df=d3, marginals=m2.nodes)
    m2.clique_beliefs = potentials1

    potentials2 = FactorDict.from_dataframe(df=d1, marginals=m3.nodes)
    m3.clique_beliefs = potentials2

    mle1 = MaximumLikelihoodEstimator(m1, d1)
    mle2 = MaximumLikelihoodEstimator(model=m2, data=d3)
    mle3 = MaximumLikelihoodEstimator(model=m3, data=d1)

    yield {
        "m1": m1,
        "model_latents": model_latents,
        "data_latents": data_latents,
        "m2": m2,
        "m3": m3,
        "d1": d1,
        "d2": d2,
        "d3": d3,
        "cpds": cpds,
        "potentials1": potentials1,
        "potentials2": potentials2,
        "mle1": mle1,
        "mle2": mle2,
        "mle3": mle3,
    }

    get_reusable_executor().shutdown(wait=True)


def test_error_latent_model(setup_data, backend):
    data = setup_data
    with pytest.raises(ValueError):
        MaximumLikelihoodEstimator(data["model_latents"], data["data_latents"])


def test_get_parameters_incomplete_data(setup_data, backend):
    data = setup_data
    assert data["mle1"].estimate_cpd("A") == data["cpds"][0]
    assert data["mle1"].estimate_cpd("B") == data["cpds"][1]
    assert data["mle1"].estimate_cpd("C") == data["cpds"][2]
    assert len(data["mle1"].get_parameters(n_jobs=1)) == 3


def test_estimate_cpd(setup_data, backend):
    data = setup_data
    assert data["mle1"].estimate_cpd("A") == data["cpds"][0]
    assert data["mle1"].estimate_cpd("B") == data["cpds"][1]
    assert data["mle1"].estimate_cpd("C") == data["cpds"][2]


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
    mle2 = MaximumLikelihoodEstimator(m, d)
    assert mle2.estimate_cpd("B") == cpd_b


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
    mle2 = MaximumLikelihoodEstimator(m, d)
    assert mle2.estimate_cpd("Color") == color_cpd


def test_class_init(setup_data, backend):
    data = setup_data
    mle = MaximumLikelihoodEstimator(
        data["m1"], data["d1"], state_names={"A": [0, 1], "B": [0, 1], "C": [0, 1]}
    )
    assert mle.estimate_cpd("A") == data["cpds"][0]
    assert mle.estimate_cpd("B") == data["cpds"][1]
    assert mle.estimate_cpd("C") == data["cpds"][2]
    assert len(mle.get_parameters(n_jobs=1)) == 3


def test_nonoccurring_values(setup_data, backend):
    data = setup_data
    mle = MaximumLikelihoodEstimator(
        data["m1"],
        data["d1"],
        state_names={"A": [0, 1, 23], "B": [0, 1], "C": [0, 42, 1], 1: [2]},
    )
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
    assert mle.estimate_cpd("A") == cpds[0]
    assert mle.estimate_cpd("B") == cpds[1]
    assert mle.estimate_cpd("C") == cpds[2]
    assert len(mle.get_parameters(n_jobs=1)) == 3


def test_missing_data(setup_data, backend):
    data = setup_data
    e1 = MaximumLikelihoodEstimator(data["m1"], data["d2"], state_names={"C": [0, 1]})
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
    assert e1.estimate_cpd("A") == cpds1[0]
    assert e1.estimate_cpd("B") == cpds1[1]
    assert e1.estimate_cpd("C") == cpds1[2]
    assert len(e1.get_parameters(n_jobs=1)) == 3


def test_estimate_potentials_smoke_test(setup_data, backend):
    data = setup_data
    joint = data["mle3"].estimate_potentials().product()
    assert joint.marginalize(variables=["B"], inplace=False) == data["potentials2"][
        ("A", "C")
    ].normalize(inplace=False)
    assert joint.marginalize(variables=["A"], inplace=False) == data["potentials2"][
        ("B", "C")
    ].normalize(inplace=False)


def test_partition_function(setup_data, backend):
    data = setup_data
    model = data["m3"].copy()
    model.clique_beliefs = data["mle3"].estimate_potentials()
    assert model.get_partition_function() == 1.0


def test_estimate_potentials(setup_data, backend):
    data = setup_data
    assert data["mle2"].estimate_potentials()[("A", "B")] == data["potentials1"][
        ("A", "B")
    ].normalize(inplace=False)
