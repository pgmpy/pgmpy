import numpy as np
import pandas as pd
import pytest
from joblib.externals.loky import get_reusable_executor
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.factors import FactorDict
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork, JunctionTree
from pgmpy.parameter_estimator import DiscreteMLE


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

    mle1 = DiscreteMLE().fit(m1, d1)

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
    }

    get_reusable_executor().shutdown(wait=True)


def test_error_latent_model(setup_data, backend):
    data = setup_data
    with pytest.raises(ValueError):
        DiscreteMLE().fit(data["model_latents"], data["data_latents"])


def test_get_parameters_incomplete_data(setup_data, backend):
    data = setup_data
    assert get_cpd(data["mle1"], "A") == data["cpds"][0]
    assert get_cpd(data["mle1"], "B") == data["cpds"][1]
    assert get_cpd(data["mle1"], "C") == data["cpds"][2]
    assert len(data["mle1"].parameters_) == 3


def test_estimate_cpd(setup_data, backend):
    data = setup_data
    assert get_cpd(data["mle1"], "A") == data["cpds"][0]
    assert get_cpd(data["mle1"], "B") == data["cpds"][1]
    assert get_cpd(data["mle1"], "C") == data["cpds"][2]


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
    mle2 = DiscreteMLE().fit(m, d)
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
    mle2 = DiscreteMLE().fit(m, d)
    assert get_cpd(mle2, "Color") == color_cpd


def test_class_init(setup_data, backend):
    data = setup_data
    mle = DiscreteMLE(state_names={"A": [0, 1], "B": [0, 1], "C": [0, 1]}).fit(data["m1"], data["d1"])
    assert get_cpd(mle, "A") == data["cpds"][0]
    assert get_cpd(mle, "B") == data["cpds"][1]
    assert get_cpd(mle, "C") == data["cpds"][2]
    assert len(mle.parameters_) == 3


def test_nonoccurring_values(setup_data, backend):
    data = setup_data
    mle = DiscreteMLE(
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
    e1 = DiscreteMLE(state_names={"C": [0, 1]}).fit(data["m1"], data["d2"])
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


def test_sample_weight_matches_row_duplication(setup_data):
    data = setup_data
    m1, d1 = data["m1"], data["d1"]

    sample_weight = np.array([2.0, 1.0, 3.0])
    weighted = DiscreteMLE().fit(m1, d1, sample_weight=sample_weight)
    duplicated = DiscreteMLE().fit(m1, d1.loc[d1.index.repeat(sample_weight.astype(int))].reset_index(drop=True))

    for var in ["A", "B", "C"]:
        assert get_cpd(weighted, var) == get_cpd(duplicated, var)


def test_sample_weight_length_mismatch(setup_data):
    data = setup_data
    with pytest.raises(ValueError, match="sample_weight has length"):
        DiscreteMLE().fit(data["m1"], data["d1"], sample_weight=np.array([1.0, 2.0]))


@pytest.mark.skip(reason="JunctionTree support is intentionally out of scope for pgmpy.parameter_estimator.")
def test_estimate_potentials_smoke_test(setup_data, backend):
    pass


@pytest.mark.skip(reason="JunctionTree support is intentionally out of scope for pgmpy.parameter_estimator.")
def test_partition_function(setup_data, backend):
    pass


@pytest.mark.skip(reason="JunctionTree support is intentionally out of scope for pgmpy.parameter_estimator.")
def test_estimate_potentials(setup_data, backend):
    pass
