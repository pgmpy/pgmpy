import numpy as np
import pytest
from mock import MagicMock, patch

from pgmpy.factors.discrete import DiscreteFactor, State, TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork, DiscreteMarkovNetwork
from pgmpy.sampling import GibbsSampling


@pytest.fixture
def bayesian_model():
    diff_cpd = TabularCPD("diff", 2, [[0.6], [0.4]])
    intel_cpd = TabularCPD("intel", 2, [[0.7], [0.3]])
    grade_cpd = TabularCPD(
        "grade",
        3,
        [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
        evidence=["diff", "intel"],
        evidence_card=[2, 2],
    )
    model = DiscreteBayesianNetwork()
    model.add_nodes_from(["diff", "intel", "grade"])
    model.add_edges_from([("diff", "grade"), ("intel", "grade")])
    model.add_cpds(diff_cpd, intel_cpd, grade_cpd)
    return model


@pytest.fixture
def bayesian_model_sprinkler():
    cpt_cloudy = TabularCPD("Cloudy", 2, [[0.5], [0.5]])
    cpt_sprinkler = TabularCPD(
        "Sprinkler",
        2,
        [[0.5, 0.9], [0.5, 0.1]],
        evidence=["Cloudy"],
        evidence_card=[2],
    )
    cpt_rain = TabularCPD(
        "Rain",
        2,
        [[0.8, 0.2], [0.2, 0.8]],
        evidence=["Cloudy"],
        evidence_card=[2],
    )
    cpt_wet_grass = TabularCPD(
        "Wet_Grass",
        2,
        [[1, 0.1, 0.1, 0.01], [0, 0.9, 0.9, 0.99]],
        evidence=["Sprinkler", "Rain"],
        evidence_card=[2, 2],
    )
    model = DiscreteBayesianNetwork()
    model.add_edges_from(
        [
            ("Cloudy", "Sprinkler"),
            ("Cloudy", "Rain"),
            ("Sprinkler", "Wet_Grass"),
            ("Rain", "Wet_Grass"),
        ]
    )
    model.add_cpds(cpt_cloudy, cpt_sprinkler, cpt_rain, cpt_wet_grass)
    return model


@pytest.fixture
def markov_model():
    model = DiscreteMarkovNetwork([("A", "B"), ("C", "B"), ("B", "D")])
    factor_ab = DiscreteFactor(["A", "B"], [2, 3], [1, 2, 3, 4, 5, 6])
    factor_cb = DiscreteFactor(["C", "B"], [4, 3], [3, 1, 4, 5, 7, 8, 1, 3, 10, 4, 5, 6])
    factor_bd = DiscreteFactor(["B", "D"], [3, 2], [5, 7, 2, 1, 9, 3])
    model.add_factors(factor_ab, factor_cb, factor_bd)
    return model


@pytest.fixture
def gibbs(bayesian_model):
    return GibbsSampling(bayesian_model)


@patch("pgmpy.sampling.GibbsSampling._get_kernel_from_markov_model", autospec=True)
def test_init_markov_model(get_kernel):
    model = MagicMock(spec_set=DiscreteMarkovNetwork)
    gibbs = GibbsSampling(model)
    get_kernel.assert_called_once_with(gibbs, model)


def test_get_kernel_from_bayesian_model(bayesian_model):
    gibbs = GibbsSampling()
    gibbs._get_kernel_from_bayesian_model(bayesian_model)
    assert list(gibbs.variables) == list(bayesian_model.nodes())
    assert gibbs.cardinalities == {"diff": 2, "intel": 2, "grade": 3}


def test_get_kernel_from_bayesian_model_sprinkler(bayesian_model_sprinkler):
    gibbs = GibbsSampling()
    gibbs._get_kernel_from_bayesian_model(bayesian_model_sprinkler)
    assert list(gibbs.variables) == list(bayesian_model_sprinkler.nodes())
    assert gibbs.cardinalities == {
        "Cloudy": 2,
        "Rain": 2,
        "Sprinkler": 2,
        "Wet_Grass": 2,
    }


def test_get_kernel_from_markov_model(markov_model):
    gibbs = GibbsSampling()
    gibbs._get_kernel_from_markov_model(markov_model)
    assert list(gibbs.variables) == list(markov_model.nodes())
    assert gibbs.cardinalities == {"A": 2, "B": 3, "C": 4, "D": 2}


def test_sample(gibbs):
    start_state = [State("diff", 0), State("intel", 0), State("grade", 0)]
    sample = gibbs.sample(start_state, 2)
    assert len(sample) == 2
    assert len(sample.columns) == 3
    assert "diff" in sample.columns
    assert "intel" in sample.columns
    assert "grade" in sample.columns
    assert set(sample["diff"]).issubset({0, 1})
    assert set(sample["intel"]).issubset({0, 1})
    assert set(sample["grade"]).issubset({0, 1, 2})


def test_sample_sprinkler(bayesian_model_sprinkler):
    gibbs = GibbsSampling(bayesian_model_sprinkler)
    nodes = set(bayesian_model_sprinkler.nodes)
    sample = gibbs.sample(size=2)
    assert len(sample) == 2
    assert len(sample.columns) == 4
    assert nodes == set(sample.columns)
    for node in nodes:
        assert set(sample[node]).issubset({0, 1})


def test_sample_limit(gibbs, bayesian_model):
    samples = gibbs.sample(size=int(1e4))
    marginal_prob = VariableElimination(bayesian_model).query(
        list(bayesian_model.nodes()), joint=False, show_progress=False
    )
    sample_prob = {node: samples.loc[:, node].value_counts() / 1e4 for node in bayesian_model.nodes()}
    for node in bayesian_model.nodes():
        assert np.allclose(
            sorted(marginal_prob[node].values),
            sorted(sample_prob[node].values),
            atol=0.05,
        )


@patch("pgmpy.sampling.GibbsSampling.random_state", autospec=True)
def test_sample_less_arg(random_state, gibbs):
    gibbs.state = None
    random_state.return_value = [
        State("diff", 0),
        State("intel", 0),
        State("grade", 0),
    ]
    sample = gibbs.sample(size=2)
    random_state.assert_called_once_with(gibbs)
    assert len(sample) == 2


def test_generate_sample(gibbs):
    start_state = [State("diff", 0), State("intel", 0), State("grade", 0)]
    gen = gibbs.generate_sample(start_state, 2)
    samples = [sample for sample in gen]
    assert len(samples) == 2
    assert {
        samples[0][0].var,
        samples[0][1].var,
        samples[0][2].var,
    } == {"diff", "intel", "grade"}
    assert {
        samples[1][0].var,
        samples[1][1].var,
        samples[1][2].var,
    } == {"diff", "intel", "grade"}


@patch("pgmpy.sampling.GibbsSampling.random_state", autospec=True)
def test_generate_sample_less_arg(random_state, gibbs):
    gibbs.state = None
    gen = gibbs.generate_sample(size=2)
    samples = [sample for sample in gen]
    random_state.assert_called_once_with(gibbs)
    assert len(samples) == 2
