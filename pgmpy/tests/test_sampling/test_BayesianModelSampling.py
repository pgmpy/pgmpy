import pytest

from pgmpy.factors.discrete import State, TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork, DiscreteMarkovNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.sampling.base import BayesianModelInference


@pytest.fixture
def bayesian_model():
    model = DiscreteBayesianNetwork([("A", "J"), ("R", "J"), ("J", "Q"), ("J", "L"), ("G", "L")])
    cpd_a = TabularCPD("A", 2, [[0.2], [0.8]])
    cpd_r = TabularCPD("R", 2, [[0.4], [0.6]])
    cpd_j = TabularCPD("J", 2, [[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]], ["R", "A"], [2, 2])
    cpd_q = TabularCPD("Q", 2, [[0.9, 0.2], [0.1, 0.8]], ["J"], [2])
    cpd_l = TabularCPD("L", 2, [[0.9, 0.45, 0.8, 0.1], [0.1, 0.55, 0.2, 0.9]], ["G", "J"], [2, 2])
    cpd_g = TabularCPD("G", 2, [[0.6], [0.4]])
    model.add_cpds(cpd_a, cpd_g, cpd_j, cpd_l, cpd_q, cpd_r)
    return model


@pytest.fixture
def bayesian_model_lat():
    model = DiscreteBayesianNetwork(
        [("A", "J"), ("R", "J"), ("J", "Q"), ("J", "L"), ("G", "L")],
        latents=["R", "Q"],
    )
    cpd_a = TabularCPD("A", 2, [[0.2], [0.8]])
    cpd_r = TabularCPD("R", 2, [[0.4], [0.6]])
    cpd_j = TabularCPD("J", 2, [[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]], ["R", "A"], [2, 2])
    cpd_q = TabularCPD("Q", 2, [[0.9, 0.2], [0.1, 0.8]], ["J"], [2])
    cpd_l = TabularCPD("L", 2, [[0.9, 0.45, 0.8, 0.1], [0.1, 0.55, 0.2, 0.9]], ["G", "J"], [2, 2])
    cpd_g = TabularCPD("G", 2, [[0.6], [0.4]])
    model.add_cpds(cpd_a, cpd_g, cpd_j, cpd_l, cpd_q, cpd_r)
    return model


@pytest.fixture
def bayesian_model_names():
    model = DiscreteBayesianNetwork([("A", "J"), ("R", "J"), ("J", "Q"), ("J", "L"), ("G", "L")])
    cpd_a = TabularCPD("A", 2, [[0.2], [0.8]], state_names={"A": ["a0", "a1"]})
    cpd_r = TabularCPD("R", 2, [[0.4], [0.6]], state_names={"R": ["r0", "r1"]})
    cpd_j = TabularCPD(
        "J",
        2,
        [[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]],
        ["R", "A"],
        [2, 2],
        state_names={"J": ["j0", "j1"], "R": ["r0", "r1"], "A": ["a0", "a1"]},
    )
    cpd_q = TabularCPD(
        "Q",
        2,
        [[0.9, 0.2], [0.1, 0.8]],
        ["J"],
        [2],
        state_names={"Q": ["q0", "q1"], "J": ["j0", "j1"]},
    )
    cpd_l = TabularCPD(
        "L",
        2,
        [[0.9, 0.45, 0.8, 0.1], [0.1, 0.55, 0.2, 0.9]],
        ["G", "J"],
        [2, 2],
        state_names={"L": ["l0", "l1"], "G": ["g0", "g1"], "J": ["j0", "j1"]},
    )
    cpd_g = TabularCPD("G", 2, [[0.6], [0.4]], state_names={"G": ["g0", "g1"]})
    model.add_cpds(cpd_a, cpd_g, cpd_j, cpd_l, cpd_q, cpd_r)
    return model


@pytest.fixture
def bayesian_model_names_lat():
    model = DiscreteBayesianNetwork(
        [("A", "J"), ("R", "J"), ("J", "Q"), ("J", "L"), ("G", "L")],
        latents=["R", "Q"],
    )
    cpd_a = TabularCPD("A", 2, [[0.2], [0.8]], state_names={"A": ["a0", "a1"]})
    cpd_r = TabularCPD("R", 2, [[0.4], [0.6]], state_names={"R": ["r0", "r1"]})
    cpd_j = TabularCPD(
        "J",
        2,
        [[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]],
        ["R", "A"],
        [2, 2],
        state_names={"J": ["j0", "j1"], "R": ["r0", "r1"], "A": ["a0", "a1"]},
    )
    cpd_q = TabularCPD(
        "Q",
        2,
        [[0.9, 0.2], [0.1, 0.8]],
        ["J"],
        [2],
        state_names={"Q": ["q0", "q1"], "J": ["j0", "j1"]},
    )
    cpd_l = TabularCPD(
        "L",
        2,
        [[0.9, 0.45, 0.8, 0.1], [0.1, 0.55, 0.2, 0.9]],
        ["G", "J"],
        [2, 2],
        state_names={"L": ["l0", "l1"], "G": ["g0", "g1"], "J": ["j0", "j1"]},
    )
    cpd_g = TabularCPD("G", 2, [[0.6], [0.4]], state_names={"G": ["g0", "g1"]})
    model.add_cpds(cpd_a, cpd_g, cpd_j, cpd_l, cpd_q, cpd_r)
    return model


@pytest.fixture
def bayesian_model_int():
    model = DiscreteBayesianNetwork([("X", "Y")])
    cpd_x = TabularCPD("X", 2, [[0.5], [0.5]], state_names={"X": [1, 2]})
    cpd_y = TabularCPD(
        "Y",
        2,
        [[1.0, 0.0], [0.0, 1.0]],
        ["X"],
        [2],
        state_names={"Y": [1, 2], "X": [1, 2]},
    )
    model.add_cpds(cpd_x, cpd_y)
    return model


@pytest.fixture
def sampling_inference(bayesian_model):
    return BayesianModelSampling(bayesian_model)


@pytest.fixture
def sampling_inference_lat(bayesian_model_lat):
    return BayesianModelSampling(bayesian_model_lat)


@pytest.fixture
def sampling_inference_names(bayesian_model_names):
    return BayesianModelSampling(bayesian_model_names)


@pytest.fixture
def sampling_inference_names_lat(bayesian_model_names_lat):
    return BayesianModelSampling(bayesian_model_names_lat)


@pytest.fixture
def sampling_inference_int(bayesian_model_int):
    return BayesianModelSampling(bayesian_model_int)


@pytest.fixture
def forward_marginals(bayesian_model):
    return VariableElimination(bayesian_model).query(bayesian_model.nodes(), joint=False, show_progress=False)


@pytest.fixture
def markov_model():
    return DiscreteMarkovNetwork()


def test_init(markov_model):
    with pytest.raises(TypeError):
        BayesianModelSampling(markov_model)


def test_pre_compute_reduce_maps(bayesian_model):
    base_infer = BayesianModelInference(bayesian_model)
    state_to_index, index_to_weight = base_infer.pre_compute_reduce_maps("J", ["A", "R"], [(1, 1), (1, 0)])
    assert state_to_index[(1, 1)] == 0
    assert state_to_index[(1, 0)] == 1
    assert list(index_to_weight[0]) == [0.1, 0.9]
    assert list(index_to_weight[1]) == [0.6, 0.4]

    # Make sure the order of the evidence variables doesn't matter
    state_to_index, index_to_weight = base_infer.pre_compute_reduce_maps("J", ["R", "A"], [(1, 1), (1, 0)])
    assert state_to_index[(1, 1)] == 0
    assert state_to_index[(1, 0)] == 1
    assert list(index_to_weight[0]) == [0.1, 0.9]
    assert list(index_to_weight[1]) == [0.7, 0.3]


def test_pre_compute_reduce_maps_partial_evidence(bayesian_model):
    base_infer = BayesianModelInference(bayesian_model)
    state_to_index, index_to_weight = base_infer.pre_compute_reduce_maps("J", ["A"], [(1,), (0,)])
    assert state_to_index[(1,)] == 0
    assert state_to_index[(0,)] == 1
    assert list(index_to_weight[0].round(2)) == [0.35, 0.65]
    assert list(index_to_weight[1].round(2)) == [0.8, 0.2]

    # Make sure the order of the evidence variables doesn't matter
    state_to_index, index_to_weight = base_infer.pre_compute_reduce_maps("J", ["R"], [(1,), (0,)])
    assert state_to_index[(1,)] == 0
    assert state_to_index[(0,)] == 1
    assert list(index_to_weight[0].round(2)) == [0.4, 0.6]
    assert list(index_to_weight[1].round(2)) == [0.75, 0.25]


def test_forward_sample(
    bayesian_model,
    sampling_inference,
    sampling_inference_lat,
    sampling_inference_names,
    sampling_inference_names_lat,
    forward_marginals,
):
    # Test without state names
    sample = sampling_inference.forward_sample(int(1e5))
    assert len(sample) == int(1e5)
    assert len(sample.columns) == 6
    assert "A" in sample.columns
    assert "J" in sample.columns
    assert "R" in sample.columns
    assert "Q" in sample.columns
    assert "G" in sample.columns
    assert "L" in sample.columns
    assert set(sample.A).issubset({0, 1})
    assert set(sample.J).issubset({0, 1})
    assert set(sample.R).issubset({0, 1})
    assert set(sample.Q).issubset({0, 1})
    assert set(sample.G).issubset({0, 1})
    assert set(sample.L).issubset({0, 1})

    # Test that the marginal distribution of samples is same as the model
    sample_marginals = {node: sample[node].value_counts() / sample.shape[0] for node in bayesian_model.nodes()}
    for node in bayesian_model.nodes():
        for state in [0, 1]:
            assert round(forward_marginals[node].get_value(**{node: state}), 1) == round(
                sample_marginals[node].loc[state], 1
            )

    # Test without state names and with latents
    sample = sampling_inference_lat.forward_sample(25, include_latents=True)
    assert len(sample) == 25
    assert len(sample.columns) == 6
    assert set(sample.columns) == {"A", "J", "R", "Q", "G", "L"}
    assert set(sample.A).issubset({0, 1})
    assert set(sample.J).issubset({0, 1})
    assert set(sample.R).issubset({0, 1})
    assert set(sample.Q).issubset({0, 1})
    assert set(sample.G).issubset({0, 1})
    assert set(sample.L).issubset({0, 1})

    sample = sampling_inference_lat.forward_sample(25, include_latents=False)
    assert len(sample) == 25
    assert len(sample.columns) == 4
    assert "R" not in sample.columns
    assert "Q" not in sample.columns

    # Test with state names
    sample = sampling_inference_names.forward_sample(25)
    assert len(sample) == 25
    assert len(sample.columns) == 6
    assert "A" in sample.columns
    assert "J" in sample.columns
    assert "R" in sample.columns
    assert "Q" in sample.columns
    assert "G" in sample.columns
    assert "L" in sample.columns
    assert set(sample.A).issubset({"a0", "a1"})
    assert set(sample.J).issubset({"j0", "j1"})
    assert set(sample.R).issubset({"r0", "r1"})
    assert set(sample.Q).issubset({"q0", "q1"})
    assert set(sample.G).issubset({"g0", "g1"})
    assert set(sample.L).issubset({"l0", "l1"})

    # Test with state names and with latents
    sample = sampling_inference_names_lat.forward_sample(25, include_latents=True)
    assert len(sample) == 25
    assert len(sample.columns) == 6
    assert set(sample.columns) == {"A", "J", "R", "Q", "G", "L"}
    assert set(sample.A).issubset({"a0", "a1"})
    assert set(sample.J).issubset({"j0", "j1"})
    assert set(sample.R).issubset({"r0", "r1"})
    assert set(sample.Q).issubset({"q0", "q1"})
    assert set(sample.G).issubset({"g0", "g1"})
    assert set(sample.L).issubset({"l0", "l1"})

    sample = sampling_inference_names_lat.forward_sample(25, include_latents=False)
    assert len(sample) == 25
    assert len(sample.columns) == 4
    assert "R" not in sample.columns
    assert "Q" not in sample.columns


def test_rejection_sample_basic(
    bayesian_model,
    sampling_inference,
    sampling_inference_lat,
    sampling_inference_names,
    sampling_inference_names_lat,
):
    # Test without state names
    sampling_inference.rejection_sample()
    sample = sampling_inference.rejection_sample([State("A", 1), State("J", 1), State("R", 1)], int(1e5))
    assert len(sample) == int(1e5)
    assert len(sample.columns) == 6
    assert set(sample.columns) == {"A", "J", "R", "Q", "G", "L"}
    assert set(sample.A).issubset({1})
    assert set(sample.J).issubset({1})
    assert set(sample.R).issubset({1})
    assert set(sample.Q).issubset({0, 1})
    assert set(sample.G).issubset({0, 1})
    assert set(sample.L).issubset({0, 1})

    # Test that the marginal distributions match the model
    rejection_marginals = VariableElimination(bayesian_model).query(
        ["Q", "G", "L"],
        evidence={"A": 1, "J": 1, "R": 1},
        joint=False,
        show_progress=False,
    )
    sample_marginals = {node: sample[node].value_counts() / sample.shape[0] for node in ["Q", "G", "L"]}
    for node in ["Q", "G", "L"]:
        for state in [0, 1]:
            assert round(rejection_marginals[node].get_value(**{node: state}), 1) == round(
                sample_marginals[node].loc[state], 1
            )

    # Test without state names with latent variables
    sample = sampling_inference_lat.rejection_sample(
        [State("A", 1), State("J", 1), State("R", 1)], 25, include_latents=True
    )
    assert len(sample) == 25
    assert len(sample.columns) == 6
    assert set(sample.A).issubset({1})
    assert set(sample.J).issubset({1})
    assert set(sample.R).issubset({1})
    assert set(sample.Q).issubset({0, 1})
    assert set(sample.G).issubset({0, 1})
    assert set(sample.L).issubset({0, 1})

    sample = sampling_inference_lat.rejection_sample(
        [State("A", 1), State("J", 1), State("R", 1)], 25, include_latents=False
    )
    assert len(sample) == 25
    assert len(sample.columns) == 4
    assert set(sample.A).issubset({1})
    assert set(sample.J).issubset({1})
    assert set(sample.G).issubset({0, 1})
    assert set(sample.L).issubset({0, 1})

    # Test with state names
    sampling_inference_names.rejection_sample()
    sample = sampling_inference_names.rejection_sample([State("A", "a1"), State("J", "j1"), State("R", "r1")], 25)
    assert len(sample) == 25
    assert len(sample.columns) == 6
    assert set(sample.columns) == {"A", "J", "R", "Q", "G", "L"}
    assert set(sample.A).issubset({"a1"})
    assert set(sample.J).issubset({"j1"})
    assert set(sample.R).issubset({"r1"})
    assert set(sample.Q).issubset({"q0", "q1"})
    assert set(sample.G).issubset({"g0", "g1"})
    assert set(sample.L).issubset({"l0", "l1"})

    # Test with state names and latent variables
    sample = sampling_inference_names_lat.rejection_sample(
        [State("A", "a1"), State("J", "j1"), State("R", "r1")],
        25,
        include_latents=True,
    )
    assert len(sample) == 25
    assert len(sample.columns) == 6
    assert set(sample.columns) == {"A", "J", "R", "Q", "G", "L"}
    assert set(sample.A).issubset({"a1"})
    assert set(sample.J).issubset({"j1"})
    assert set(sample.R).issubset({"r1"})
    assert set(sample.Q).issubset({"q0", "q1"})
    assert set(sample.G).issubset({"g0", "g1"})
    assert set(sample.L).issubset({"l0", "l1"})

    sample = sampling_inference_names_lat.rejection_sample(
        [State("A", "a1"), State("J", "j1"), State("R", "r1")],
        25,
        include_latents=False,
    )
    assert len(sample) == 25
    assert len(sample.columns) == 4
    assert set(sample.columns) == {"A", "J", "G", "L"}
    assert set(sample.A).issubset({"a1"})
    assert set(sample.J).issubset({"j1"})
    assert set(sample.G).issubset({"g0", "g1"})
    assert set(sample.L).issubset({"l0", "l1"})


def test_likelihood_weighted_sample(
    sampling_inference,
    sampling_inference_lat,
    sampling_inference_names,
    sampling_inference_names_lat,
):
    # Test without state names
    sampling_inference.likelihood_weighted_sample()
    sample = sampling_inference.likelihood_weighted_sample([State("A", 0), State("J", 1), State("R", 0)], 25)
    assert len(sample) == 25
    assert len(sample.columns) == 7
    assert set(sample.columns) == {"A", "J", "R", "Q", "G", "L", "_weight"}
    assert set(sample.A).issubset({0})
    assert set(sample.J).issubset({1})
    assert set(sample.R).issubset({0})
    assert set(sample.Q).issubset({0, 1})
    assert set(sample.G).issubset({0, 1})
    assert set(sample.L).issubset({0, 1})

    # Test without state names and with latent variables
    sample = sampling_inference_lat.likelihood_weighted_sample(
        [State("A", 0), State("J", 1), State("R", 0)], 25, include_latents=True
    )
    assert len(sample) == 25
    assert len(sample.columns) == 7
    assert set(sample.columns) == {"A", "J", "R", "Q", "G", "L", "_weight"}
    assert set(sample.A).issubset({0})
    assert set(sample.J).issubset({1})
    assert set(sample.R).issubset({0})
    assert set(sample.Q).issubset({0, 1})
    assert set(sample.G).issubset({0, 1})
    assert set(sample.L).issubset({0, 1})

    sample = sampling_inference_lat.likelihood_weighted_sample(
        [State("A", 0), State("J", 1), State("R", 0)], 25, include_latents=False
    )
    assert len(sample) == 25
    assert len(sample.columns) == 5
    assert set(sample.columns) == {"A", "J", "G", "L", "_weight"}
    assert set(sample.A).issubset({0})
    assert set(sample.J).issubset({1})
    assert set(sample.G).issubset({0, 1})
    assert set(sample.L).issubset({0, 1})

    # Test with state names
    sampling_inference_names.likelihood_weighted_sample()
    sample = sampling_inference_names.likelihood_weighted_sample(
        [State("A", "a0"), State("J", "j1"), State("R", "r0")], 25
    )
    assert len(sample) == 25
    assert len(sample.columns) == 7
    assert set(sample.columns) == {"A", "J", "R", "Q", "G", "L", "_weight"}
    assert set(sample.A).issubset({"a0"})
    assert set(sample.J).issubset({"j1"})
    assert set(sample.R).issubset({"r0"})
    assert set(sample.Q).issubset({"q0", "q1"})
    assert set(sample.G).issubset({"g0", "g1"})
    assert set(sample.L).issubset({"l0", "l1"})

    # Test with state names and with latent variables
    sample = sampling_inference_names_lat.likelihood_weighted_sample(
        [State("A", "a0"), State("J", "j1"), State("R", "r0")],
        25,
        include_latents=True,
    )
    assert len(sample) == 25
    assert len(sample.columns) == 7
    assert set(sample.columns) == {"A", "J", "R", "Q", "G", "L", "_weight"}
    assert set(sample.A).issubset({"a0"})
    assert set(sample.J).issubset({"j1"})
    assert set(sample.R).issubset({"r0"})
    assert set(sample.Q).issubset({"q0", "q1"})
    assert set(sample.G).issubset({"g0", "g1"})
    assert set(sample.L).issubset({"l0", "l1"})

    sample = sampling_inference_names_lat.likelihood_weighted_sample(
        [State("A", "a0"), State("J", "j1"), State("R", "r0")],
        25,
        include_latents=False,
    )
    assert len(sample) == 25
    assert len(sample.columns) == 5
    assert set(sample.columns) == {"A", "J", "G", "L", "_weight"}
    assert set(sample.A).issubset({"a0"})
    assert set(sample.J).issubset({"j1"})
    assert set(sample.G).issubset({"g0", "g1"})
    assert set(sample.L).issubset({"l0", "l1"})


def test_rejection_sample_integer_state_names(sampling_inference_int):
    sampled_y = sampling_inference_int.rejection_sample(evidence=[State("X", 2)], size=1)["Y"][0]
    assert sampled_y == 2
