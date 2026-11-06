import inspect
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from pgmpy.base import DAG
from pgmpy.causal_discovery import SP, BootstrapEstimator
from pgmpy.ci_tests import HSIC, GeneralizedCov
from pgmpy.datasets import load_dataset
from pgmpy.datasets._base import BaseCovarianceDataset, BaseDataset
from pgmpy.datasets.anm import AdditiveNoiseModel
from pgmpy.datasets.lead import Lead
from pgmpy.datasets.linear_gaussian_scm import LinearGaussianSCM
from pgmpy.example_models import load_model
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.discrete import DiscreteFactor, State, TabularCPD
from pgmpy.inference import ApproxInference
from pgmpy.models import (
    DiscreteBayesianNetwork,
    DynamicBayesianNetwork,
    LinearGaussianBayesianNetwork,
    MarkovChain,
)
from pgmpy.parameter_estimator import DiscreteEM
from pgmpy.parameterization._base import BaseParameter
from pgmpy.parameterization.distributions import NominalDistribution
from pgmpy.prediction import DoubleMLRegressor
from pgmpy.sampling import BayesianModelSampling, GibbsSampling
from pgmpy.utils import mathext
from pgmpy.utils._random import _rename_to_rng
from pgmpy.utils.mathext import sample_discrete, sample_discrete_maps


@_rename_to_rng("seed")
def _toy(n, rng=None, flag=False):
    return rng


class _Toy:
    @_rename_to_rng("random_state")
    def __init__(self, a=1, rng=None):
        self.rng = rng


def _markov_chain():
    model = MarkovChain(["A", "B"], [2, 2])
    uniform_tm = {0: {0: 0.5, 1: 0.5}, 1: {0: 0.5, 1: 0.5}}
    model.add_transition_model("A", uniform_tm)
    model.add_transition_model("B", uniform_tm)
    return model


def _asia():
    return load_model("bnlearn/asia")


def _two_slice_dbn():
    dbn = DynamicBayesianNetwork([(("A", 0), ("A", 1))])
    dbn.add_cpds(
        TabularCPD(("A", 0), 2, [[0.5], [0.5]]),
        TabularCPD(("A", 1), 2, [[0.5, 0.5], [0.5, 0.5]], evidence=[("A", 0)], evidence_card=[2]),
    )
    dbn.initialize_initial_state()
    return dbn


@pytest.mark.parametrize(
    "seeded_call",
    [
        lambda: _markov_chain().sample(size=5, rng=0),
        lambda: list(_markov_chain().generate_sample(size=5, rng=0)),
        lambda: BayesianModelSampling(_asia()).forward_sample(size=20, rng=0),
        lambda: BayesianModelSampling(_asia()).rejection_sample(evidence=[State("smoke", "yes")], size=20, rng=0),
        lambda: BayesianModelSampling(_asia()).likelihood_weighted_sample(
            evidence=[State("smoke", "yes")], size=20, rng=0
        ),
        lambda: GibbsSampling(_asia()).sample(size=5, rng=0),
        lambda: list(GibbsSampling(_asia()).generate_sample(size=5, rng=0)),
        lambda: _asia().simulate(20, rng=0),
        lambda: ApproxInference(_asia()).query(["lung"], n_samples=100, rng=0),
        lambda: _two_slice_dbn().simulate(n_samples=20, n_time_slices=3, rng=0, show_progress=False),
        lambda: _asia().predict(_asia().simulate(20, rng=0).drop(columns=["lung"]), n_jobs=1),
    ],
    ids=[
        "MarkovChain.sample",
        "MarkovChain.generate_sample",
        "forward_sample",
        "rejection_sample",
        "likelihood_weighted_sample",
        "GibbsSampling.sample",
        "GibbsSampling.generate_sample",
        "simulate",
        "ApproxInference.query",
        "DynamicBayesianNetwork.simulate",
        "DiscreteBayesianNetwork.predict(MAP)",
    ],
)
def test_seeded_call_leaves_global_state_untouched(seeded_call):
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    seeded_call()
    assert np.random.random() == expected


def _discrete_bn_cpd_values():
    model = DiscreteBayesianNetwork.get_random(n_nodes=4, edge_prob=0.5)
    cpds = sorted(model.get_cpds(), key=lambda cpd: cpd.variable)
    return np.concatenate([cpd.values.ravel() for cpd in cpds])


def _lgbn_betas():
    model = LinearGaussianBayesianNetwork.get_random(n_nodes=4, edge_prob=0.5)
    return np.concatenate([cpd.beta for cpd in model.cpds])


def _dagitty_betas():
    model = DAG.from_dagitty("dag{X -> Y [beta=0.4]; Z -> Y}")
    return np.concatenate([cpd.beta for cpd in model.cpds])


def _hsic_p_value():
    from pgmpy.ci_tests import HSIC

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"X": rng.normal(size=60), "Y": rng.normal(size=60)})
    test = HSIC(data=df, null_dist="permutation", n_permutations=50, use_cache=False)
    p_values = []
    for _ in range(5):
        test("X", "Y")
        p_values.append(test.p_value_)
    return np.array(p_values)


def _generalized_cov_p_value():
    from pgmpy.ci_tests import GeneralizedCov

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"X": rng.normal(size=60), "Y": rng.normal(size=60)})
    test = GeneralizedCov(data=df, n_permutations=50, use_cache=False)
    p_values = []
    for _ in range(5):
        test("X", "Y")
        p_values.append(test.p_value_)
    return np.array(p_values)


def _bootstrap_samples():
    from pgmpy.causal_discovery import BootstrapEstimator, HillClimbSearch

    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.integers(0, 2, size=(50, 3)), columns=["A", "B", "C"]).astype("category")
    estimator = BootstrapEstimator(estimator=HillClimbSearch(return_type="dag"), n_bootstraps=2, n_jobs=1)
    return estimator.fit(df).bootstrap_samples_


def _nominal_distribution():
    from pgmpy.parameterization.distributions import NominalDistribution

    return NominalDistribution([[0.1, 0.9], [0.7, 0.3]], ["A", "B"])


@pytest.mark.parametrize(
    "unseeded_call",
    [
        lambda: TabularCPD.get_random("A", evidence=["B"], cardinality={"A": 3, "B": 2}).values,
        lambda: np.asarray(DiscreteFactor(["A"], [3], [0.2, 0.3, 0.5]).sample(20)),
        lambda: LinearGaussianCPD.get_random("Y", ["X"]).beta,
        lambda: np.asarray(_nominal_distribution().sample(20)),
        lambda: np.array(sorted(DAG.get_random(n_nodes=6, edge_prob=0.5).edges())),
        _discrete_bn_cpd_values,
        _lgbn_betas,
        _dagitty_betas,
        lambda: LinearGaussianBayesianNetwork.get_random(n_nodes=3, edge_prob=1, rng=0).simulate(50).to_numpy(),
        _hsic_p_value,
        _generalized_cov_p_value,
        _bootstrap_samples,
        lambda: Lead.load_dataframe(n_samples=20).to_numpy(),
        lambda: load_dataset("sachs_discrete", n_samples=20).data.to_numpy(),
        lambda: BayesianModelSampling(_asia()).forward_sample(size=50, show_progress=False).to_numpy(),
        lambda: _asia().simulate(n_samples=50, show_progress=False).to_numpy(),
        lambda: ApproxInference(_asia()).query(["lung", "bronc", "tub"], n_samples=20000, show_progress=False).values,
        lambda: sample_discrete([0, 1, 2], [0.2, 0.3, 0.5], size=50),
        lambda: _markov_chain().sample(size=20).to_numpy(),
    ],
    ids=[
        "TabularCPD.get_random",
        "DiscreteFactor.sample",
        "LinearGaussianCPD.get_random",
        "NominalDistribution.sample",
        "DAG.get_random",
        "DiscreteBayesianNetwork.get_random",
        "LinearGaussianBayesianNetwork.get_random",
        "DAG.from_dagitty",
        "LinearGaussianBayesianNetwork.simulate",
        "HSIC",
        "GeneralizedCov",
        "BootstrapEstimator",
        "Lead.load_dataframe",
        "load_dataset.static_subsample",
        "BayesianModelSampling.forward_sample",
        "DiscreteBayesianNetwork.simulate",
        "ApproxInference.query",
        "sample_discrete",
        "MarkovChain.sample",
    ],
)
@pytest.mark.filterwarnings("ignore:pgmpy.parameterization is experimental:UserWarning")
def test_none_seed_ignores_global_state(unseeded_call):
    np.random.seed(0)
    first = unseeded_call()
    np.random.seed(0)
    assert not np.array_equal(unseeded_call(), first)


def test_rename_to_rng_keyword_passes_through():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _toy(1, rng=5) == 5


def test_rename_to_rng_old_keyword_warns_and_forwards():
    with pytest.warns(
        DeprecationWarning,
        match=r"^`seed` is deprecated since v1\.2\.0 and will be removed in v2\.0\. Use `rng` instead\. "
        r"pgmpy follows SPEC 7 for random number generation: https://scientific-python\.org/specs/spec-0007/$",
    ) as record:
        assert _toy(1, seed=5) == 5
    assert record[0].filename == __file__
    with pytest.warns(DeprecationWarning, match=r"^`random_state` is deprecated"):
        assert _Toy(random_state=3).rng == 3
    with pytest.warns(DeprecationWarning, match=r"^`seed` is deprecated"):
        assert _toy(1, seed=None) is None


def test_rename_to_rng_positional_warns_and_forwards():
    message = (
        r"^Passing `rng` positionally is deprecated since v1\.2\.0\. "
        r"From v2\.0 it is keyword-only; pass it as `rng=\.\.\.`\. "
        r"pgmpy follows SPEC 7 for random number generation: https://scientific-python\.org/specs/spec-0007/$"
    )
    with pytest.warns(DeprecationWarning, match=message):
        assert _toy(1, 5) == 5
    with pytest.warns(DeprecationWarning, match=r"^Passing `rng` positionally"):
        assert _Toy(1, 4).rng == 4


def test_rename_to_rng_rejects_both_names():
    for call in (lambda: _toy(1, rng=2, seed=3), lambda: _toy(1, 2, seed=3), lambda: _toy(1, rng=None, seed=None)):
        with pytest.raises(TypeError):
            call()


def test_rename_to_rng_keeps_signature():
    assert list(inspect.signature(_toy).parameters) == ["n", "rng", "flag"]
    assert _toy._rng_old_name == "seed"
    assert _toy(1) is None and _Toy().rng is None


_RELEASED_SITES = [
    (DAG, "get_random", "seed"),
    (GeneralizedCov, "__init__", "random_state"),
    (LinearGaussianCPD, "get_random", "seed"),
    (TabularCPD, "get_random", "seed"),
    (DiscreteFactor, "sample", "seed"),
    (ApproxInference, "query", "seed"),
    (ApproxInference, "map_query", "seed"),
    (DiscreteBayesianNetwork, "get_random", "seed"),
    (DiscreteBayesianNetwork, "get_random_cpds", "seed"),
    (DiscreteBayesianNetwork, "predict", "seed"),
    (DiscreteBayesianNetwork, "simulate", "seed"),
    (DynamicBayesianNetwork, "simulate", "seed"),
    (LinearGaussianBayesianNetwork, "get_random", "seed"),
    (LinearGaussianBayesianNetwork, "get_random_cpds", "seed"),
    (LinearGaussianBayesianNetwork, "simulate", "seed"),
    (MarkovChain, "sample", "seed"),
    (MarkovChain, "generate_sample", "seed"),
    (DiscreteEM, "__init__", "seed"),
    (DoubleMLRegressor, "__init__", "seed"),
    (BayesianModelSampling, "forward_sample", "seed"),
    (BayesianModelSampling, "rejection_sample", "seed"),
    (BayesianModelSampling, "likelihood_weighted_sample", "seed"),
    (GibbsSampling, "sample", "seed"),
    (GibbsSampling, "generate_sample", "seed"),
    (mathext, "sample_discrete", "seed"),
    (mathext, "sample_discrete_maps", "seed"),
]


@pytest.mark.parametrize(("owner", "name", "old_name"), _RELEASED_SITES)
def test_released_site_renamed(owner, name, old_name):
    site = getattr(owner, name)
    assert site._rng_old_name == old_name
    assert "rng" in inspect.signature(site).parameters


@pytest.mark.parametrize("name", ["fit", "simulate"])
def test_functional_bayesian_network_site_renamed(name):
    pytest.importorskip("pyro")
    from pgmpy.models import FunctionalBayesianNetwork

    site = getattr(FunctionalBayesianNetwork, name)
    assert site._rng_old_name == "seed"
    assert "rng" in inspect.signature(site).parameters


def test_old_name_gives_same_result():
    asia = _asia()
    with pytest.warns(DeprecationWarning, match=r"^`seed` is deprecated"):
        old = asia.simulate(50, seed=3, show_progress=False)
    pd.testing.assert_frame_equal(old, asia.simulate(50, rng=3, show_progress=False))

    with pytest.warns(DeprecationWarning, match=r"^`seed` is deprecated"):
        old_cpd = TabularCPD.get_random("A", cardinality={"A": 3}, seed=3)
    assert old_cpd == TabularCPD.get_random("A", cardinality={"A": 3}, rng=3)


def test_positional_rng_on_static_and_instance_methods():
    asia = _asia()
    with pytest.warns(DeprecationWarning, match=r"^Passing `rng` positionally"):
        positional = asia.simulate(50, None, None, None, None, None, False, None, 3, False)
    pd.testing.assert_frame_equal(positional, asia.simulate(50, rng=3, show_progress=False))

    with pytest.warns(DeprecationWarning, match=r"^Passing `rng` positionally"):
        positional_cpd = TabularCPD.get_random("A", None, {"A": 3}, {}, 3)
    assert positional_cpd == TabularCPD.get_random("A", cardinality={"A": 3}, rng=3)


def test_old_name_estimator_survives_clone():
    dag = DAG([("X", "T"), ("X", "Y"), ("T", "Y")], roles={"exposures": "T", "adjustment": "X", "outcomes": "Y"})
    with pytest.warns(DeprecationWarning, match=r"^`seed` is deprecated"):
        est = DoubleMLRegressor(
            causal_graph=dag,
            nuisance_estimators=LinearRegression(),
            effect_estimator=LinearRegression(),
            n_folds=3,
            seed=3,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        cloned = clone(est)
    assert cloned.get_params()["rng"] == 3


def test_decorated_function_is_picklable():
    assert pickle.loads(pickle.dumps(sample_discrete)) is sample_discrete
    assert pickle.loads(pickle.dumps(sample_discrete_maps)) is sample_discrete_maps


def test_get_uniform_seed_deprecated():
    with pytest.warns(DeprecationWarning, match=r"^`seed` is deprecated since v1\.2\.0 .* It has no effect\.$"):
        TabularCPD.get_uniform("A", cardinality={"A": 2}, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TabularCPD.get_uniform("A", cardinality={"A": 2})


@pytest.mark.parametrize(
    "func",
    [
        BootstrapEstimator.__init__,
        SP.__init__,
        HSIC.__init__,
        load_dataset,
        BaseDataset.load_dataframe,
        BaseCovarianceDataset.load_dataframe,
        AdditiveNoiseModel.__init__,
        LinearGaussianSCM.__init__,
        BaseParameter.sample,
        BaseParameter._sample,
    ],
)
def test_dev_only_site_takes_keyword_only_rng(func):
    parameters = inspect.signature(func).parameters
    assert parameters["rng"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "seed" not in parameters and "random_state" not in parameters


def test_nominal_distribution_keeps_random_state():
    assert "random_state" in inspect.signature(NominalDistribution.sample).parameters


def _set_params_estimators():
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 3)), columns=["A", "B", "C"])
    return [
        (DiscreteEM(), "seed"),
        (DoubleMLRegressor(causal_graph=DAG([("A", "B")])), "seed"),
        (GeneralizedCov(data=df), "random_state"),
    ]


@pytest.mark.parametrize("index", [0, 1, 2], ids=["DiscreteEM", "DoubleMLRegressor", "GeneralizedCov"])
def test_set_params_old_name_deprecated(index):
    est, old_name = _set_params_estimators()[index]
    message = (
        rf"^`{old_name}` is deprecated since v1\.2\.0 and will be removed in v2\.0\. Use `rng` instead\. "
        r"pgmpy follows SPEC 7 for random number generation: https://scientific-python\.org/specs/spec-0007/$"
    )
    with pytest.warns(DeprecationWarning, match=message):
        est.set_params(**{old_name: 3})
    assert est.get_params()["rng"] == 3

    with pytest.raises(TypeError, match="got both"):
        est.set_params(**{old_name: 4, "rng": 5})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (est if isinstance(est, GeneralizedCov) else clone(est)).set_params(rng=7)
