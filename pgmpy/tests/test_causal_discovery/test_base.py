import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pytest
from skbase.lookup import all_objects

from pgmpy.base import DAG
from pgmpy.causal_discovery import (
    ANM,
    GES,
    IGCI,
    PC,
    SP,
    BootstrapEstimator,
    ChowLiu,
    ExpertInLoop,
    LLMPairwise,
    R2Sort,
    VarSort,
)
from pgmpy.causal_discovery._base import BaseCausalDiscovery, _TreeSearchMixin
from pgmpy.causal_discovery.bivariate_scores import BaseBivariateScore
from pgmpy.ci_tests import GCM, BaseCITest, IndependenceMatch, get_ci_test
from pgmpy.ci_tests._base import _CITestResult
from pgmpy.independencies import Independencies
from pgmpy.structure_score import get_scoring_method


@pytest.mark.parametrize(
    ("data", "estimator"),
    [
        (
            pd.DataFrame(
                {
                    "const": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    "y": [2.0, 1.0, 3.0, 2.0, 1.0, 3.0],
                }
            ),
            R2Sort(),
        ),
        (
            pd.DataFrame(
                {
                    "const": ["a", "a", "a", "a", "a", "a"],
                    "x": ["p", "q", "p", "q", "q", "p"],
                    "y": ["m", "n", "n", "m", "n", "m"],
                }
            ),
            ChowLiu(),
        ),
    ],
    ids=["continuous", "discrete"],
)
def test_fit_warns_on_constant_column(data, estimator):
    with pytest.warns(UserWarning, match="constant"):
        estimator.fit(data)


def test_fit_rejects_incompatible_data_types():
    rng = np.random.default_rng(0)
    integers = pd.DataFrame(rng.integers(0, 2, size=(50, 2)), columns=["A", "B"])
    discrete = pd.DataFrame({"A": rng.choice(["a", "b"], 50), "B": rng.choice(["x", "y"], 50)})
    booleans = integers.astype(bool)

    with pytest.raises(ValueError, match=r"R2Sort supports \['continuous'\] data, but the data is discrete"):
        R2Sort().fit(discrete)
    for data, dataset_type in [
        (integers, "continuous"),
        (booleans.assign(C=rng.normal(size=50)), "continuous"),
        (integers.assign(C=discrete["A"]), "mixed"),
    ]:
        with pytest.raises(ValueError, match=rf"ChowLiu supports \['discrete'\] data, but the data is {dataset_type}"):
            ChowLiu().fit(data)

    class RecordingChowLiu(ChowLiu):
        def _fit(self, X):
            self.fit_dtypes_ = X.dtypes.astype(str).to_dict()
            return super()._fit(X)

    for data in [discrete, booleans, integers.astype("category")]:
        assert RecordingChowLiu().fit(data).fit_dtypes_ == {"A": "category", "B": "category"}
    assert set(R2Sort().fit(booleans).causal_graph_.nodes()) == {"A", "B"}


def test_estimators_use_identity_equality_and_hash():
    pc = PC()
    assert pc == pc
    assert PC() != PC()
    assert len({pc, PC()}) == 2


def test_fit_narrows_union_tags_to_component():
    rng = np.random.default_rng(42)
    x = rng.normal(size=200)
    data = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(size=200), "z": rng.normal(size=200)})

    assert PC.get_class_tag("assumption:linearity") is False
    pc = PC(ci_test="pearsonr").fit(data)
    assert pc.get_tag("data_types") == ["continuous"]
    assert pc.get_tag("assumption:linearity") is True
    assert pc.get_tag("assumption:faithfulness") is True

    pc.set_params(ci_test=lambda X, Y, Z, **kwargs: True).fit(data)
    assert pc.get_tag("assumption:linearity") is False
    assert pc.get_tag("data_types") == PC.get_class_tag("data_types")

    assert GCM(data=data).get_tag("assumption:linearity") is True
    assert GCM.get_class_tag("assumption:linearity") is False

    ges = GES(scoring_method="bic-g").fit(data)
    assert ges.get_tag("assumption:gaussian_noise") is True
    assert ges.get_tag("data_types") == ["continuous"]
    ges.set_params(scoring_method="bic-cg").fit(data)
    assert ges.get_tag("data_types") == ["discrete", "continuous", "mixed"]

    anm = ANM().fit(data[["x", "y"]])
    assert anm.get_tag("data_types") == ["continuous"]

    # Storing the resolved component as `ci_test_` or `scoring_method_` is enough; `fit` narrows the tags.
    class MinimalPC(PC):
        def _fit(self, X, independencies=None):
            self.ci_test_ = get_ci_test(test=self.ci_test, data=X)
            return self

    class MinimalGES(GES):
        def _fit(self, X):
            self.scoring_method_ = get_scoring_method(self.scoring_method, X)
            return self

    assert MinimalPC(ci_test="pearsonr").fit(data).get_tag("assumption:linearity") is True
    assert MinimalGES(scoring_method="bic-g").fit(data).get_tag("assumption:gaussian_noise") is True


def test_fit_narrowing_handles_untagged_and_wider_objects():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, 300)
    data = pd.DataFrame({"x": x, "y": x**3 + rng.normal(0, 0.5, 300)})

    class UntaggedCITest(BaseCITest):
        def _compute_result(self, X, Y, Z):
            return _CITestResult(statistic=0.0, p_value=1.0)

    pc = PC(ci_test=UntaggedCITest()).fit(data)
    assert pc.get_tag("data_types") == ["discrete", "continuous", "mixed"]
    assert pc.get_tag("assumption:linearity") is False

    class AnyDataScore(BaseBivariateScore):
        _tags = {"name": "any_data", "input_type": "cause_residual", "data_types": ["discrete", "continuous", "mixed"]}

        def __call__(self, x, y):
            return float(np.var(y))

    anm = ANM(scoring_method=AnyDataScore()).fit(data)
    assert anm.get_tag("data_types") == ["continuous"]
    assert anm.get_tag("assumption:linearity") is False

    class UntaggedAlgorithm(BaseCausalDiscovery):
        def _fit(self, X):
            self.ci_test_ = get_ci_test(test="pearsonr", data=X)
            self.causal_graph_ = DAG()
            self.causal_graph_.add_nodes_from(X.columns)
            return self

    assert UntaggedAlgorithm().fit(data).get_tag("data_types") == ["continuous"]

    # Tags that the wrapped estimator leaves unset go back to the class-level value on refit.
    bootstrap = BootstrapEstimator(PC(show_progress=False), n_bootstraps=2, n_jobs=1, show_progress=False).fit(data)
    assert bootstrap.get_tag("identifiable_graph") == "cpdag"
    bootstrap.set_params(estimator=UntaggedAlgorithm()).fit(data)
    assert bootstrap.get_tag("data_types") == ["continuous"]
    assert bootstrap.get_tag("identifiable_graph") == BootstrapEstimator.get_class_tag("identifiable_graph")


def test_fit_narrows_tags_to_components_used():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, 300)
    data = pd.DataFrame({"x": x, "y": x**3 + rng.normal(0, 0.5, 300), "z": rng.normal(size=300)})

    pc = PC(show_progress=False).fit(data, independencies=Independencies(["x", "z"], ["y", "z"]))
    assert isinstance(pc.ci_test_, IndependenceMatch)
    assert pc.get_tag("data_types") == ["discrete", "continuous", "mixed"]
    assert pc.get_tag("assumption:linearity") is False

    expert_in_loop = ExpertInLoop(pairwise_estimator=IGCI(), show_progress=False).fit(data)
    assert expert_in_loop.get_tag("assumption:low_noise") is True
    assert expert_in_loop.get_tag("data_types") == ["continuous"]
    assert expert_in_loop.get_tag("capability:multivariate") is True

    # A wrapped estimator determines every tag except the name.
    pc = PC(ci_test="pearsonr", show_progress=False)
    bootstrap = BootstrapEstimator(pc, n_bootstraps=2, n_jobs=1, show_progress=False).fit(data)
    assert bootstrap.get_tags() == {**pc.fit(data).get_tags(), "name": "bootstrap_estimator"}
    assert bootstrap.get_tag("identifiable_graph") == "cpdag"

    # Components narrow the tags taken over from a wrapped estimator.
    class BootstrapWithCITest(BootstrapEstimator):
        def _fit(self, X):
            self.ci_test_ = get_ci_test(test="pearsonr", data=X)
            return super()._fit(X)

    pc = PC(ci_test=lambda X, Y, Z, **kwargs: True, show_progress=False)
    wrapper = BootstrapWithCITest(pc, n_bootstraps=2, n_jobs=1, show_progress=False).fit(data)
    assert wrapper.get_tag("identifiable_graph") == "cpdag"
    assert wrapper.get_tag("assumption:faithfulness") is True
    assert wrapper.get_tag("assumption:linearity") is True
    assert wrapper.get_tag("data_types") == ["continuous"]


def test_filter_algorithms_by_tags():
    names = {
        cls.__name__
        for cls in all_objects(
            object_types=BaseCausalDiscovery,
            package_name="pgmpy.causal_discovery",
            return_names=False,
            filter_tags={"data_types": "continuous", "assumption:linearity": False, "capability:multivariate": True},
        )
    }
    assert {"PC", "GES"} <= names
    assert not names & {"ANM", "IGCI", "VarSort", "R2Sort"}

    dag_learners = {
        cls.__name__
        for cls in all_objects(
            object_types=BaseCausalDiscovery,
            package_name="pgmpy.causal_discovery",
            return_names=False,
            filter_tags={"identifiable_graph": "dag", "capability:multivariate": True},
        )
    }
    assert dag_learners == {"BootstrapEstimator", "ExpertInLoop", "VarSort", "R2Sort"}


@pytest.mark.parametrize(
    ("data", "exc_type", "match"),
    [
        (pd.DataFrame({"a": [1, 2]}).drop(columns=["a"]), ValueError, "0 feature"),
        (pd.DataFrame({"a": [1], "b": [2]}), ValueError, "at least 2 are required"),
        (pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0]}), ValueError, "Input contains NaN/null values"),
        (
            pd.DataFrame({"a": pd.Series(["x", np.nan], dtype=object), "b": [1, 2]}),
            ValueError,
            "Input contains NaN/null values",
        ),
        (pd.DataFrame({"a": [np.inf, 1.0], "b": [1.0, 2.0]}), ValueError, "Input contains infinite values"),
        (pd.DataFrame({"a": [-np.inf, 1.0], "b": [1.0, 2.0]}), ValueError, "Input contains infinite values"),
        (
            pd.DataFrame({"a": pd.Series(["x", np.inf], dtype=object), "b": [1, 2]}),
            ValueError,
            "Input contains infinite values",
        ),
        (
            pd.DataFrame({"a": [[1], [2]], "b": [1, 2]}),
            TypeError,
            "argument must be a string, number, or hashable object",
        ),
        (
            pd.DataFrame([[1, 2], [3, 4]], columns=["A", 1]),
            TypeError,
            "Feature names are only supported if all input features have string names",
        ),
    ],
    ids=[
        "0_features",
        "1_row",
        "float_nan",
        "object_nan",
        "pos_inf",
        "neg_inf",
        "object_inf",
        "unhashable_list",
        "mixed_columns",
    ],
)
def test_check_fit_invalid_input(data, exc_type, match):
    nw_df = nw.from_native(data)
    with pytest.raises(exc_type, match=match):
        ChowLiu(n_jobs=1).fit(nw_df)


@pytest.mark.parametrize("backend", ["polars", "pyarrow"])
def test_check_fit_unhashable_backends(backend):
    if backend == "polars":
        pl = pytest.importorskip("polars")
        data = pl.DataFrame({"a": [[1], [2]], "b": [1, 2]})
    elif backend == "pyarrow":
        pa = pytest.importorskip("pyarrow")
        data = pa.table({"a": [[1], [2]], "b": [1, 2]})

    with pytest.raises(TypeError, match="argument must be a string, number, or hashable object"):
        ChowLiu(n_jobs=1).fit(data)


def test_check_fit_unhashable_struct_or_list():
    df = pd.DataFrame({"a": [1, 2], "b": [2, 1]})
    nw_df = nw.from_native(df)

    class FakeSeries:
        def __init__(self, s):
            self._s = s
            self.dtype = nw.dtypes.List(nw.dtypes.Int64())

        def __getattr__(self, name):
            return getattr(self._s, name)

    orig_get_column = nw_df.get_column
    nw_df.get_column = lambda col: FakeSeries(orig_get_column(col)) if col == "a" else orig_get_column(col)

    with pytest.raises(TypeError, match="argument must be a string, number, or hashable object"):
        ChowLiu(n_jobs=1).fit(nw_df)


def test_check_fit_score_parity():
    pandas_df = pd.DataFrame({"a": ["1", "2", "1", "2"], "b": ["2", "1", "2", "1"]})
    nw_df = nw.from_native(pandas_df)
    est = ChowLiu(n_jobs=1)
    est.fit(nw_df)
    assert hasattr(est, "causal_graph_")

    arr = np.array([["1", "2"], ["2", "1"], ["1", "2"], ["2", "1"]])
    est_arr = ChowLiu(n_jobs=1).fit(arr)
    assert list(est_arr.feature_names_in_) == ["x0", "x1"]
    assert hasattr(est_arr, "causal_graph_")

    score_pd = est.score(pandas_df)
    score_nw = est.score(nw_df)
    score_np = est.score(pandas_df.to_numpy())
    assert np.isclose(score_pd, score_nw)
    assert np.isclose(score_pd, score_np)


@pytest.mark.parametrize("backend", ["pandas_nullable", "polars", "pyarrow"])
def test_check_fit_unmigrated_bridge(backend):
    captured = {}

    class UnmigratedDiscovery(BaseCausalDiscovery):
        def _fit(self, X):
            assert isinstance(X, pd.DataFrame)
            captured["df"] = X
            self.causal_graph_ = DAG()
            return self

    if backend == "pandas_nullable":
        df = pd.DataFrame({"a": pd.Series([1, 2, 1, 2], dtype="Int64"), "b": [2, 1, 2, 1]})
        data = nw.from_native(df)
        expected_a_dtype = "Int64"
    elif backend == "polars":
        pl = pytest.importorskip("polars")
        data = pl.DataFrame({"a": [1, 2, 1, 2], "b": [2, 1, 2, 1]})
        expected_a_dtype = "int64"
    elif backend == "pyarrow":
        pa = pytest.importorskip("pyarrow")
        data = pa.table({"a": [1, 2, 1, 2], "b": [2, 1, 2, 1]})
        expected_a_dtype = "int64"

    unmigrated = UnmigratedDiscovery()
    res = unmigrated.fit(data)
    assert res is unmigrated
    fitted_df = captured["df"]
    assert list(fitted_df.columns) == ["a", "b"]
    assert list(fitted_df["a"]) == [1, 2, 1, 2]
    assert list(fitted_df["b"]) == [2, 1, 2, 1]
    assert str(fitted_df["a"].dtype) == expected_a_dtype


@pytest.mark.parametrize("backend", ["polars", "pyarrow"])
def test_score_dataframe_backends(backend):
    pandas_df = pd.DataFrame({"a": [1, 2, 1, 2], "b": [2, 1, 2, 1]})
    est = ChowLiu(n_jobs=1).fit(pandas_df)
    score_pd = est.score(pandas_df)

    if backend == "polars":
        pl = pytest.importorskip("polars")
        data = pl.DataFrame(pandas_df)
    elif backend == "pyarrow":
        pa = pytest.importorskip("pyarrow")
        data = pa.Table.from_pandas(pandas_df)

    assert np.isclose(est.score(data), score_pd)


def test_tree_search_mixin_weights():
    arr = np.array([[1, 2], [2, 1], [1, 2], [2, 1]])
    weights = _TreeSearchMixin._get_weights(arr, n_jobs=1, show_progress=False)
    assert weights.shape == (2, 2)
    assert np.allclose(np.diag(weights), 0)
    assert np.allclose(weights, weights.T)
    assert np.isclose(weights[0, 1], np.log(2))


def test_fit_and_score_integer_columns():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            0: rng.choice([0, 1], size=100, p=[0.8, 0.2]),
            1: rng.choice([0, 1], size=100, p=[0.3, 0.7]),
            2: rng.choice([0, 1], size=100, p=[0.5, 0.5]),
        }
    )
    est = ChowLiu(root_node="x2", n_jobs=1).fit(df)
    assert list(est.feature_names_in_) == ["x0", "x1", "x2"]
    assert est.n_features_in_ == 3
    assert not hasattr(est, "_fit_columns")
    assert list(est.causal_graph_.nodes())[0] == "x2"

    # Verify score works and maps NumPy array columns to the fitted columns correctly
    score_df = est.score(df)
    score_arr = est.score(df.to_numpy())
    assert np.isclose(score_df, score_arr)

    # Refitting with integer columns updates feature_names_in_ to defaults
    est_str = ChowLiu(n_jobs=1).fit(pd.DataFrame({"a": [1, 2], "b": [2, 1]}))
    assert list(est_str.feature_names_in_) == ["a", "b"]
    est_str.fit(df)
    assert list(est_str.feature_names_in_) == ["x0", "x1", "x2"]

    # Wrapping with Narwhals frame sets default feature_names_in_
    est_str_nw = ChowLiu(n_jobs=1).fit(pd.DataFrame({"a": [1, 2], "b": [2, 1]}))
    assert list(est_str_nw.feature_names_in_) == ["a", "b"]
    est_str_nw.fit(nw.from_native(df))
    assert list(est_str_nw.feature_names_in_) == ["x0", "x1", "x2"]

    # Fallback in score() when feature_names_in_ is absent
    est_fallback = ChowLiu(n_jobs=1).fit(pd.DataFrame({"a": [1, 2], "b": [2, 1]}))
    del est_fallback.feature_names_in_
    assert est_fallback.score(np.array([[1, 2], [2, 1]])) is not None


def test_tree_search_preserve_zero_weight_edges():
    columns = ["A", "B", "C"]
    weights = np.zeros((3, 3))
    dag = _TreeSearchMixin._create_tree_and_dag(weights, columns, root_node="A")
    assert set(dag.nodes()) == set(columns)
    assert len(dag.edges()) == len(columns) - 1

    df = pd.DataFrame({"A": [0, 1, 0, 1], "B": [1, 1, 1, 1], "C": [0, 0, 1, 1]})
    est = ChowLiu(edge_weights_fn=lambda x, y: 0.0, n_jobs=1).fit(df)
    assert set(est.causal_graph_.nodes()) == {"A", "B", "C"}
    assert est.adjacency_matrix_.shape == (3, 3)

    # Permuted integer labels and single-column retention
    dag_perm = _TreeSearchMixin._create_tree_and_dag(np.ones((3, 3)), [1, 0, 2], root_node=0)
    assert set(dag_perm.nodes()) == {1, 0, 2}
    assert len(dag_perm.edges()) == 2

    dag_single = _TreeSearchMixin._create_tree_and_dag(np.zeros((1, 1)), [0], root_node=0)
    assert set(dag_single.nodes()) == {0}
    assert len(dag_single.edges()) == 0


def test_pyarrow_missing_fallback_in_bridge_and_score(monkeypatch):
    class UnmigratedDiscovery(BaseCausalDiscovery):
        def _fit(self, X):
            assert isinstance(X, pd.DataFrame)
            self.causal_graph_ = DAG([("a", "b")])
            return self

    df = pd.DataFrame({"a": [1, 2, 1, 2], "b": [2, 1, 2, 1]})
    nw_df = nw.from_native(df)

    def raise_module_not_found(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'pyarrow'")

    monkeypatch.setattr(nw, "to_native", lambda x: "not_a_df")
    monkeypatch.setattr(nw.DataFrame, "to_pandas", raise_module_not_found)

    # Test bridge fallback in fit
    est = UnmigratedDiscovery().fit(nw_df)
    assert est.causal_graph_ is not None

    # Test bridge fallback in score
    score = est.score(nw_df)
    assert isinstance(score, float)


def test_unmigrated_estimators_integer_columns(monkeypatch):
    x = np.linspace(0.1, 2.0, 30)
    y = np.exp(x)
    df_biv = pd.DataFrame({0: x, 1: y})

    rng = np.random.default_rng(42)
    df_multi = pd.DataFrame(
        rng.normal(size=(50, 3)),
        columns=[0, 1, 2],
    )

    # Bivariate estimators
    ANM().fit(df_biv)
    IGCI().fit(df_biv)
    llm = LLMPairwise()
    monkeypatch.setattr(llm, "_query_llm", lambda msg: "1")
    llm.fit(df_biv)
    assert ("x0", "x1") in llm.causal_graph_.edges()

    # Multivariate estimators
    VarSort().fit(df_multi)
    R2Sort().fit(df_multi)
    SP().fit(df_multi)
    be = BootstrapEstimator(estimator=ChowLiu(root_node="x0", show_progress=False), n_bootstraps=2)
    be.fit(pd.DataFrame(rng.integers(0, 2, size=(30, 3)), columns=[0, 1, 2]))
    adj = be.get_adjacency_matrix(0.5)
    assert adj.shape == (3, 3)
