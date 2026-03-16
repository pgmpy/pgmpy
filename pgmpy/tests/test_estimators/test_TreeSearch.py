import networkx as nx
import numpy as np
import pandas as pd
import pytest
from joblib.externals.loky import get_reusable_executor

from pgmpy.estimators import TreeSearch
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.utils import get_example_model


@pytest.fixture
def data12():
    # set random seed
    np.random.seed(0)

    # test data for chow-liu
    return pd.DataFrame(
        np.random.randint(low=0, high=2, size=(100, 5)),
        columns=["A", "B", "C", "D", "E"],
    )


@pytest.fixture
def data13():
    np.random.seed(0)
    # test data for chow-liu
    model = DiscreteBayesianNetwork(
        [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F")]
    )
    cpd_a = TabularCPD("A", 2, [[0.4], [0.6]])
    cpd_b = TabularCPD(
        "B",
        3,
        [[0.6, 0.2], [0.3, 0.5], [0.1, 0.3]],
        evidence=["A"],
        evidence_card=[2],
    )
    cpd_c = TabularCPD(
        "C", 2, [[0.3, 0.4], [0.7, 0.6]], evidence=["A"], evidence_card=[2]
    )
    cpd_d = TabularCPD(
        "D",
        3,
        [[0.5, 0.3, 0.1], [0.4, 0.4, 0.8], [0.1, 0.3, 0.1]],
        evidence=["B"],
        evidence_card=[3],
    )
    cpd_e = TabularCPD(
        "E",
        2,
        [[0.3, 0.5, 0.2], [0.7, 0.5, 0.8]],
        evidence=["B"],
        evidence_card=[3],
    )
    cpd_f = TabularCPD(
        "F",
        3,
        [[0.3, 0.6], [0.5, 0.2], [0.2, 0.2]],
        evidence=["C"],
        evidence_card=[2],
    )

    model.add_cpds(cpd_a, cpd_b, cpd_c, cpd_d, cpd_e, cpd_f)
    inference = BayesianModelSampling(model)
    return inference.forward_sample(size=10000)


@pytest.fixture
def data22():
    np.random.seed(0)
    # test data for TAN
    model = DiscreteBayesianNetwork(
        [
            ("A", "R"),
            ("A", "B"),
            ("A", "C"),
            ("A", "D"),
            ("A", "E"),
            ("R", "B"),
            ("R", "C"),
            ("R", "D"),
            ("R", "E"),
        ]
    )
    cpd_a = TabularCPD("A", 2, [[0.7], [0.3]])
    cpd_r = TabularCPD(
        "R",
        3,
        [[0.6, 0.2], [0.3, 0.5], [0.1, 0.3]],
        evidence=["A"],
        evidence_card=[2],
    )
    cpd_b = TabularCPD(
        "B",
        3,
        [
            [0.1, 0.1, 0.2, 0.2, 0.7, 0.1],
            [0.1, 0.3, 0.1, 0.2, 0.1, 0.2],
            [0.8, 0.6, 0.7, 0.6, 0.2, 0.7],
        ],
        evidence=["A", "R"],
        evidence_card=[2, 3],
    )
    cpd_c = TabularCPD(
        "C",
        2,
        [[0.7, 0.2, 0.2, 0.5, 0.1, 0.3], [0.3, 0.8, 0.8, 0.5, 0.9, 0.7]],
        evidence=["A", "R"],
        evidence_card=[2, 3],
    )
    cpd_d = TabularCPD(
        "D",
        3,
        [
            [0.3, 0.8, 0.2, 0.8, 0.4, 0.7],
            [0.4, 0.1, 0.4, 0.1, 0.1, 0.1],
            [0.3, 0.1, 0.4, 0.1, 0.5, 0.2],
        ],
        evidence=["A", "R"],
        evidence_card=[2, 3],
    )
    cpd_e = TabularCPD(
        "E",
        2,
        [[0.5, 0.6, 0.6, 0.5, 0.5, 0.4], [0.5, 0.4, 0.4, 0.5, 0.5, 0.6]],
        evidence=["A", "R"],
        evidence_card=[2, 3],
    )
    model.add_cpds(cpd_a, cpd_r, cpd_b, cpd_c, cpd_d, cpd_e)
    inference = BayesianModelSampling(model)
    return inference.forward_sample(size=10000)


@pytest.fixture
def alarm_df():
    return get_example_model("alarm").simulate(int(1e4), seed=42)


# ============================================================================
# TESTS FOR SKLEARN-COMPATIBLE API (NEW)
# ============================================================================


class TestTreeSearchSklearnAPI:
    """Test sklearn compatibility of refactored TreeSearch."""

    def test_fit_returns_self(self, data12):
        """Test that fit() returns self for method chaining."""
        ts = TreeSearch()
        result = ts.fit(data12)
        assert result is ts

    def test_fit_creates_model_attribute(self, data12):
        """Test that fit() creates model_ attribute."""
        ts = TreeSearch(root_node="A", estimator_type="chow-liu")
        ts.fit(data12)
        assert hasattr(ts, "model_")
        assert ts.model_ is not None

    def test_fit_creates_root_node_attribute(self, data12):
        """Test that fit() creates root_node_ attribute."""
        ts = TreeSearch(estimator_type="chow-liu")
        ts.fit(data12)
        assert hasattr(ts, "root_node_")
        assert ts.root_node_ in data12.columns

    def test_fit_with_dataframe(self, data12):
        """Test fit() accepts pandas DataFrame."""
        ts = TreeSearch(root_node="A", estimator_type="chow-liu")
        ts.fit(data12)
        assert ts.model_ is not None

    def test_fit_type_error_with_non_dataframe(self, data12):
        """Test fit() raises TypeError with non-DataFrame input."""
        ts = TreeSearch()
        with pytest.raises(TypeError):
            ts.fit(data12.values)  # numpy array instead of DataFrame

    def test_get_params(self, data12):
        """Test get_params() returns all parameters."""
        params = {
            "estimator_type": "chow-liu",
            "root_node": "A",
            "n_jobs": 2,
            "show_progress": False,
        }
        ts = TreeSearch(**params)
        retrieved = ts.get_params()

        for key, value in params.items():
            assert retrieved[key] == value

    def test_get_params_deep(self, data12):
        """Test get_params() with deep=True."""
        ts = TreeSearch(root_node="A", edge_weights_fn="mutual_info")
        params = ts.get_params(deep=True)
        assert "root_node" in params
        assert "edge_weights_fn" in params

    def test_set_params(self, data12):
        """Test set_params() updates parameters."""
        ts = TreeSearch()
        result = ts.set_params(root_node="B", n_jobs=2)

        assert ts.root_node == "B"
        assert ts.n_jobs == 2
        assert result is ts  # Check method chaining

    def test_set_params_method_chaining(self, data12):
        """Test set_params() allows method chaining."""
        ts = TreeSearch().set_params(root_node="A", estimator_type="chow-liu")
        assert ts.root_node == "A"
        assert ts.estimator_type == "chow-liu"

    def test_set_params_invalid_params(self, data12):
        """Test set_params() raises error on invalid parameters."""
        ts = TreeSearch()
        with pytest.raises(ValueError):
            ts.set_params(invalid_param="value")

    def test_set_params_returns_self(self, data12):
        """Test set_params() returns self."""
        ts = TreeSearch()
        result = ts.set_params(root_node="A")
        assert result is ts

    @pytest.mark.parametrize(
        "estimator_type",
        ["chow-liu", "tan"],
    )
    def test_fit_chow_liu_sklearn(self, data12, estimator_type):
        """Test sklearn-compatible fit for Chow-Liu."""
        if estimator_type == "tan":
            ts = TreeSearch(
                estimator_type=estimator_type,
                root_node="B",
                class_node="A",
                show_progress=False,
            )
        else:
            ts = TreeSearch(
                estimator_type=estimator_type,
                root_node="A",
                show_progress=False,
            )

        ts.fit(data12)

        assert ts.model_ is not None
        assert set(ts.model_.nodes()) == set(data12.columns)

    def test_fit_chow_liu_auto_root(self, data12):
        """Test sklearn fit with auto-selected root node."""
        ts = TreeSearch(estimator_type="chow-liu", show_progress=False)
        ts.fit(data12)

        assert ts.model_ is not None
        assert ts.root_node_ is not None
        assert ts.root_node_ in data12.columns

    def test_fit_tan_requires_class_node(self, data12):
        """Test that TAN requires class_node."""
        ts = TreeSearch(estimator_type="tan")
        with pytest.raises(ValueError):
            ts.fit(data12)

    def test_fit_invalid_root_node(self, data12):
        """Test error on invalid root_node."""
        ts = TreeSearch(root_node="invalid_node", estimator_type="chow-liu")
        with pytest.raises(ValueError):
            ts.fit(data12)

    def test_fit_invalid_class_node(self, data12):
        """Test error on invalid class_node."""
        ts = TreeSearch(
            estimator_type="tan",
            class_node="invalid_node",
            root_node="A",
        )
        with pytest.raises(ValueError):
            ts.fit(data12)

    def test_fit_invalid_estimator_type(self, data12):
        """Test error on invalid estimator_type."""
        ts = TreeSearch(estimator_type="invalid")
        with pytest.raises(ValueError):
            ts.fit(data12)

    def test_sklearn_pipeline_integration(self, data12):
        """Test integration with sklearn Pipeline."""
        from sklearn.pipeline import Pipeline

        # Create a simple pipeline with TreeSearch
        pipeline = Pipeline([("tree_search", TreeSearch(estimator_type="chow-liu"))])
        pipeline.fit(data12)

        assert hasattr(pipeline.named_steps["tree_search"], "model_")

    def test_sklearn_param_grid(self, data12):
        """Test parameter grid generation for GridSearchCV."""
        param_grid = {
            "root_node": ["A", "B"],
            "edge_weights_fn": ["mutual_info", "adjusted_mutual_info"],
        }

        # Generate parameter combinations
        from sklearn.model_selection import ParameterGrid

        grid = ParameterGrid(param_grid)
        assert len(list(grid)) == 4

        # Test that each combination can be set
        ts = TreeSearch(estimator_type="chow-liu")
        for params in grid:
            ts.set_params(**params)
            assert ts.root_node in params.values()

    def test_fit_y_parameter_ignored(self, data12):
        """Test that y parameter is ignored (sklearn convention)."""
        ts = TreeSearch(root_node="A", estimator_type="chow-liu")
        # y should be ignored
        ts.fit(data12, y=None)
        assert ts.model_ is not None

    def test_multiple_fit_calls(self, data12, data13):
        """Test that multiple fit() calls update the model."""
        ts = TreeSearch(root_node="A", estimator_type="chow-liu")

        # First fit
        ts.fit(data12)
        first_model = ts.model_

        # Second fit with different data
        ts.fit(data13)
        second_model = ts.model_

        # Models should be different
        assert set(first_model.nodes()) != set(second_model.nodes())


# ============================================================================
# TESTS FOR BACKWARD COMPATIBILITY (OLD API)
# ============================================================================


@pytest.mark.parametrize(
    "weight_fn", ["mutual_info", "adjusted_mutual_info", "normalized_mutual_info"]
)
@pytest.mark.parametrize("n_jobs", [2, 1])
def test_estimate_chow_liu(data12, data13, weight_fn, n_jobs):
    """Test backward compatibility with old estimate() method."""
    # Test with data12
    # FIXED: show_progress is now a parameter of __init__, not fit()
    est = TreeSearch(
        estimator_type="chow-liu",
        root_node="A",
        n_jobs=n_jobs,
        show_progress=False,
    )
    est.fit(data12)
    dag = est.model_

    # check number of nodes and edges are as expected
    assert set(dag.nodes()) == {"A", "B", "C", "D", "E"}
    assert nx.is_tree(dag)

    # learn tree structure using A as root node
    est = TreeSearch(
        estimator_type="chow-liu",
        root_node="A",
        n_jobs=n_jobs,
        show_progress=False,
    )
    est.fit(data13)
    dag = est.model_

    # check number of nodes and edges are as expected
    assert set(dag.nodes()) == {"A", "B", "C", "D", "E", "F"}
    assert set(dag.edges()) == {
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("B", "E"),
        ("C", "F"),
    }

    # check tree structure exists
    assert dag.has_edge("A", "B")
    assert dag.has_edge("A", "C")
    assert dag.has_edge("B", "D")
    assert dag.has_edge("B", "E")
    assert dag.has_edge("C", "F")


@pytest.mark.parametrize(
    "weight_fn", ["mutual_info", "adjusted_mutual_info", "normalized_mutual_info"]
)
@pytest.mark.parametrize("n_jobs", [2, 1])
def test_estimate_tan(data22, weight_fn, n_jobs):
    """Test backward compatibility with TAN."""
    # learn graph structure
    # FIXED: show_progress is now a parameter of __init__, not fit()
    est = TreeSearch(
        estimator_type="tan",
        root_node="R",
        class_node="A",
        n_jobs=n_jobs,
        show_progress=False,
    )
    est.fit(data22)
    dag = est.model_

    # check number of nodes and edges are as expected
    assert set(dag.nodes()) == {"A", "B", "C", "D", "E", "R"}
    assert set(dag.edges()) == {
        ("A", "B"),
        ("A", "C"),
        ("A", "D"),
        ("A", "E"),
        ("A", "R"),
        ("R", "B"),
        ("R", "C"),
        ("R", "D"),
        ("R", "E"),
    }

    # check directed edge between class and independent variables
    assert dag.has_edge("A", "B")
    assert dag.has_edge("A", "C")
    assert dag.has_edge("A", "D")
    assert dag.has_edge("A", "E")

    # check tree structure exists over independent variables
    assert dag.has_edge("R", "B")
    assert dag.has_edge("R", "C")
    assert dag.has_edge("R", "D")
    assert dag.has_edge("R", "E")


def test_estimate_chow_liu_auto_root_node(data12):
    """Test backward compatibility with auto root node selection."""
    # learn tree structure using auto root node
    est = TreeSearch(estimator_type="chow-liu", show_progress=False)
    est.fit(data12)

    # root node selection
    weights = est._get_weights(data12)
    sum_weights = weights.sum(axis=0)
    maxw_idx = np.argsort(sum_weights)[::-1]
    root_node = data12.columns[maxw_idx[0]]

    dag = est.model_
    nodes = list(dag.nodes())
    assert nodes[0] == root_node
    assert nodes == ["D", "A", "C", "B", "E"]


def test_estimate_tan_auto_class_node(data22):
    """Test backward compatibility with auto class node selection."""
    # FIXED: TAN requires class_node to be specified
    # For auto-selection, we need to compute it first

    # learn tree structure using auto root node
    est = TreeSearch(estimator_type="chow-liu", show_progress=False)
    est.fit(data22)

    # root and class node selection based on edge weights
    weights = est._get_weights(data22)
    sum_weights = weights.sum(axis=0)
    maxw_idx = np.argsort(sum_weights)[::-1]
    root_node = data22.columns[maxw_idx[0]]
    class_node = data22.columns[maxw_idx[1]]

    # Now fit TAN with selected class node
    est_tan = TreeSearch(
        estimator_type="tan",
        root_node=root_node,
        class_node=class_node,
        show_progress=False,
    )
    est_tan.fit(data22)
    dag = est_tan.model_

    nodes = list(dag.nodes())
    assert nodes[0] == root_node
    assert nodes[-1] == class_node
    assert sorted(nodes) == sorted(["C", "R", "A", "D", "E", "B"])


def test_tan_real_dataset(alarm_df):
    """Test TAN on real dataset."""
    # Expected values taken from bnlearn.
    expected_edges = [
        ("CVP", "LVFAILURE"),
        ("CVP", "INTUBATION"),
        ("CVP", "TPR"),
        ("CVP", "DISCONNECT"),
        ("CVP", "VENTMACH"),
        ("CVP", "HR"),
        ("CVP", "FIO2"),
        ("CVP", "HRBP"),
        ("CVP", "VENTLUNG"),
        ("CVP", "PAP"),
        ("CVP", "HISTORY"),
        ("CVP", "PCWP"),
        ("CVP", "INSUFFANESTH"),
        ("CVP", "SAO2"),
        ("CVP", "EXPCO2"),
        ("CVP", "PRESS"),
        ("CVP", "PULMEMBOLUS"),
        ("CVP", "ARTCO2"),
        ("CVP", "MINVOLSET"),
        ("LVFAILURE", "HISTORY"),
        ("LVFAILURE", "PCWP"),
        ("INTUBATION", "INSUFFANESTH"),
        ("EXPCO2", "INTUBATION"),
        ("HR", "TPR"),
        ("PRESS", "DISCONNECT"),
        ("VENTLUNG", "VENTMACH"),
        ("VENTMACH", "PRESS"),
        ("VENTMACH", "MINVOLSET"),
        ("HR", "HRBP"),
        ("ARTCO2", "HR"),
        ("SAO2", "FIO2"),
        ("VENTLUNG", "PAP"),
        ("PCWP", "VENTLUNG"),
        ("VENTLUNG", "EXPCO2"),
        ("VENTLUNG", "ARTCO2"),
        ("PAP", "PULMEMBOLUS"),
        ("ARTCO2", "SAO2"),
    ]
    features = [
        "LVFAILURE",
        "INTUBATION",
        "TPR",
        "DISCONNECT",
        "VENTMACH",
        "HR",
        "FIO2",
        "HRBP",
        "VENTLUNG",
        "PAP",
        "HISTORY",
        "PCWP",
        "INSUFFANESTH",
        "SAO2",
        "EXPCO2",
        "PRESS",
        "PULMEMBOLUS",
        "ARTCO2",
        "MINVOLSET",
    ]
    target = "CVP"
    est = TreeSearch(
        estimator_type="tan",
        root_node=features[0],
        class_node=target,
        show_progress=False,
    )
    est.fit(alarm_df[features + [target]])
    edges = est.model_.edges()
    assert set(expected_edges) == set(edges)


@pytest.fixture(autouse=True)
def shutdown_executor():
    yield
    get_reusable_executor().shutdown(wait=True)
