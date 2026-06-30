import pandas as pd

from pgmpy.base import DAG
from pgmpy.datasets import list_datasets, load_dataset


def test_load_linear_gaussian_dataset():
    # Default parameters produce a 1000x5 DataFrame with a matching DAG.
    ds = load_dataset("linear_gaussian", seed=42)
    assert ds.data.shape == (1000, 5)
    assert set(ds.ground_truth.nodes()) == set(ds.data.columns)
    assert isinstance(ds.ground_truth, DAG)

    # Custom n_nodes and n_samples control output dimensions.
    ds_custom = load_dataset("linear_gaussian", n_samples=200, seed=42, n_nodes=8, edge_prob=0.3)
    assert ds_custom.data.shape == (200, 8)
    assert set(ds_custom.data.columns) == set(ds_custom.ground_truth.nodes())

    # Same seed produces identical data and graph.
    ds_repeat = load_dataset("linear_gaussian", seed=42)
    pd.testing.assert_frame_equal(ds.data, ds_repeat.data)
    assert set(ds.ground_truth.edges()) == set(ds_repeat.ground_truth.edges())

    # scale changes generated data but not the ground-truth graph structure.
    ds_low = load_dataset("linear_gaussian", seed=42, scale=0.1)
    ds_high = load_dataset("linear_gaussian", seed=42, scale=10.0)
    assert not ds_low.data.equals(ds_high.data)
    assert set(ds_low.ground_truth.edges()) == set(ds_high.ground_truth.edges())

    # edge_prob=0 produces a DAG with all isolated nodes and no edges.
    ds_empty = load_dataset("linear_gaussian", seed=42, n_nodes=8, edge_prob=0)
    assert set(ds_empty.ground_truth.nodes()) == {f"X_{i}" for i in range(8)}
    assert not ds_empty.ground_truth.edges()

    # The dataset appears in list_datasets with the is_simulated tag.
    assert "linear_gaussian" in list_datasets(is_simulated=True)


def test_linear_gaussian_dataset_tags():
    # load_dataset attaches the simulator's class tags to the returned Dataset,
    # with n_variables and n_samples populated from the generated data.
    ds = load_dataset("linear_gaussian", seed=42)
    assert ds.tags == {
        "name": "linear_gaussian",
        "n_variables": 5,
        "n_samples": 1000,
        "has_ground_truth": True,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": True,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    # n_variables and n_samples track the requested dimensions.
    ds_custom = load_dataset("linear_gaussian", n_samples=200, seed=7, n_nodes=8, edge_prob=0.3)
    assert ds_custom.tags["n_variables"] == 8
    assert ds_custom.tags["n_samples"] == 200
