import numpy as np
import pandas as pd
import pytest

from pgmpy.base import DAG
from pgmpy.benchmarking._base import _get_dataset_labels, _resolve_dataset
from pgmpy.datasets._base import Dataset
from pgmpy.datasets.anm import AdditiveNoiseModel
from pgmpy.datasets.linear_gaussian_scm import LinearGaussianSCM


class TestBenchmarkDatasetResolution:
    def test_get_dataset_labels(self):
        sim = AdditiveNoiseModel(n_nodes=8, seed=1)
        ds = Dataset(name="prepared", data=pd.DataFrame({"a": [1]}))
        df = pd.DataFrame(np.zeros((3, 2)))

        labels = _get_dataset_labels(["sachs_discrete", sim, ds, df])
        assert labels == [
            "sachs_discrete",
            "AdditiveNoiseModel(n_nodes=8, seed=1)",
            "prepared",
            "DataFrame_3(3x2)",
        ]

        sim_multi = AdditiveNoiseModel(
            function_type=(np.sin, np.cos, np.tan, np.sinh, np.cosh, np.tanh, np.exp, np.log)
        )
        assert "\n" in repr(sim_multi)
        assert _get_dataset_labels([sim_multi]) == [
            "AdditiveNoiseModel(function_type=(<ufunc 'sin'>, <ufunc 'cos'>, <ufunc 'tan'>, "
            "<ufunc 'sinh'>, <ufunc 'cosh'>, <ufunc 'tanh'>, <ufunc 'exp'>, <ufunc 'log'>))"
        ]

        sim_a = LinearGaussianSCM(seed=1)
        sim_b = LinearGaussianSCM(seed=1)
        assert _get_dataset_labels([sim_a, sim_b]) == [
            "LinearGaussianSCM(seed=1)",
            "LinearGaussianSCM(seed=1)_1",
        ]

        assert _get_dataset_labels(["a", "a", "a_0"]) == ["a", "a_1", "a_0"]
        assert _get_dataset_labels(["a", "a", "a_1"]) == ["a", "a_1", "a_1_2"]

        df1 = pd.DataFrame(np.zeros((3, 2)))
        df2 = pd.DataFrame(np.zeros((3, 2)))
        ds1 = Dataset(name="same", data=pd.DataFrame({"x": [1]}))
        ds2 = Dataset(name="same", data=pd.DataFrame({"y": [2]}))
        assert _get_dataset_labels([df1, df2, ds1, ds2]) == [
            "DataFrame_0(3x2)",
            "DataFrame_1(3x2)",
            "same",
            "same_3",
        ]

        for bad_input, received in [
            (42, "int"),
            ([1, 2], "list"),
            (AdditiveNoiseModel, "the class AdditiveNoiseModel"),
        ]:
            with pytest.raises(TypeError, match=f"got {received}"):
                _get_dataset_labels([bad_input])

    def test_resolve_dataset(self):
        registered = [
            ("sachs_discrete", 100, (100, 11), 11, 20),
            ("linear_gaussian", 200, (200, 5), 5, 5),
            ("lead", 1000, (1000, 7), None, None),
        ]
        for name, n_samples, expected_shape, n_nodes, n_edges in registered:
            ds = _resolve_dataset(name, label=name, n_samples=n_samples, seed=42)
            assert ds.name == name
            assert ds.data.shape == expected_shape
            assert ds.tags["n_samples"] == expected_shape[0]
            assert ds.tags["n_variables"] == expected_shape[1]
            if n_nodes is not None:
                assert len(ds.ground_truth.nodes()) == n_nodes
                assert len(ds.ground_truth.edges()) == n_edges
            else:
                assert ds.ground_truth is None
            ds2 = _resolve_dataset(name, label=name, n_samples=n_samples, seed=42)
            assert np.array_equal(ds.data.values, ds2.data.values)

        tubingen_ds = _resolve_dataset("tubingen/1", label="tubingen/1", n_samples=None, seed=42)
        assert tubingen_ds.name == "tubingen/1"
        assert tubingen_ds.data.shape == (349, 2)
        assert list(tubingen_ds.data.columns) == ["x", "y"]
        assert tubingen_ds.data.iloc[0, 0] == 205.0 and tubingen_ds.data.iloc[0, 1] == 9.7
        assert tubingen_ds.data.iloc[-1, 0] == 615.0 and tubingen_ds.data.iloc[-1, 1] == 6.2
        assert set(tubingen_ds.ground_truth.nodes()) == {"x", "y"}
        assert set(tubingen_ds.ground_truth.edges()) == {("x", "y")}

        sim = AdditiveNoiseModel(n_nodes=8, seed=1)
        ds_sim = _resolve_dataset(sim, label="anm_8", n_samples=50, seed=999)
        assert ds_sim.name == "anm_8"
        assert ds_sim.data.shape == (50, 8)
        assert set(ds_sim.ground_truth.nodes()) == {f"X_{i}" for i in range(8)}
        assert set(ds_sim.ground_truth.edges()) == {
            ("X_0", "X_7"),
            ("X_1", "X_2"),
            ("X_1", "X_4"),
            ("X_1", "X_6"),
            ("X_2", "X_6"),
            ("X_2", "X_7"),
            ("X_4", "X_2"),
            ("X_4", "X_3"),
            ("X_4", "X_6"),
            ("X_5", "X_0"),
            ("X_5", "X_3"),
            ("X_5", "X_4"),
            ("X_5", "X_6"),
            ("X_6", "X_3"),
        }
        assert ds_sim.tags["n_samples"] == 50
        assert ds_sim.tags["n_variables"] == 8
        assert np.array_equal(ds_sim.data.values, sim.load_dataframe(n_samples=50).values)

        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        ds_obj = Dataset(name="mine", data=df)
        resolved_ds = _resolve_dataset(ds_obj, label="mine", n_samples=2, seed=42)
        assert resolved_ds is ds_obj
        assert resolved_ds.name == "mine"
        assert resolved_ds.data.shape == (3, 2)

        resolved_df = _resolve_dataset(df, label="DataFrame_0(3x2)", n_samples=2, seed=42)
        assert resolved_df.data is df
        assert resolved_df.name == "DataFrame_0(3x2)"
        assert resolved_df.data.shape == (3, 2)
        assert resolved_df.ground_truth is None

        gt = DAG([("a", "b")])
        ds_nogt = Dataset(name="no_gt", data=pd.DataFrame({"a": [1], "b": [2]}))
        resolved_gt = _resolve_dataset(ds_nogt, label="no_gt", n_samples=None, seed=None, ground_truth={"no_gt": gt})
        assert resolved_gt.ground_truth is gt
        assert resolved_gt.tags["has_ground_truth"] is True
        assert ds_nogt.ground_truth is None
        assert ds_nogt.tags is None

        df_gt = pd.DataFrame({"a": [1], "b": [2]})
        assert (
            _resolve_dataset(df_gt, label="my_df", n_samples=None, seed=None, ground_truth={"my_df": gt}).ground_truth
            is gt
        )

        lead_gt = _resolve_dataset("lead", label="lead", n_samples=100, seed=42, ground_truth={"lead": gt})
        assert lead_gt.ground_truth is gt
        assert lead_gt.tags["has_ground_truth"] is True
        assert lead_gt.tags["n_samples"] == 100

        sim_5 = AdditiveNoiseModel(n_nodes=5, seed=1)
        assert (
            _resolve_dataset(sim_5, label="anm", n_samples=50, seed=None, ground_truth={"anm": gt}).ground_truth
            is not gt
        )

        assert (
            _resolve_dataset(
                ds_nogt, label="no_gt", n_samples=None, seed=None, ground_truth={"other_key": gt}
            ).ground_truth
            is None
        )

        with pytest.warns(UserWarning, match="Requested"):
            ds_excess = _resolve_dataset("sachs_discrete", label="sachs_discrete", n_samples=999999, seed=42)
        assert ds_excess.name == "sachs_discrete"
        assert ds_excess.data.shape[0] == 5400

        with pytest.raises(ValueError, match="not found"):
            _resolve_dataset("no_such_dataset", label="no_such_dataset", n_samples=100, seed=42)
