import pytest

from pgmpy.base import DAG
from pgmpy.datasets import list_datasets, load_dataset

causalchamber = pytest.importorskip("causalchamber")


class TestCausalChamberDatasets:
    """Tests for CausalChamber datasets."""

    def test_light_tunnel_palette_in_list(self):
        """Test that light_tunnel_palette appears in dataset list."""
        all_datasets = list_datasets()
        assert "light_tunnel_palette" in all_datasets

    def test_light_tunnel_pure_colors_bright_in_list(self):
        """Test that light_tunnel_pure_colors_bright appears in dataset list."""
        all_datasets = list_datasets()
        assert "light_tunnel_pure_colors_bright" in all_datasets

    def test_causalchamber_datasets_have_ground_truth(self):
        """Test that CausalChamber datasets show up with ground truth filter."""
        datasets_with_gt = list_datasets(has_ground_truth=True)
        assert "light_tunnel_palette" in datasets_with_gt
        assert "light_tunnel_pure_colors_bright" in datasets_with_gt

    def test_load_light_tunnel_palette(self):
        """Test loading light_tunnel_palette dataset."""
        dataset = load_dataset("light_tunnel_palette")

        # Check basic properties
        assert dataset.name == "light_tunnel_palette"
        assert dataset.data.shape == (224, 52)
        assert dataset.tags["n_samples"] == 224
        assert dataset.tags["n_variables"] == 52

        # Check ground truth
        assert dataset.ground_truth is not None
        assert isinstance(dataset.ground_truth, DAG)
        assert len(dataset.ground_truth.nodes()) == 38
        assert len(dataset.ground_truth.edges()) == 57

        # Check no expert knowledge
        assert dataset.expert_knowledge is None

    def test_load_light_tunnel_pure_colors_bright(self):
        """Test loading light_tunnel_pure_colors_bright dataset."""
        dataset = load_dataset("light_tunnel_pure_colors_bright")

        # Check basic properties
        assert dataset.name == "light_tunnel_pure_colors_bright"
        assert dataset.data.shape == (3, 52)
        assert dataset.tags["n_samples"] == 3
        assert dataset.tags["n_variables"] == 52

        # Check ground truth
        assert dataset.ground_truth is not None
        assert isinstance(dataset.ground_truth, DAG)
        assert len(dataset.ground_truth.nodes()) == 38
        assert len(dataset.ground_truth.edges()) == 57

    def test_ground_truth_structure(self):
        """Test that ground truth DAG has expected causal variables."""
        dataset = load_dataset("light_tunnel_palette")
        dag = dataset.ground_truth

        # Check for some expected variables from the light tunnel
        expected_vars = ["red", "green", "blue", "current", "angle_1", "angle_2"]
        for var in expected_vars:
            assert var in dag.nodes(), f"Expected variable {var} not in ground truth"

        # Check for some expected causal relationships
        # current -> RGB colors
        assert dag.has_edge("current", "red")
        assert dag.has_edge("current", "green")
        assert dag.has_edge("current", "blue")

    def test_data_columns(self):
        """Test that data has expected columns."""
        dataset = load_dataset("light_tunnel_palette")
        df = dataset.data

        # Check for metadata columns
        assert "timestamp" in df.columns
        assert "config" in df.columns
        assert "intervention" in df.columns

        # Check for some causal variables
        assert "red" in df.columns
        assert "green" in df.columns
        assert "blue" in df.columns
