"""
Tests for pgmpy.utils.get_example_dataset — specifically verifying the
base_url fix introduced to resolve issue #2567.

Run with:
    pytest tests/test_get_example_dataset.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from pgmpy.utils.get_example_model import (
    get_example_dataset,
    DEFAULT_BASE_URL,
)

class TestDefaultBaseUrl:
    def test_default_url_is_set(self):
        """Issue #2567: DEFAULT_BASE_URL must be defined and non-empty."""
        assert DEFAULT_BASE_URL, "DEFAULT_BASE_URL must not be empty"

    def test_default_url_points_to_raw_github(self):
        """The default URL must point to raw.githubusercontent.com."""
        assert "raw.githubusercontent.com" in DEFAULT_BASE_URL

    def test_default_url_contains_example_datasets(self):
        """The default URL must reference the example_datasets repository."""
        assert "example_datasets" in DEFAULT_BASE_URL

    def test_default_url_ends_with_slash(self):
        """URL must end with / so path joins work correctly."""
        assert DEFAULT_BASE_URL.endswith("/")



class TestBaseUrlParameter:

    @patch("pgmpy.utils.get_example_dataset.pd.read_csv")
    def test_default_base_url_is_used_when_not_specified(self, mock_read_csv):
        """Without an explicit base_url, DEFAULT_BASE_URL should be used."""
        dummy_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        mock_read_csv.return_value = dummy_df

        get_example_dataset("sachs")

        called_url = mock_read_csv.call_args[0][0]
        assert DEFAULT_BASE_URL.rstrip("/") in called_url, (
            f"Expected DEFAULT_BASE_URL in the constructed URL, got: {called_url}"
        )

    @patch("pgmpy.utils.get_example_dataset.pd.read_csv")
    def test_custom_base_url_is_respected(self, mock_read_csv):
        """
        Issue #2567: A custom base_url passed to get_example_dataset()
        must be forwarded to the download URL — not silently ignored.
        """
        dummy_df = pd.DataFrame({"x": [1], "y": [2]})
        mock_read_csv.return_value = dummy_df

        custom_url = "https://raw.githubusercontent.com/myfork/example_datasets/dev/"
        get_example_dataset("sachs", base_url=custom_url)

        called_url = mock_read_csv.call_args[0][0]
        assert "myfork" in called_url, (
            f"Custom base_url was NOT forwarded to the request. Got: {called_url}"
        )

    @patch("pgmpy.utils.get_example_dataset.pd.read_csv")
    def test_base_url_without_trailing_slash_is_normalised(self, mock_read_csv):
        """base_url without a trailing slash must still produce a valid URL."""
        dummy_df = pd.DataFrame({"a": [1]})
        mock_read_csv.return_value = dummy_df

        # Note: no trailing slash
        url_without_slash = "https://raw.githubusercontent.com/pgmpy/example_datasets/main"
        get_example_dataset("sachs", base_url=url_without_slash)

        called_url = mock_read_csv.call_args[0][0]
        # Should NOT produce double-slash or missing separator
        assert "//" not in called_url.replace("https://", ""), (
            f"Normalisation produced a double-slash in URL: {called_url}"
        )
        assert "sachs" in called_url

    @patch("pgmpy.utils.get_example_dataset.pd.read_csv")
    def test_dataset_name_appears_in_constructed_url(self, mock_read_csv):
        """The constructed URL must contain the dataset name."""
        dummy_df = pd.DataFrame({"col": [1]})
        mock_read_csv.return_value = dummy_df

        get_example_dataset("my_dataset")

        called_url = mock_read_csv.call_args[0][0]
        assert "my_dataset" in called_url, (
            f"Dataset name not found in URL: {called_url}"
        )


# ---------------------------------------------------------------------------
# 3.  Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    @patch("pgmpy.utils.get_example_dataset.pd.read_csv", side_effect=Exception("404"))
    def test_raises_value_error_for_unknown_dataset(self, mock_read_csv):
        """A dataset that doesn't exist should raise a clear ValueError."""
        with pytest.raises(ValueError, match="Could not find dataset"):
            get_example_dataset("dataset_that_does_not_exist_xyz")

    @patch("pgmpy.utils.get_example_dataset.pd.read_csv", side_effect=Exception("404"))
    def test_error_message_contains_dataset_name(self, mock_read_csv):
        """ValueError message must include the requested dataset name."""
        name = "nonexistent_dataset"
        with pytest.raises(ValueError, match=name):
            get_example_dataset(name)

    @patch("pgmpy.utils.get_example_dataset.pd.read_csv", side_effect=Exception("404"))
    def test_error_message_contains_base_url(self, mock_read_csv):
        """ValueError message must include the base_url that was tried."""
        custom = "https://raw.githubusercontent.com/custom/example_datasets/main/"
        with pytest.raises(ValueError, match="custom"):
            get_example_dataset("nonexistent", base_url=custom)