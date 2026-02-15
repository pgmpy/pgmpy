"""Tests for LGBN JSON schema validation."""

import os
import unittest

import pytest

try:
    from jsonschema import Draft202012Validator  # noqa: F401

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

from pgmpy.schemas import get_lgbn_schema, validate_lgbn_file, validate_lgbn_json


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
class TestLGBNSchema(unittest.TestCase):
    """Test cases for LGBN JSON schema validation."""

    def test_valid_simple_model(self):
        """Test validation of a simple valid model."""
        data = {
            "nodes": ["A", "B", "C"],
            "arcs": [["A", "B"], ["B", "C"]],
            "cpds": {
                "A": {
                    "coefficients": {"(Intercept)": [1.0]},
                    "variance": [4.0],
                    "parents": [],
                },
                "B": {
                    "coefficients": {"(Intercept)": [-5.0], "A": [0.5]},
                    "variance": [4.0],
                    "parents": ["A"],
                },
                "C": {
                    "coefficients": {"(Intercept)": [4.0], "B": [-1.0]},
                    "variance": [3.0],
                    "parents": ["B"],
                },
            },
        }
        is_valid, errors = validate_lgbn_json(data)
        self.assertTrue(is_valid, f"Validation failed: {errors}")

    def test_valid_root_node_only(self):
        """Test validation of a single root node model."""
        data = {
            "nodes": ["X"],
            "arcs": [],
            "cpds": {
                "X": {
                    "coefficients": {"(Intercept)": [0.0]},
                    "variance": [1.0],
                    "parents": [],
                }
            },
        }
        is_valid, errors = validate_lgbn_json(data)
        self.assertTrue(is_valid, f"Validation failed: {errors}")

    def test_valid_multiple_parents(self):
        """Test validation of a node with multiple parents."""
        data = {
            "nodes": ["A", "B", "C"],
            "arcs": [["A", "C"], ["B", "C"]],
            "cpds": {
                "A": {
                    "coefficients": {"(Intercept)": [1.0]},
                    "variance": [1.0],
                    "parents": [],
                },
                "B": {
                    "coefficients": {"(Intercept)": [2.0]},
                    "variance": [1.0],
                    "parents": [],
                },
                "C": {
                    "coefficients": {"(Intercept)": [0.0], "A": [0.5], "B": [0.3]},
                    "variance": [1.0],
                    "parents": ["A", "B"],
                },
            },
        }
        is_valid, errors = validate_lgbn_json(data)
        self.assertTrue(is_valid, f"Validation failed: {errors}")

    def test_missing_nodes(self):
        """Test that missing nodes field is detected."""
        data = {
            "arcs": [],
            "cpds": {
                "X": {
                    "coefficients": {"(Intercept)": [0.0]},
                    "variance": [1.0],
                    "parents": [],
                }
            },
        }
        is_valid, errors = validate_lgbn_json(data)
        self.assertFalse(is_valid)
        self.assertTrue(any("nodes" in e for e in errors))

    def test_missing_arcs(self):
        """Test that missing arcs field is detected."""
        data = {
            "nodes": ["X"],
            "cpds": {
                "X": {
                    "coefficients": {"(Intercept)": [0.0]},
                    "variance": [1.0],
                    "parents": [],
                }
            },
        }
        is_valid, errors = validate_lgbn_json(data)
        self.assertFalse(is_valid)
        self.assertTrue(any("arcs" in e for e in errors))

    def test_missing_cpds(self):
        """Test that missing cpds field is detected."""
        data = {"nodes": ["X"], "arcs": []}
        is_valid, errors = validate_lgbn_json(data)
        self.assertFalse(is_valid)
        self.assertTrue(any("cpds" in e for e in errors))

    def test_empty_nodes(self):
        """Test that empty nodes list is rejected."""
        data = {
            "nodes": [],
            "arcs": [],
            "cpds": {},
        }
        is_valid, errors = validate_lgbn_json(data)
        self.assertFalse(is_valid)

    def test_invalid_arc_format(self):
        """Test that invalid arc format is detected."""
        data = {
            "nodes": ["A", "B"],
            "arcs": [["A"]],
            "cpds": {
                "A": {
                    "coefficients": {"(Intercept)": [0.0]},
                    "variance": [1.0],
                    "parents": [],
                },
                "B": {
                    "coefficients": {"(Intercept)": [0.0]},
                    "variance": [1.0],
                    "parents": [],
                },
            },
        }
        is_valid, errors = validate_lgbn_json(data)
        self.assertFalse(is_valid)

    def test_arc_with_unknown_node(self):
        """Test that arcs referencing unknown nodes are detected."""
        data = {
            "nodes": ["A", "B"],
            "arcs": [["A", "C"]],
            "cpds": {
                "A": {
                    "coefficients": {"(Intercept)": [0.0]},
                    "variance": [1.0],
                    "parents": [],
                },
                "B": {
                    "coefficients": {"(Intercept)": [0.0]},
                    "variance": [1.0],
                    "parents": [],
                },
            },
        }
        is_valid, errors = validate_lgbn_json(data)
        self.assertFalse(is_valid)
        self.assertTrue(any("C" in e for e in errors))

    def test_missing_intercept(self):
        """Test that missing intercept is detected."""
        data = {
            "nodes": ["X"],
            "arcs": [],
            "cpds": {
                "X": {
                    "coefficients": {},
                    "variance": [1.0],
                    "parents": [],
                }
            },
        }
        is_valid, errors = validate_lgbn_json(data)
        self.assertFalse(is_valid)

    def test_negative_variance(self):
        """Test that negative variance is rejected."""
        data = {
            "nodes": ["X"],
            "arcs": [],
            "cpds": {
                "X": {
                    "coefficients": {"(Intercept)": [0.0]},
                    "variance": [-1.0],
                    "parents": [],
                }
            },
        }
        is_valid, errors = validate_lgbn_json(data)
        self.assertFalse(is_valid)

    def test_zero_variance(self):
        """Test that zero variance is rejected."""
        data = {
            "nodes": ["X"],
            "arcs": [],
            "cpds": {
                "X": {
                    "coefficients": {"(Intercept)": [0.0]},
                    "variance": [0.0],
                    "parents": [],
                }
            },
        }
        is_valid, errors = validate_lgbn_json(data)
        self.assertFalse(is_valid)

    def test_cpd_parents_mismatch(self):
        """Test that CPD parents must match arc structure."""
        data = {
            "nodes": ["A", "B"],
            "arcs": [["A", "B"]],
            "cpds": {
                "A": {
                    "coefficients": {"(Intercept)": [0.0]},
                    "variance": [1.0],
                    "parents": [],
                },
                "B": {
                    "coefficients": {"(Intercept)": [0.0]},
                    "variance": [1.0],
                    "parents": [],
                },
            },
        }
        is_valid, errors = validate_lgbn_json(data)
        self.assertFalse(is_valid)
        self.assertTrue(any("parents" in e.lower() for e in errors))

    def test_coefficient_parent_mismatch(self):
        """Test that coefficient keys must match declared parents."""
        data = {
            "nodes": ["A", "B"],
            "arcs": [["A", "B"]],
            "cpds": {
                "A": {
                    "coefficients": {"(Intercept)": [0.0]},
                    "variance": [1.0],
                    "parents": [],
                },
                "B": {
                    "coefficients": {"(Intercept)": [0.0], "C": [0.5]},
                    "variance": [1.0],
                    "parents": ["A"],
                },
            },
        }
        is_valid, errors = validate_lgbn_json(data)
        self.assertFalse(is_valid)

    def test_node_without_cpd(self):
        """Test that all nodes must have CPDs."""
        data = {
            "nodes": ["A", "B"],
            "arcs": [],
            "cpds": {
                "A": {
                    "coefficients": {"(Intercept)": [0.0]},
                    "variance": [1.0],
                    "parents": [],
                }
            },
        }
        is_valid, errors = validate_lgbn_json(data)
        self.assertFalse(is_valid)
        self.assertTrue(any("B" in e for e in errors))

    def test_duplicate_nodes(self):
        """Test that duplicate nodes are rejected."""
        data = {
            "nodes": ["A", "A"],
            "arcs": [],
            "cpds": {
                "A": {
                    "coefficients": {"(Intercept)": [0.0]},
                    "variance": [1.0],
                    "parents": [],
                }
            },
        }
        is_valid, errors = validate_lgbn_json(data)
        self.assertFalse(is_valid)

    def test_get_schema(self):
        """Test that schema can be loaded."""
        schema = get_lgbn_schema()
        self.assertIn("$schema", schema)
        self.assertIn("properties", schema)
        self.assertIn("nodes", schema["properties"])
        self.assertIn("arcs", schema["properties"])
        self.assertIn("cpds", schema["properties"])

    def test_validate_arth150_example(self):
        """Test validation against actual example model file."""
        filepath = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "utils",
            "example_models",
            "arth150.json",
        )
        if os.path.exists(filepath):
            is_valid, errors = validate_lgbn_file(filepath)
            self.assertTrue(is_valid, f"arth150.json validation failed: {errors}")


if __name__ == "__main__":
    unittest.main()
