"""Tests for LGBN JSON schema validation against example models."""

import json
import os
import unittest

import pytest

try:
    from jsonschema import Draft202012Validator

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "lgbn_schema.json")
EXAMPLE_MODELS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "pgmpy", "utils", "example_models"
)


def validate_lgbn(data: dict) -> tuple[bool, list[str]]:
    """Validate data against LGBN schema with semantic checks."""
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    validator = Draft202012Validator(schema)
    errors = [
        f"{'.'.join(str(p) for p in e.absolute_path) or 'root'}: {e.message}"
        for e in validator.iter_errors(data)
    ]
    if errors:
        return False, errors

    # Semantic checks
    nodes = set(data.get("nodes", []))
    arcs = data.get("arcs", [])
    cpds = data.get("cpds", {})

    parents_map = {n: set() for n in nodes}
    for src, tgt in arcs:
        if src not in nodes:
            errors.append(f"Arc source '{src}' not in nodes")
        if tgt not in nodes:
            errors.append(f"Arc target '{tgt}' not in nodes")
        elif tgt in parents_map:
            parents_map[tgt].add(src)

    for node, cpd in cpds.items():
        cpd_parents = set(cpd.get("parents", []))
        if cpd_parents != parents_map.get(node, set()):
            errors.append(f"CPD parents for '{node}' don't match arcs")
        coef_keys = set(cpd.get("coefficients", {}).keys()) - {"(Intercept)"}
        if coef_keys != cpd_parents:
            errors.append(f"Coefficient keys for '{node}' don't match parents")

    for node in nodes:
        if node not in cpds:
            errors.append(f"Node '{node}' missing CPD")

    return len(errors) == 0, errors


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
class TestLGBNSchemaWithExamples(unittest.TestCase):
    """Test LGBN schema against actual example model files."""

    def test_arth150(self):
        """Validate arth150.json example model."""
        filepath = os.path.join(EXAMPLE_MODELS_PATH, "arth150.json")
        with open(filepath) as f:
            data = json.load(f)
        is_valid, errors = validate_lgbn(data)
        self.assertTrue(is_valid, f"arth150.json failed: {errors}")


if __name__ == "__main__":
    unittest.main()
