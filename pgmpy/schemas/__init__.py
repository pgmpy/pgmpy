"""JSON schema definitions and validation utilities for pgmpy model files."""

import json
import os

try:
    from jsonschema import Draft202012Validator

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


SCHEMA_DIR = os.path.dirname(__file__)


def get_lgbn_schema() -> dict:
    """Return the JSON schema for Linear Gaussian Bayesian Network files."""
    schema_path = os.path.join(SCHEMA_DIR, "lgbn_schema.json")
    with open(schema_path) as f:
        return json.load(f)


def validate_lgbn_json(data: dict) -> tuple[bool, list[str]]:
    """Validate a dictionary against the LGBN JSON schema.

    Parameters
    ----------
    data : dict
        The data to validate (parsed JSON).

    Returns
    -------
    tuple
        (is_valid, errors) where is_valid is True if validation passes,
        and errors is a list of error messages.

    Raises
    ------
    ImportError
        If jsonschema package is not installed.

    """
    if not HAS_JSONSCHEMA:
        raise ImportError(
            "jsonschema package required for validation. "
            "Install with: pip install jsonschema"
        )

    schema = get_lgbn_schema()
    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(data))

    if not errors:
        semantic_errors = _check_lgbn_semantics(data)
        if semantic_errors:
            return False, semantic_errors
        return True, []

    error_messages = []
    for error in errors:
        path = (
            ".".join(str(p) for p in error.absolute_path)
            if error.absolute_path
            else "root"
        )
        error_messages.append(f"{path}: {error.message}")

    return False, error_messages


def _check_lgbn_semantics(data: dict) -> list[str]:
    """Perform semantic validation beyond JSON schema."""
    errors = []
    nodes = set(data.get("nodes", []))
    arcs = data.get("arcs", [])
    cpds = data.get("cpds", {})

    for arc in arcs:
        if arc[0] not in nodes:
            errors.append(f"Arc source '{arc[0]}' not in nodes list")
        if arc[1] not in nodes:
            errors.append(f"Arc target '{arc[1]}' not in nodes list")

    for node in cpds:
        if node not in nodes:
            errors.append(f"CPD defined for '{node}' which is not in nodes list")

    for node in nodes:
        if node not in cpds:
            errors.append(f"Node '{node}' has no CPD defined")

    parents_from_arcs = {node: set() for node in nodes}
    for arc in arcs:
        if arc[1] in parents_from_arcs:
            parents_from_arcs[arc[1]].add(arc[0])

    for node, cpd in cpds.items():
        cpd_parents = set(cpd.get("parents", []))
        arc_parents = parents_from_arcs.get(node, set())

        if cpd_parents != arc_parents:
            errors.append(
                f"CPD parents for '{node}' {cpd_parents} don't match "
                f"arc structure {arc_parents}"
            )

        coefficients = cpd.get("coefficients", {})
        coef_keys = set(coefficients.keys()) - {"(Intercept)"}

        if coef_keys != cpd_parents:
            errors.append(
                f"Coefficient keys for '{node}' {coef_keys} don't match "
                f"declared parents {cpd_parents}"
            )

    return errors


def validate_lgbn_file(filepath: str) -> tuple[bool, list[str]]:
    """Validate a JSON file against the LGBN schema.

    Parameters
    ----------
    filepath : str
        Path to the JSON file.

    Returns
    -------
    tuple
        (is_valid, errors) where is_valid is True if validation passes.

    """
    try:
        with open(filepath) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    except FileNotFoundError:
        return False, [f"File not found: {filepath}"]

    return validate_lgbn_json(data)
