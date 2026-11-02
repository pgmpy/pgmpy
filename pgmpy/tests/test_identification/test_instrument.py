import pytest

from pgmpy.base import DAG
from pgmpy.identification import InstrumentalVariables


@pytest.fixture
def canonical_iv_model():
    # Canonical IV: Z -> X -> Y with latent confounder U -> X and U -> Y
    return DAG(
        [("Z", "X"), ("X", "Y"), ("U", "X"), ("U", "Y")],
        roles={"exposures": "X", "outcomes": "Y"},
        latents={"U"},
    )


@pytest.fixture
def multiple_iv_model():
    # Multiple IVs: Z1 -> X -> Y, Z2 -> X -> Y with latent confounder U
    return DAG(
        [("Z1", "X"), ("Z2", "X"), ("X", "Y"), ("U", "X"), ("U", "Y")],
        roles={"exposures": "X", "outcomes": "Y"},
        latents={"U"},
    )


@pytest.fixture
def conditional_iv_model():
    # Conditional IV: I -> X -> Y, W -> I, W -> Y, U -> X, U -> Y
    return DAG(
        [("I", "X"), ("X", "Y"), ("W", "I"), ("W", "Y"), ("U", "X"), ("U", "Y")],
        roles={"exposures": "X", "outcomes": "Y"},
        latents={"U"},
    )


@pytest.fixture
def total_conditional_iv_invalid_model():
    # Topological: I -> X -> M -> Y, with I -> M and X -> Y.
    # In G \ {X -> Y}, M d-separates I from Y, so (I, {M}) is a valid conditional IV.
    # But M lies on causal path X -> M -> Y, so it is NOT a valid total conditional IV.
    return DAG(
        [("I", "X"), ("X", "Y"), ("X", "M"), ("M", "Y"), ("I", "M")],
        roles={"exposures": "X", "outcomes": "Y"},
    )


@pytest.fixture
def no_iv_model():
    # No IV available: only X -> Y with latent confounder U
    return DAG(
        [("X", "Y"), ("U", "X"), ("U", "Y")],
        roles={"exposures": "X", "outcomes": "Y"},
        latents={"U"},
    )


@pytest.fixture
def invalid_exclusion_model():
    # Z has a direct arrow to Y, violating the exclusion restriction
    return DAG(
        [("Z", "X"), ("X", "Y"), ("Z", "Y"), ("U", "X"), ("U", "Y")],
        roles={"exposures": "X", "outcomes": "Y"},
        latents={"U"},
    )


def test_unconditional_iv_single(canonical_iv_model):
    identified_dag, is_identified = InstrumentalVariables(variant="unconditional").identify(canonical_iv_model)

    assert is_identified is True
    assert identified_dag.get_role("exposures") == ["X"]
    assert identified_dag.get_role("outcomes") == ["Y"]
    assert identified_dag.get_role("instruments") == ["Z"]
    assert identified_dag.latents == {"U"}
    assert InstrumentalVariables(variant="unconditional").validate(identified_dag) is True


def test_unconditional_iv_multiple(multiple_iv_model):
    identified_dag, is_identified = InstrumentalVariables(variant="unconditional").identify(multiple_iv_model)

    assert is_identified is True
    assert identified_dag.get_role("exposures") == ["X"]
    assert identified_dag.get_role("outcomes") == ["Y"]
    assert set(identified_dag.get_role("instruments")) == {"Z1", "Z2"}
    assert InstrumentalVariables(variant="unconditional").validate(identified_dag) is True


def test_conditional_iv(conditional_iv_model):
    identified_dag, is_identified = InstrumentalVariables(variant="conditional").identify(conditional_iv_model)

    assert is_identified is True
    # Can return a single DAG or list of DAGs depending on number of valid pairs
    if isinstance(identified_dag, list):
        assert len(identified_dag) >= 1
        first_dag = identified_dag[0]
    else:
        first_dag = identified_dag

    assert first_dag.get_role("exposures") == ["X"]
    assert first_dag.get_role("outcomes") == ["Y"]
    assert first_dag.get_role("instruments") == ["I"]
    assert first_dag.get_role("conditioning") == ["W"]
    assert InstrumentalVariables(variant="conditional").validate(first_dag) is True


def test_total_conditional_iv_excludes_path_nodes(total_conditional_iv_invalid_model):
    # Under conditional IV, (I, {W}) might separate I and Y, but W is on causal path X -> W -> Y
    # Under total-conditional, it must not identify
    _, is_identified = InstrumentalVariables(variant="total-conditional").identify(total_conditional_iv_invalid_model)
    assert is_identified is False


def test_no_iv(no_iv_model):
    dag, is_identified = InstrumentalVariables(variant="unconditional").identify(no_iv_model)
    assert is_identified is False
    assert dag.get_role("instruments") == []


def test_exclusion_restriction_violation(invalid_exclusion_model):
    dag, is_identified = InstrumentalVariables(variant="unconditional").identify(invalid_exclusion_model)
    assert is_identified is False
    assert dag.get_role("instruments") == []


def test_validate_invalid_graph(canonical_iv_model):
    # Calling validate directly on a graph without instruments or with bad instruments returns False
    assert InstrumentalVariables(variant="unconditional").validate(canonical_iv_model) is False

    # Graph with invalid instrument (Z directly connects to Y)
    bad_iv_dag = canonical_iv_model.with_role("instruments", ["Z"], inplace=False)
    bad_iv_dag.add_edge("Z", "Y")
    assert InstrumentalVariables(variant="unconditional").validate(bad_iv_dag) is False


def test_input_validation_multiple_exposures_outcomes():
    dag_multi_exp = DAG(
        [("X1", "Y"), ("X2", "Y"), ("Z", "X1")],
        roles={"exposures": {"X1", "X2"}, "outcomes": "Y"},
    )
    with pytest.raises(ValueError, match="only one exposure"):
        InstrumentalVariables().identify(dag_multi_exp)

    dag_multi_out = DAG(
        [("X", "Y1"), ("X", "Y2"), ("Z", "X")],
        roles={"exposures": "X", "outcomes": {"Y1", "Y2"}},
    )
    with pytest.raises(ValueError, match="only one outcome"):
        InstrumentalVariables().identify(dag_multi_out)


def test_input_validation_missing_edge():
    # No edge X -> Y in graph
    dag_no_edge = DAG([("Z", "X"), ("W", "Y")], roles={"exposures": "X", "outcomes": "Y"})
    with pytest.raises(ValueError, match="does not exist"):
        InstrumentalVariables().identify(dag_no_edge)


def test_invalid_variant():
    with pytest.raises(ValueError, match="Invalid variant"):
        InstrumentalVariables(variant="invalid_name")
