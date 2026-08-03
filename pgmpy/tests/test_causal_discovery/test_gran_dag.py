"""
Tests for the GraNDAG class in pgmpy.causal_discovery.
"""

import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.causal_discovery.gran_dag import _dag_constraint

requires_torch = pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="execute only if required dependency present",
)


@requires_torch
def test_dag_constraint_behavior():
    import torch

    # Zero matrix (DAG) should return a scalar tensor with value ~0.0
    U_zero = torch.zeros(4, 4)
    res_zero = _dag_constraint(U_zero)
    assert res_zero.shape == torch.Size([])
    assert res_zero.item() == pytest.approx(0.0, abs=1e-5)

    # Cyclic graph (node 0 -> node 1 -> node 0) should be positive
    U_cyclic = torch.zeros(4, 4)
    U_cyclic[0, 1] = 1.0
    U_cyclic[1, 0] = 1.0
    assert _dag_constraint(U_cyclic).item() > 0.0


class TestOptimizerValidation:
    @requires_torch
    def test_invalid_optimizer_string_raises(self):
        from pgmpy.causal_discovery.gran_dag import _validate_optimizer

        with pytest.raises(ValueError, match="Supported optimizers are"):
            _validate_optimizer("invalid_optimizer", {})

    @requires_torch
    def test_invalid_optimizer_type_raises(self):
        from pgmpy.causal_discovery.gran_dag import _validate_optimizer

        with pytest.raises(ValueError, match="optimizer must be a string"):
            _validate_optimizer(123, {})

    @requires_torch
    def test_invalid_params_raises(self):
        from pgmpy.causal_discovery.gran_dag import _validate_optimizer

        with pytest.raises(ValueError, match="'params' cannot be passed"):
            _validate_optimizer("adam", {"params": [1, 2, 3]})
