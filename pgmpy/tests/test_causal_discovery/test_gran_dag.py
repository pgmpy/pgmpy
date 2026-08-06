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
    @pytest.fixture
    def numeric_df(self):
        import pandas as pd

        return pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [2.0, 4.0, 6.0, 8.0, 10.0]})

    @requires_torch
    def test_invalid_optimizer_string_raises(self, numeric_df):
        from pgmpy.causal_discovery.gran_dag import GraNDAG

        with pytest.raises(ValueError, match="Supported optimizers are"):
            GraNDAG(optimizer="invalid_optimizer", max_epochs=1).fit(numeric_df)

    @requires_torch
    @pytest.mark.parametrize(
        ("optimizer", "bad_kwargs", "match"),
        [
            ("adam", {"momentum": 0.9}, "Unknown optimizer_params"),
            ("sgd", {"betas": (0.9, 0.999)}, "Unknown optimizer_params"),
            ("adamw", {"nesterov": True}, "Unknown optimizer_params"),
            ("rmsprop", {"nesterov": True}, "Unknown optimizer_params"),
            ("adam", {"params": [1, 2, 3]}, "params"),
        ],
    )
    def test_invalid_kwargs_raises(self, numeric_df, optimizer, bad_kwargs, match):
        from pgmpy.causal_discovery.gran_dag import GraNDAG

        with pytest.raises(ValueError, match=match):
            GraNDAG(optimizer=optimizer, optimizer_params=bad_kwargs, max_epochs=1).fit(numeric_df)
