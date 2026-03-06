import numpy as np
import numpy.testing as npt
import pytest
from skbase.utils.dependencies import _safe_import

from pgmpy import config
from pgmpy.utils import optimize, pinverse

torch = _safe_import("torch")


class TestOptimize:
    """
    Example usage of the test setup:

    >>> self = TestOptimize()
    >>> self.setup_method(None)
    """

    def setup_method(self, method):
        self.A = torch.randn(
            5, 5, device=config.DEVICE, dtype=config.DTYPE, requires_grad=True
        )
        self.B = torch.ones(
            5, 5, device=config.DEVICE, dtype=config.DTYPE, requires_grad=False
        )

    def loss_fn(self, params, loss_params):
        A = params["A"]
        B = loss_params["B"]

        return (A - B).pow(2).sum()

    @pytest.mark.skipif(config.BACKEND == "numpy", reason="backend is numpy")
    def test_optimize(self):
        # TODO: Add tests for other optimizers
        for opt in ["adadelta", "adam", "adamax", "asgd", "lbfgs", "rmsprop", "rprop"]:
            A = torch.randn(
                5, 5, device=config.DEVICE, dtype=config.DTYPE, requires_grad=True
            )
            B = torch.ones(
                5, 5, device=config.DEVICE, dtype=config.DTYPE, requires_grad=False
            )
            params = optimize(
                self.loss_fn,
                params={"A": A},
                loss_args={"B": B},
                opt=opt,
                max_iter=int(1e6),
            )

            npt.assert_almost_equal(
                B.data.cpu().numpy(),
                params["A"].detach().cpu().numpy().round(),
                decimal=1,
            )


class TestPinverse:
    @pytest.mark.skipif(config.BACKEND == "numpy", reason="backend is numpy")
    def test_pinverse(self):
        mat = np.random.randn(5, 5)
        np_inv = np.linalg.pinv(mat)
        inv = pinverse(torch.tensor(mat))
        npt.assert_array_almost_equal(np_inv, inv.numpy())

    @pytest.mark.skipif(config.BACKEND == "numpy", reason="backend is numpy")
    def test_pinverse_zeros(self):
        mat = np.zeros((5, 5))
        np_inv = np.linalg.pinv(mat)
        inv = pinverse(torch.tensor(mat))
        npt.assert_array_almost_equal(np_inv, inv)
