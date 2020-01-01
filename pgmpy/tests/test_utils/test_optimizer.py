import unittest
import numpy as np
import numpy.testing as npt

from pgmpy.utils import optimize, pinverse
from pgmpy.global_vars import device, dtype

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None


class TestOptimize(unittest.TestCase):
    """
    self = TestOptimize()
    self.setUp()
    """

    def setUp(self):
        self.A = torch.randn(5, 5, device=device, dtype=dtype, requires_grad=True)
        self.B = torch.ones(5, 5, device=device, dtype=dtype, requires_grad=False)

    def loss_fn(self, params, loss_params):
        A = params["A"]
        B = loss_params["B"]

        return (A - B).pow(2).sum()

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_optimize(self):
        # TODO: Add tests for other optimizers
        for opt in ["adadelta", "adam", "adamax", "asgd", "lbfgs", "rmsprop", "rprop"]:
            A = torch.randn(5, 5, device=device, dtype=dtype, requires_grad=True)
            B = torch.ones(5, 5, device=device, dtype=dtype, requires_grad=False)
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


class Testpinverse(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_pinverse(self):
        mat = np.random.randn(5, 5)
        np_inv = np.linalg.pinv(mat)
        inv = pinverse(torch.tensor(mat))
        npt.assert_array_almost_equal(np_inv, inv.numpy())

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_pinverse_zeros(self):
        mat = np.zeros((5, 5))
        np_inv = np.linalg.pinv(mat)
        inv = pinverse(torch.tensor(mat))
        npt.assert_array_almost_equal(np_inv, inv)
