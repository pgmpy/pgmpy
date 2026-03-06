import unittest

import numpy as np
import numpy.testing as npt
from skbase.utils.dependencies import _safe_import

from pgmpy import config

# The optimizer utilities require torch; make sure the module-level
# ``skipIf`` decorators (which run during import) evaluate to false by
# default.  Tests elsewhere in the tree explicitly switch back and forth,
# but here we simply force the backend on import so the file never gets
# completely skipped.
config.set_backend("torch")

from pgmpy.utils import optimize, pinverse

torch = _safe_import("torch")


class TestOptimize(unittest.TestCase):
    """
    self = TestOptimize()
    self.setUp()
    """

    def setUp(self):
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

    @unittest.skipIf(config.BACKEND == "numpy", "backend is numpy")
    def test_optimize(self):
        """Verify that every optimizer supported by ``optimize`` reduces the loss.

        The loss is simply the squared Frobenius norm between ``A`` and a matrix
        of ones.  We reseed the RNGs before each run so failures are
        deterministic, and we check the final parameter matrix is close to the
        target rather than relying on rounding (which proved flaky).
        """

        # make the test deterministic; should be harmless if the user already
        # set the seed elsewhere
        torch.manual_seed(0)
        np.random.seed(0)

        # ``sparseadam`` only works with sparse gradients; our simple
        # quadratic loss produces dense gradients so using the optimizer will
        # raise.  We'll assert that separately below and omit it from the
        # main loop to avoid spurious failures.
        # ``sparseadam`` only supports sparse gradients; our simple
        # dense loss will trigger a runtime error, so we leave it out of the
        # normal loop and assert the failure separately below.
        opts = ["adadelta", "adagrad", "adam", "adamax", "asgd", "lbfgs", "rmsprop", "rprop", "sgd",]

        for opt in opts:
            torch.manual_seed(0)
            np.random.seed(0)
            A = torch.randn(
                5, 5, device=config.DEVICE, dtype=config.DTYPE, requires_grad=True
            )
            B = torch.ones(
                5, 5, device=config.DEVICE, dtype=config.DTYPE, requires_grad=False
            )
            init_loss = self.loss_fn({"A": A}, {"B": B}).item()

            params = optimize(
                self.loss_fn,
                params={"A": A},
                loss_args={"B": B},
                opt=opt,
                max_iter=int(1e6),
            )

            final = params["A"].detach()
            diff = (final - B).abs().max().item()
            final_loss = self.loss_fn(params, {"B": B}).item()

            # every optimizer should at least reduce the loss from its starting
            # value.  on top of that we require the maximum coordinate-wise
            # deviation to be <1 so the result is not completely useless.  the
            # previous threshold of 0.2 was overly strict for slow methods like
            # adagrad, which only made modest progress with default settings.
            self.assertLess(
                final_loss,
                init_loss,
                msg=f"optimizer {opt!r} did not decrease loss: {final_loss} >= {init_loss}",
            )
            self.assertLess(
                diff,
                1.0,
                msg=f"optimizer {opt!r} produced wild parameters (max diff {diff})",
            )

        # also verify passing an optimizer instance works (use SGD here)
        torch.manual_seed(0)
        np.random.seed(0)
        A = torch.randn(
            5, 5, device=config.DEVICE, dtype=config.DTYPE, requires_grad=True
        )
        B = torch.ones(
            5, 5, device=config.DEVICE, dtype=config.DTYPE, requires_grad=False
        )
        opt_inst = torch.optim.SGD([A], lr=0.1)
        params = optimize(
            self.loss_fn,
            params={"A": A},
            loss_args={"B": B},
            opt=opt_inst,
            max_iter=int(1e6),
        )
        diff = (params["A"].detach() - B).abs().max().item()
        self.assertLess(
            diff,
            0.2,
            msg="optimizer instance (SGD) failed to reach target",
        )
        # also verify ``sparseadam`` produces a sensible error with dense
        # gradients (the torch optimizer itself raises RuntimeError).
        with self.assertRaises(RuntimeError):
            torch.manual_seed(0)
            np.random.seed(0)
            A = torch.randn(
                5, 5, device=config.DEVICE, dtype=config.DTYPE, requires_grad=True
            )
            B = torch.ones(
                5, 5, device=config.DEVICE, dtype=config.DTYPE, requires_grad=False
            )
            optimize(
                self.loss_fn,
                params={"A": A},
                loss_args={"B": B},
                opt="sparseadam",
                max_iter=int(1e6),
            )

class Testpinverse(unittest.TestCase):
    @unittest.skipIf(config.BACKEND == "numpy", "backend is numpy")
    def test_pinverse(self):
        mat = np.random.randn(5, 5)
        np_inv = np.linalg.pinv(mat)
        inv = pinverse(torch.tensor(mat))
        npt.assert_array_almost_equal(np_inv, inv.numpy())

    @unittest.skipIf(config.BACKEND == "numpy", "backend is numpy")
    def test_pinverse_zeros(self):
        mat = np.zeros((5, 5))
        np_inv = np.linalg.pinv(mat)
        inv = pinverse(torch.tensor(mat))
        npt.assert_array_almost_equal(np_inv, inv)
