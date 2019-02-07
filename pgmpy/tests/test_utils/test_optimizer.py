import unittest

import torch
import numpy.testing as npt

from pgmpy.utils import optimize
from pgmpy.global_vars import device, dtype


class TestOptimize(unittest.TestCase):
    def setUp(self):
        self.A = torch.randn(5, 5, device=device, dtype=dtype, requires_grad=True)
        self.B = torch.ones(5, 5, device=device, dtype=dtype, requires_grad=False)

    @staticmethod
    def loss_fn(params, loss_params):
        A = params['A']
        B = loss_params['B']

        return (A - B).pow(2).sum()

    def test_optimize(self):
        # TODO: Add tests for other optimizers
        for opt in ['adadelta', 'adam', 'adamax',
                    'asgd', 'lbfgs', 'rmsprop', 'rprop']:
            A = torch.randn(5, 5, device=device, dtype=dtype, requires_grad=True)
            B = torch.ones(5, 5, device=device, dtype=dtype, requires_grad=False)
            params = optimize(self.loss_fn, params={'A': A}, loss_args={'B': B}, opt=opt, max_iter=int(1e6))

            npt.assert_almost_equal(B, params['A'].detach().numpy(), decimal=2)
