import numpy as np
import numpy.testing as npt
import pytest
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

from pgmpy import config
from pgmpy.utils import optimize, pinverse

torch = _safe_import("torch")

requires_torch = pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"), reason="torch is not installed"
)


@pytest.fixture(autouse=True)
def torch_backend():
    previous_backend = config.get_backend()
    previous_device = config.get_device()
    previous_dtype = config.get_dtype()
    config.set_backend("torch")
    yield
    config.set_backend(
        previous_backend,
        device=None if previous_device is None else str(previous_device),
        dtype=previous_dtype,
    )


def loss_fn(params, loss_params):
    A = params["A"]
    B = loss_params["B"]

    return (A - B).pow(2).sum()


@pytest.fixture
def A():
    return torch.randn(5, 5, device=config.DEVICE, dtype=config.DTYPE, requires_grad=True)


@pytest.fixture
def B():
    return torch.ones(5, 5, device=config.DEVICE, dtype=config.DTYPE, requires_grad=False)


@requires_torch
def test_optimize(A, B):
    # TODO: Add tests for other optimizers
    for opt in ["adadelta", "adam", "adamax", "asgd", "lbfgs", "rmsprop", "rprop"]:
        params = optimize(
            loss_fn,
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


@requires_torch
def test_pinverse():
    mat = np.random.randn(5, 5)
    np_inv = np.linalg.pinv(mat)
    inv = pinverse(torch.tensor(mat))
    npt.assert_array_almost_equal(np_inv, inv.numpy())


@requires_torch
def test_pinverse_zeros():
    mat = np.zeros((5, 5))
    np_inv = np.linalg.pinv(mat)
    inv = pinverse(torch.tensor(mat))
    npt.assert_array_almost_equal(np_inv, inv)
