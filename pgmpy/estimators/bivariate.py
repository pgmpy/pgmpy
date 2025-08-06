import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from torch_hsic import hsic


def anm_bivariate(x, y):
    """
    Performs bivariate causal discovery using the Additive Noise Model (ANM).

    This function tests for a causal relationship between two variables, x and y,
    in both directions (x -> y and y -> x) based on the ANM principle. It uses
    a Gaussian Process Regressor to model the functional relationship and the
    Hilbert-Schmidt Independence Criterion (HSIC) to test for the independence
    of the residuals.

    Args:
        x (np.ndarray): A 1D numpy array representing the cause variable.
        y (np.ndarray): A 1D numpy array representing the effect variable.

    Returns:
        dict: A dictionary containing the p-values for both causal directions.
              'p_value_xy': p-value for the causal direction x -> y.
              'p_value_yx': p-value for the causal direction y -> x.
    """

    # Reshape data for sklearn
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # Define the Gaussian Process Regressor kernel
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

    # Test for X -> Y
    gp_xy = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp_xy.fit(x, y)
    y_pred = gp_xy.predict(x)
    res_xy = y - y_pred
    p_value_xy = hsic_test(
        torch.from_numpy(x).float(), torch.from_numpy(res_xy).float()
    )

    # Test for Y -> X
    gp_yx = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp_yx.fit(y, x)
    x_pred = gp_yx.predict(y)
    res_yx = x - x_pred
    p_value_yx = hsic_test(
        torch.from_numpy(y).float(), torch.from_numpy(res_yx).float()
    )

    return {"p_value_xy": p_value_xy, "p_value_yx": p_value_yx}


def hsic_test(x, y):
    """
    Performs the Hilbert-Schmidt Independence Criterion (HSIC) test.

    Args:
        x (torch.Tensor): A torch tensor of the first variable.
        y (torch.Tensor): A torch tensor of the second variable.

    Returns:
        float: The p-value from the HSIC test.
    """
    return hsic.HSIC(x, y)
