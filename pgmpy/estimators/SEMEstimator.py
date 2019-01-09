import pandas as pd
import torch

from pgmpy.models import SEM
from pgmpy.data import Data


class SEMEstimator(object):
    """
    Base class of SEM estimators. All the estimators inherit this class.
    """
    def __init__(self, model):
        if not isinstance(model, SEM):
            raise ValueError("model should be an instance of SEM class. Got type: {t}".format(t=type(model)))

        self.model = model

    def get_ols_fn(self, cov_mat, sigma_yy, sigma_yx, sigma_xx):
        pass

    def get_uls_fn(self):
        pass

    def get_gls_fn(self):
        pass

    def fit(self, data, method):
        """
        Estimate the parameters of the model from the data.

        Parameters
        ----------
        data: pandas DataFrame or pgmpy.data.Data instance
            The data from which to estimate the parameters of the model.

        method: str ("ols"|"uls"|"gls"|"2sls")
            The fitting function to use.
            OLS: Ordinary Least Squares/Maximum Likelihood
            ULS: Unweighted Least Squares
            GLS: Generalized Least Squares
            2sls: 2-SLS estimator

        Returns
        -------
            pgmpy.model.SEM instance: Instance of the model with estimated parameters
        """
        if not isinstance(data, [pd.DataFrame, Data]):
            raise ValueError("data must be a pandas DataFrame. Got type: {t}".format(t=type(data)))

        (B_mask, gamma_mask, wedge_y_mask, wedge_x_mask, phi_mask, theta_e_mask,
                             theta_del_mask, psi_mask) = self.model.get_masks()

        # Initialize varibles for optimization
        # TODO: Move next line into a separate file to get machine parameters.
        device, dtype = (torch.device("cpu"), torch.float)

        B = torch.randn(*B_mask.shape, device=device, dtype=dtype, requires_grad=True)
        B_mask = torch.tensor(B_mask, device=device, dtype=dtype, requires_grad=False)
        B_masked = torch.mul(B, B_mask)
        B_eye = torch.eye(B.shape[0], device=device, dtype=dtype, requires_grad=False)
        B_inv = (B_eye - B).inverse()

        gamma = torch.randn(*gamma_mask.shape, device=device, dtype=dtype, requires_grad=True)
        gamma_mask = torch.tensor(gamma_mask, device=device, dtype=dtype, requires_grad=False)
        gamma = torch.mul(gamma, gamma_mask)

        wedge_y = torch.randn(*wedge_y_mask.shape, device=device, dtype=dtype, requires_grad=True)
        wedge_y_mask = torch.tensor(wedge_y_mask, device=device, dtype=dtype, requires_grad=False)
        wedge_y = torch.mul(wedge_y, wedge_y_mask)

        wedge_x = torch.randn(*wedge_x_mask.shape, device=device, dtype=dtype, requires_grad=True)
        wedge_x_mask = torch.tensor(wedge_x_mask, device=device, dtype=dtype, requires_grad=False)
        wedge_x = torch.mul(wedge_x, wedge_x_mask)

        phi = torch.randn(*phi_mask.shape, device=device, dtype=dtype, requires_grad=True)
        phi_mask = torch.tensor(phi_mask, device=device, dtype=dtype, requires_grad=False)
        phi = torch.mul(phi, phi_mask)

        theta_e = torch.randn(*theta_e_mask.shape, device=device, dtype=dtype, requires_grad=True)
        theta_e_mask = torch.tensor(theta_e_mask, device=device, dtype=dtype, requires_grad=False)
        theta_e = torch.mul(theta_e, theta_e_mask)

        theta_del = torch.randn(*theta_del_mask.shape, device=device, dtype=dtype, requires_grad=True)
        theta_del_mask = torch.tensor(theta_del_mask, device=device, dtype=dtype, requires_grad=False)
        theta_del = torch.mul(theta_del, theta_del_mask)

        psi = torch.randn(*psi_mask.shape, device=device, dtype=dtype, requires_grad=True)
        psi_mask = torch.tensor(psi_mask, device=device, dtype=dtype, requires_grad=False)
        psi = torch.mul(psi, psi_mask)

        # Compute model implied covariance matrix
        sigma_yy = wedge_y @ B_inv @ (gamma @ phi @ gamma.t() + psi) @ B_inv.t() @ wedge_y.t() + theta_e
        sigma_yx = wedge_y @ B_inv @ gamma @ phi @ wedge_x.t()
        sigma_xy = sigma_yx.t()
        sigma_xx = wedge_x @ phi @ wedge_x.t() + theta_del

        if method == 'ols':
            minimization_fun = self.get_ols_fn(cov_mat, sigma_yy, sigma_yx, sigma_xx)

        elif method == 'uls':
            minimization_fun = self.get_uls_fn()

        elif method == 'gls':
            minimization_fun = self.get_gls_fn()

        elif method == '2sls':
            raise NotImplementedError("2-SLS is not implemented yet")
