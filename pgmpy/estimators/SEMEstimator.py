import numpy as np
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

    def get_ols_fn(self, S, sigma):
        return (sigma.logdet().clamp(min=1e-4) + (S @ sigma.inverse()).trace() - S.logdet() -
                (len(self.model.y)+ len(self.model.x)))

    def get_uls_fn(self):
        pass

    def get_gls_fn(self):
        pass

    def fit(self, data, method, max_iter=1000):
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
        if not isinstance(data, (pd.DataFrame, Data)):
            raise ValueError("data must be a pandas DataFrame. Got type: {t}".format(t=type(data)))

        (B_mask, gamma_mask, wedge_y_mask, wedge_x_mask, phi_mask, theta_e_mask,
         theta_del_mask, psi_mask) = self.model.get_masks()

        # Initialize varibles for optimization
        # TODO: Move next line into a separate file to get machine parameters.
        device, dtype = (torch.device("cpu"), torch.float64)

        B = torch.rand(*B_mask.shape, device=device, dtype=dtype, requires_grad=True)
        B_mask = torch.tensor(B_mask, device=device, dtype=dtype, requires_grad=False)
        B_masked = torch.mul(B, B_mask)
        B_eye = torch.eye(B.shape[0], device=device, dtype=dtype, requires_grad=False)
        B_inv = (B_eye - B).inverse()

        gamma = torch.rand(*gamma_mask.shape, device=device, dtype=dtype, requires_grad=True)
        gamma_mask = torch.tensor(gamma_mask, device=device, dtype=dtype, requires_grad=False)
        gamma_masked = torch.mul(gamma, gamma_mask)

        wedge_y = np.random.rand(*wedge_y_mask.shape)
        wedge_y[0, 0] = 1
        wedge_y[4, 0] = 1
        wedge_y = torch.tensor(wedge_y, device=device, dtype=dtype, requires_grad=True)
        wedge_y_mask = torch.tensor(wedge_y_mask, device=device, dtype=dtype, requires_grad=False)
        wedge_y_masked = torch.mul(wedge_y, wedge_y_mask)

        wedge_x = np.random.rand(*wedge_x_mask.shape)
        wedge_x[0, 0] = 1
        wedge_x = torch.tensor(wedge_x, device=device, dtype=dtype, requires_grad=True)
        wedge_x_mask = torch.tensor(wedge_x_mask, device=device, dtype=dtype, requires_grad=False)
        wedge_x_masked = torch.mul(wedge_x, wedge_x_mask)

        phi = torch.rand(*phi_mask.shape, device=device, dtype=dtype, requires_grad=True)
        phi_mask = torch.tensor(phi_mask, device=device, dtype=dtype, requires_grad=False)
        phi_masked = torch.mul(phi, phi_mask)

        theta_e = torch.rand(*theta_e_mask.shape, device=device, dtype=dtype, requires_grad=True)
        theta_e_mask = torch.tensor(theta_e_mask, device=device, dtype=dtype, requires_grad=False)
        theta_e_masked = torch.mul(theta_e, theta_e_mask)

        theta_del = torch.rand(*theta_del_mask.shape, device=device, dtype=dtype, requires_grad=True)
        theta_del_mask = torch.tensor(theta_del_mask, device=device, dtype=dtype, requires_grad=False)
        theta_del_masked = torch.mul(theta_del, theta_del_mask)

        psi = torch.rand(*psi_mask.shape, device=device, dtype=dtype, requires_grad=True)
        psi_mask = torch.tensor(psi_mask, device=device, dtype=dtype, requires_grad=False)
        psi_masked = torch.mul(psi, psi_mask)

        # Compute model implied covariance matrix
        sigma_yy = wedge_y_masked @ B_inv @ (gamma_masked @ phi_masked @ gamma_masked.t() + psi_masked) @ B_inv.t() @ wedge_y_masked.t() + theta_e_masked
        sigma_yx = wedge_y_masked @ B_inv @ gamma_masked @ phi_masked @ wedge_x_masked.t()
        sigma_xy = sigma_yx.t()
        sigma_xx = wedge_x_masked @ phi_masked @ wedge_x_masked.t() + theta_del_masked

        # Concatenate all the sigma's in a single covariance matrix.
        #y_len, x_len = (len(self.model.y), len(self.model.x))
        #sigma = torch.zeros(y_len + x_len, y_len + x_len, device=device, dtype=dtype, requires_grad=False)
        #sigma[:y_len, :y_len] = sigma_yy
        #sigma[:y_len, y_len:] = sigma_yx
        #sigma[y_len:, :y_len] = sigma_xy
        #sigma[y_len:, y_len:] = sigma_xx
        sigma = torch.cat((torch.cat((sigma_yy, sigma_yx), 1), torch.cat((sigma_xy, sigma_xx), 1)), 0)

        masks = dict(zip(['B_mask', 'gamma_mask', 'wedge_y_mask', 'wedge_x_mask', 'phi_mask',
                          'theta_e_mask', 'theta_del_mask', 'psi_mask'],
                         [B_mask, gamma_mask, wedge_y_mask, wedge_x_mask, phi_mask,
                          theta_e_mask, theta_del_mask, psi_mask]))

        variable_order = self.model.y + self.model.x
        S = data.cov().reindex(variable_order, axis=1).reindex(variable_order, axis=0)
        S = torch.tensor(S.values, device=device, dtype=dtype, requires_grad=False)

        if method == 'ols':
            minimization_fun = self.get_ols_fn

        elif method == 'uls':
            minimization_fun = self.get_uls_fn()

        elif method == 'gls':
            minimization_fun = self.get_gls_fn()

        elif method == '2sls':
            raise NotImplementedError("2-SLS is not implemented yet")

        lr = 1e-1
        optim = torch.optim.Adam([B, gamma, wedge_y, wedge_x, phi, theta_e, theta_del, psi], lr=lr)
        #optim = torch.optim.Adam([B], lr=lr)
        for t in range(max_iter):
            loss = minimization_fun(S, sigma).log()
            print(S.logdet(), sigma.logdet(), loss.item())
            loss.backward(retain_graph=False)
            optim.step()

        return B, gamma, wedge_y, wedge_x, phi, theta_e, theta_del, psi
