import numpy as np
import pandas as pd
import torch

from pgmpy.models import SEM
from pgmpy.data import Data
from pgmpy.global_vars import device, dtype
from pgmpy.utils import optimize


class SEMEstimator(object):
    """
    Base class of SEM estimators. All the estimators inherit this class.
    """
    def __init__(self, model):
        if not isinstance(model, SEM):
            raise ValueError("model should be an instance of SEM class. Got type: {t}".format(t=type(model)))

        self.model = model

        # Initialize mask tensors
        masks = self.model.get_masks()
        fixed_masks = self.model.get_fixed_masks()
        self.masks = {}
        self.fixed_masks = {}
        model_params = ['B', 'gamma', 'wedge_y', 'wedge_x', 'phi', 'theta_e', 'theta_del', 'psi']
        for i, model_param in enumerate(model_params):
            self.masks[model_param] = torch.tensor(masks[i], device=device, dtype=dtype, requires_grad=False)
            self.fixed_masks[model_param] = torch.tensor(fixed_masks[i], device=device, dtype=dtype,
                                                         requires_grad=False)
        self.B_eye = torch.eye(self.masks['B'].shape[0], device=device, dtype=dtype, requires_grad=False)

    def _get_implied_cov(self, B, gamma, wedge_y, wedge_x, phi, theta_e, theta_del, psi):
        """
        Computes the implied covariance matrix from the given parameters.
        """
        B_masked = torch.mul(B, self.masks['B']) + self.fixed_masks['B']
        B_inv = (self.B_eye - B_masked).inverse()
        gamma_masked = torch.mul(gamma, self.masks['gamma']) + self.fixed_masks['gamma']
        wedge_y_masked = torch.mul(wedge_y, self.masks['wedge_y']) + self.fixed_masks['wedge_y']
        wedge_x_masked = torch.mul(wedge_x, self.masks['wedge_x']) + self.fixed_masks['wedge_x']
        phi_masked = torch.mul(phi, self.masks['phi']) + self.fixed_masks['phi']
        theta_e_masked = torch.mul(theta_e, self.masks['theta_e']) + self.fixed_masks['theta_e']
        theta_del_masked = torch.mul(theta_del, self.masks['theta_del']) + self.fixed_masks['theta_del']
        psi_masked = torch.mul(psi, self.masks['psi']) + self.fixed_masks['psi']

        sigma_yy = wedge_y_masked @ B_inv @ (gamma_masked @ phi_masked @ gamma_masked.t() + psi_masked) @ \
                   B_inv.t() @ wedge_y_masked.t() + theta_e_masked
        sigma_yx = wedge_y_masked @ B_inv @ gamma_masked @ phi_masked @ wedge_x_masked.t()
        sigma_xy = sigma_yx.t()
        sigma_xx = wedge_x_masked @ phi_masked @ wedge_x_masked.t() + theta_del_masked

        sigma = torch.cat((torch.cat((sigma_yy, sigma_yx), 1), torch.cat((sigma_xy, sigma_xx), 1)), 0)
        return sigma

    def ols_loss(self, params, loss_args):
        S = loss_args['S']
        sigma = self._get_implied_cov(params['B'], params['gamma'], params['wedge_y'],
                                      params['wedge_x'], params['phi'], params['theta_e'],
                                      params['theta_del'], params['psi'])
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

        B = torch.rand(*self.masks['B'].shape, device=device, dtype=dtype, requires_grad=True)
        gamma = torch.rand(*self.masks['gamma'].shape, device=device, dtype=dtype, requires_grad=True)
        wedge_y = torch.rand(*self.masks['wedge_y'].shape, device=device, dtype=dtype, requires_grad=True)
        wedge_x = torch.rand(*self.masks['wedge_x'].shape, device=device, dtype=dtype, requires_grad=True)
        phi = torch.rand(*self.masks['phi'].shape, device=device, dtype=dtype, requires_grad=True)
        theta_e = torch.rand(*self.masks['theta_e'].shape, device=device, dtype=dtype, requires_grad=True)
        theta_del = torch.rand(*self.masks['theta_del'].shape, device=device, dtype=dtype, requires_grad=True)
        psi = torch.rand(*self.masks['psi'].shape, device=device, dtype=dtype, requires_grad=True)

        variable_order = self.model.y + self.model.x
        S = data.cov().reindex(variable_order, axis=1).reindex(variable_order, axis=0)
        S = torch.tensor(S.values, device=device, dtype=dtype, requires_grad=False)

        if method == 'ols':
            return optimize(self.ols_loss, params={'B': B, 'gamma': gamma, 'wedge_y': wedge_y,
                                                   'wedge_x': wedge_x, 'phi': phi, 'theta_e':
                                                   theta_e, 'theta_del': theta_del, 'psi': psi},
                            loss_args={'S': S}, opt='lbfgs')

        elif method == 'uls':
            minimization_fun = self.get_uls_fn()

        elif method == 'gls':
            minimization_fun = self.get_gls_fn()

        elif method == '2sls':
            raise NotImplementedError("2-SLS is not implemented yet")

        # lr = 1e-1
        # optim = torch.optim.Adam([B, gamma, wedge_y, wedge_x, phi, theta_e, theta_del, psi], lr=lr)
        # #optim = torch.optim.Adam([B], lr=lr)
        # for t in range(max_iter):
        #     loss = minimization_fun(S, sigma).log()
        #     print(S.logdet(), sigma.logdet(), loss.item())
        #     loss.backward(retain_graph=False)
        #     optim.step()

        return B, gamma, wedge_y, wedge_x, phi, theta_e, theta_del, psi
