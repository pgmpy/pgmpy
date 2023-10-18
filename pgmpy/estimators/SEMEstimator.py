import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch

from pgmpy import config
from pgmpy.models import SEM, SEMAlg, SEMGraph
from pgmpy.utils import optimize, pinverse


class SEMEstimator(object):
    """
    Base class of SEM estimators. All the estimators inherit this class.
    """

    def __init__(self, model):
        if config.BACKEND == "numpy":
            raise ValueError(
                f"SEMEstimator requires torch backend. Currently it's numpy. Call pgmpy.config.set_backend('torch') to switch"
            )

        if isinstance(model, (SEMGraph, SEM)):
            self.model = model.to_lisrel()
        elif isinstance(model, SEMAlg):
            self.model = model
        else:
            raise ValueError(
                f"Model should be an instance of either SEMGraph or SEMAlg class. Got type: {type(model)}"
            )

        # Initialize trainable and fixed mask tensors
        self.B_mask = torch.tensor(
            self.model.B_mask,
            device=config.DEVICE,
            dtype=config.DTYPE,
            requires_grad=False,
        )
        self.zeta_mask = torch.tensor(
            self.model.zeta_mask,
            device=config.DEVICE,
            dtype=config.DTYPE,
            requires_grad=False,
        )

        self.B_fixed_mask = torch.tensor(
            self.model.B_fixed_mask,
            device=config.DEVICE,
            dtype=config.DTYPE,
            requires_grad=False,
        )
        self.zeta_fixed_mask = torch.tensor(
            self.model.zeta_fixed_mask,
            device=config.DEVICE,
            dtype=config.DTYPE,
            requires_grad=False,
        )

        self.wedge_y = torch.tensor(
            self.model.wedge_y,
            device=config.DEVICE,
            dtype=config.DTYPE,
            requires_grad=False,
        )
        self.B_eye = torch.eye(
            self.B_mask.shape[0],
            device=config.DEVICE,
            dtype=config.DTYPE,
            requires_grad=False,
        )

    def _get_implied_cov(self, B, zeta):
        """
        Computes the implied covariance matrix from the given parameters.
        """
        B_masked = torch.mul(B, self.B_mask) + self.B_fixed_mask
        B_inv = pinverse(self.B_eye - B_masked)
        zeta_masked = torch.mul(zeta, self.zeta_mask) + self.zeta_fixed_mask

        return self.wedge_y @ B_inv @ zeta_masked @ B_inv.t() @ self.wedge_y.t()

    def ml_loss(self, params, loss_args):
        r"""
        Method to compute the Maximum Likelihood loss function. The optimizer calls this
        method after each iteration with updated params to compute the new loss.

        The fitting function for ML is:
        .. math:: F_{ML} = \log |\Sigma(\theta)| + tr(S \Sigma^{-1}(\theta)) - \log S - (p+q)

        Parameters
        ----------
        params: dict
            params contain all the variables which are updated in each iteration of the
            optimization.

        loss_args: dict
            loss_args contain all the variable which are not updated in each iteration but
            are required to compute the loss.

        Returns
        -------
        torch.tensor: The loss value for the given params and loss_args
        """
        S = loss_args["S"]
        sigma = self._get_implied_cov(params["B"], params["zeta"])

        return (
            sigma.det().clamp(min=1e-4).log()
            + (S @ pinverse(sigma)).trace()
            - S.logdet()
            - len(self.model.y)
        )

    def uls_loss(self, params, loss_args):
        r"""
        Method to compute the Unweighted Least Squares fitting function. The optimizer calls
        this method after each iteration with updated params to compute the new loss.

        The fitting function for ML is:
        .. math:: F_{ULS} = tr[(S - \Sigma(\theta))^2]

        Parameters
        ----------
        params: dict
            params contain all the variables which are updated in each iteration of the
            optimization.

        loss_args: dict
            loss_args contain all the variable which are not updated in each iteration but
            are required to compute the loss.

        Returns
        -------
        torch.tensor: The loss value for the given params and loss_args
        """
        S = loss_args["S"]
        sigma = self._get_implied_cov(params["B"], params["zeta"])
        return (S - sigma).pow(2).trace()

    def gls_loss(self, params, loss_args):
        r"""
        Method to compute the Weighted Least Squares fitting function. The optimizer calls
        this method after each iteration with updated params to compute the new loss.

        The fitting function for ML is:
        .. math:: F_{ULS} = tr \{ [(S - \Sigma(\theta)) W^{-1}]^2 \}

        Parameters
        ----------
        params: dict
            params contain all the variables which are updated in each iteration of the
            optimization.

        loss_args: dict
            loss_args contain all the variable which are not updated in each iteration but
            are required to compute the loss.

        Returns
        -------
        torch.tensor: The loss value for the given params and loss_args
        """
        S = loss_args["S"]
        W_inv = pinverse(loss_args["W"])
        sigma = self._get_implied_cov(params["B"], params["zeta"])
        return ((S - sigma) @ W_inv).pow(2).trace()

    def get_init_values(self, data, method):
        """
        Computes the starting values for the optimizer.

        Reference
        ---------
        .. [1] Table 4C.1: Bollen, K. (2014). Structural Equations with Latent Variables.
                New York, NY: John Wiley & Sons.

        """
        # Initialize all the values even if the edge doesn't exist, masks would take care of that.
        a = 0.4
        scaling_vars = self.model.to_SEMGraph().get_scaling_indicators()
        eta, m = self.model.eta, len(self.model.eta)

        if method == "random":
            B = np.random.rand(m, m)
            zeta = np.random.rand(m, m)

        elif method == "std":
            # Add observed vars to `scaling_vars to point to itself. Trick to keep code short.
            for observed_var in self.model.y:
                scaling_vars[observed_var] = observed_var

            B = np.random.rand(m, m)
            for i in range(m):
                for j in range(m):
                    if scaling_vars[eta[i]] == eta[j]:
                        B[i, j] = 1.0
                    elif i != j:
                        B[i, j] = a * (
                            data.loc[:, scaling_vars[eta[i]]].std()
                            / data.loc[:, scaling_vars[eta[j]]].std()
                        )
            zeta = np.random.rand(m, m)
            for i in range(m):
                zeta[i, i] = a * ((data.loc[:, scaling_vars[eta[i]]].std()) ** 2)
            for i in range(m):
                for j in range(m):
                    zeta[i, j] = zeta[j, i] = a * np.sqrt(zeta[i, i] * zeta[j, j])

        elif method.lower() == "iv":
            raise NotImplementedError("IV initialization not supported yet.")

        return B, zeta

    def fit(
        self,
        data,
        method,
        opt="adam",
        init_values="random",
        exit_delta=1e-4,
        max_iter=1000,
        **kwargs,
    ):
        """
        Estimate the parameters of the model from the data.

        Parameters
        ----------
        data: pandas DataFrame or pgmpy.data.Data instance
            The data from which to estimate the parameters of the model.

        method: str ("ml"|"uls"|"gls"|"2sls")
            The fitting function to use.
            ML : Maximum Likelihood
            ULS: Unweighted Least Squares
            GLS: Generalized Least Squares
            2sls: 2-SLS estimator

        init_values: str or dict
            Options for str: random | std | iv
            dict: dictionary with keys `B` and `zeta`.

        **kwargs: dict
            Extra parameters required in case of some estimators.
            GLS:
                W: np.array (n x n) where n is the number of observe variables.
            2sls:
                x:
                y:

        Returns
        -------
            pgmpy.model.SEM instance: Instance of the model with estimated parameters

        References
        ----------
        .. [1] Bollen, K. A. (2010). Structural equations with latent variables. New York: Wiley.
        """
        # Check if given arguments are valid
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"data must be a pandas DataFrame. Got type: {type(data)}")

        if not sorted(data.columns) == sorted(self.model.y):
            raise ValueError(
                f"The column names data do not match the variables in the model. Expected: {sorted(self.model.observed)}. Got: {sorted(data.columns)}"
            )

        # Initialize the values of parameters as tensors.
        if isinstance(init_values, dict):
            B_init, zeta_init = init_values["B"], init_values["zeta"]
        else:
            B_init, zeta_init = self.get_init_values(data, method=init_values.lower())
        B = torch.tensor(
            B_init, device=config.DEVICE, dtype=config.DTYPE, requires_grad=True
        )
        zeta = torch.tensor(
            zeta_init, device=config.DEVICE, dtype=config.DTYPE, requires_grad=True
        )

        # Compute the covariance of the data
        variable_order = self.model.y
        S = data.cov().reindex(variable_order, axis=1).reindex(variable_order, axis=0)
        S = torch.tensor(
            S.values, device=config.DEVICE, dtype=config.DTYPE, requires_grad=False
        )

        # Optimize the parameters
        if method.lower() == "ml":
            params = optimize(
                self.ml_loss,
                params={"B": B, "zeta": zeta},
                loss_args={"S": S},
                opt=opt,
                exit_delta=exit_delta,
                max_iter=max_iter,
            )

        elif method.lower() == "uls":
            params = optimize(
                self.uls_loss,
                params={"B": B, "zeta": zeta},
                loss_args={"S": S},
                opt=opt,
                exit_delta=exit_delta,
                max_iter=max_iter,
            )

        elif method.lower() == "gls":
            W = torch.tensor(
                kwargs["W"],
                device=config.DEVICE,
                dtype=config.DTYPE,
                requires_grad=False,
            )
            params = optimize(
                self.gls_loss,
                params={"B": B, "zeta": zeta},
                loss_args={"S": S, "W": W},
                opt=opt,
                exit_delta=exit_delta,
                max_iter=max_iter,
            )

        elif method.lower() == "2sls" or method.lower() == "2-sls":
            raise NotImplementedError("2-SLS is not implemented yet")

        B = params["B"] * self.B_mask + self.B_fixed_mask
        zeta = params["zeta"] * self.zeta_mask + self.zeta_fixed_mask

        # Compute goodness of fit statistics.
        N = data.shape[0]
        sample_cov = S.detach().numpy()
        sigma_hat = self._get_implied_cov(B, zeta).detach().numpy()
        residual = sample_cov - sigma_hat

        norm_residual = np.zeros(residual.shape)
        for i in range(norm_residual.shape[0]):
            for j in range(norm_residual.shape[1]):
                norm_residual[i, j] = (sample_cov[i, j] - sigma_hat[i, j]) / np.sqrt(
                    ((sigma_hat[i, i] * sigma_hat[j, j]) + (sigma_hat[i, j] ** 2)) / N
                )

        # Compute chi-square value.
        likelihood_ratio = -(N - 1) * (
            np.log(np.linalg.det(sigma_hat))
            + (np.linalg.inv(sigma_hat) @ S).trace()
            - np.log(np.linalg.det(S))
            - S.shape[0]
        )
        if method.lower() == "ml":
            error = self.ml_loss(params, loss_args={"S": S})
        elif method.lower() == "uls":
            error = self.uls_loss(params, loss_args={"S": S})
        elif method.lower() == "gls":
            error = self.gls_loss(params, loss_args={"S": S, "W": W})
        chi_square = likelihood_ratio / error.detach().numpy()

        free_params = self.B_mask.sum()
        dof = ((S.shape[0] * (S.shape[0] + 1)) / 2) - free_params

        summary = {
            "Sample Size": N,
            "Sample Covariance": sample_cov,
            "Model Implied Covariance": sigma_hat,
            "Residual": residual,
            "Normalized Residual": norm_residual,
            "chi_square": chi_square,
            "dof": dof,
        }

        # Update the model with the learned params
        self.model.set_params(
            B=params["B"].detach().numpy(), zeta=params["B"].detach().numpy()
        )
        return summary


class IVEstimator:
    """
    Initialize IVEstimator object.

    Parameters
    ----------
    model: pgmpy.models.SEM
        The model for which estimation need to be done.

    Examples
    --------
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, Y, data, ivs=None, civs=None):
        """
        Estimates the parameter X -> Y.

        Parameters
        ----------
        X: str
            The covariate variable of the parameter being estimated.

        Y: str
            The predictor variable of the parameter being estimated.

        data: pd.DataFrame
            The data from which to learn the parameter.

        ivs: List (default: None)
            List of variable names which should be used as Instrumental Variables (IV).
            If not specified, tries to find the IVs from the model structure, fails if
            can't find either IV or Conditional IV.

        civs: List of tuples (tuple form: (var, coditional_var))
            List of conditional IVs to use for estimation.
            If not specified, tries to find the IVs from the model structure, fails if
            can't find either IV or Conditional IVs.

        Examples
        --------
        >>> from pgmpy.estimators import IVEstimator # TODO: Finish example.
        """
        if (ivs is None) and (civs is None):
            ivs = self.model.get_ivs(X, Y)
            civs = self.model.get_conditional_ivs(X, Y)

        civs = [civ for civ in civs if civ[0] not in ivs]

        reg_covars = []
        for var in self.model.graph.predecessors(X):
            if var in self.model.observed:
                reg_covars.append(var)

        # Get CIV conditionals
        civ_conditionals = []
        for civ in civs:
            civ_conditionals.extend(civ[1])

        # First stage regression.
        params = (
            sm.OLS(data.loc[:, X], data.loc[:, reg_covars + civ_conditionals])
            .fit()
            .params
        )

        data["X_pred"] = np.zeros(data.shape[0])
        for var in reg_covars:
            data.X_pred += params[var] * data.loc[:, var]

        summary = sm.OLS(
            data.loc[:, Y], data.loc[:, ["X_pred"] + civ_conditionals]
        ).fit()
        return summary.params["X_pred"], summary
