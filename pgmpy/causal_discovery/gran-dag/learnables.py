import sys

import numpy as np
import torch
import torch.nn as nn

from .base_model import BaseModel

# import torch.nn.functional as F


sys.path.insert(0, "../")


class LearnableModel(BaseModel):
    def __init__(
        self,
        num_vars,
        num_layers,
        hid_dim,
        num_params,
        nonlin="leaky-relu",
        norm_prod="path",
        square_prod=False,
    ):
        super(LearnableModel, self).__init__(
            num_vars,
            num_layers,
            hid_dim,
            num_params,
            nonlin=nonlin,
            norm_prod=norm_prod,
            square_prod=square_prod,
        )
        self.reset_params()

    def compute_log_likelihood(self, x, weights, biases, extra_params, detach=False):
        """
        Return log-likelihood of the model for each example.
        WARNING: This is really a joint distribution only if the DAGness constraint on the mask is satisfied.
                 Otherwise the joint does not integrate to one.
        :param x: (batch_size, num_vars)
        :param weights: list of tensor that are coherent with self.weights
        :param biases: list of tensor that are coherent with self.biases
        :return: (batch_size, num_vars) log-likelihoods
        """
        density_params = self.forward_given_params(x, weights, biases)

        if len(extra_params) != 0:
            extra_params = self.transform_extra_params(self.extra_params)
        log_probs = []
        for i in range(self.num_vars):
            density_param = list(torch.unbind(density_params[i], 1))
            if len(extra_params) != 0:
                density_param.extend(list(torch.unbind(extra_params[i], 0)))
            conditional = self.get_distribution(density_param)
            x_d = x[:, i].detach() if detach else x[:, i]
            log_probs.append(conditional.log_prob(x_d).unsqueeze(1))

        return torch.cat(log_probs, 1)

    def get_distribution(self, dp):
        raise NotImplementedError

    def transform_extra_params(self, extra_params):
        raise NotImplementedError


class LearnableModel_NonLinGauss(LearnableModel):
    def __init__(
        self,
        num_vars,
        num_layers,
        hid_dim,
        nonlin="leaky-relu",
        norm_prod="path",
        square_prod=False,
    ):
        super(LearnableModel_NonLinGauss, self).__init__(
            num_vars,
            num_layers,
            hid_dim,
            2,
            nonlin=nonlin,
            norm_prod=norm_prod,
            square_prod=square_prod,
        )

    def get_distribution(self, dp):
        return torch.distributions.normal.Normal(dp[0], torch.exp(dp[1]))


class LearnableModel_NonLinGaussANM(LearnableModel):
    def __init__(
        self,
        num_vars,
        num_layers,
        hid_dim,
        nonlin="leaky-relu",
        norm_prod="path",
        square_prod=False,
    ):
        super(LearnableModel_NonLinGaussANM, self).__init__(
            num_vars,
            num_layers,
            hid_dim,
            1,
            nonlin=nonlin,
            norm_prod=norm_prod,
            square_prod=square_prod,
        )
        # extra parameters are log_std
        extra_params = np.ones((self.num_vars,))
        np.random.shuffle(
            extra_params
        )  # TODO: make sure this init does not bias toward gt model
        # each element in the list represents a variable, the size of the element is the number of extra_params per var
        self.extra_params = nn.ParameterList()
        for extra_param in extra_params:
            self.extra_params.append(
                nn.Parameter(
                    torch.tensor(np.log(extra_param).reshape(1)).type(torch.Tensor)
                )
            )

    def get_distribution(self, dp):
        return torch.distributions.normal.Normal(dp[0], dp[1])

    def transform_extra_params(self, extra_params):
        transformed_extra_params = []
        for extra_param in extra_params:
            transformed_extra_params.append(torch.exp(extra_param))
        return transformed_extra_params  # returns std_dev
