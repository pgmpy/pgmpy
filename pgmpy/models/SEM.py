#!/usr/bin/env python3

import numpy as np

from pgmpy.base import DirectedGraph


class SEM(DirectedGraph):
    """
    Base class for linear Structural Equation Models.

    All the nodes by default has an associated error term and doesn't need to be specified.
    Each edge has the linear parameter.
    """
    def __init__(self, ebunch=None, latents=[], err_corr={}):
        """
        Parameters
        ----------
        ebunch: Array of tuples of type (u, v, parameter)
        latents: list of nodes
            List of nodes which are latent. By default all others are considered observed.
        err_corr: dict of correlations

        """
        super(SEM, self).__init__()
        # TODO: Check if ebunch has len 3 for each element in ebunch.
        self.latents = set(latents)
        self.observed = set()

        if ebunch:
            for (u, v) in ebunch:
                self.add_edge(u, v, weight=np.NaN)

        for node, di in self.node.items():
            if node in self.latents:
                di['latent'] = True
            else:
                di['latent'] = False
                self.observed.add(node)

        self.err_corr = err_corr

    def get_params(self):
        """
        Get the parameters B, \Gamma, \Phi from the graph structure

        Returns
        -------
        B: np.array
        \Gamma: np.array
        \Phi: np.array
        """
        pass

    def set_params(self, B, gamma, phi):
        """
        Sets the parameter values to the graph structure.

        Parameters
        ----------
        B: np.array
        \Gamma: np.array
        \Phi: np.array

        """
        pass

    def get_ivs(self, X, Y):
        """
        Returns the Instrumental variables for the relation X -> Y

        Parameters
        ----------
        X: The observed variable name
        Y: The observed variable name

        Returns
        -------
        set: The set of Instrumental Variables for the predicted value.
        """
        pass
