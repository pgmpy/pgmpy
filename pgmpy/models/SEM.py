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
