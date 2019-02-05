#!/usr/bin/env python3

import networkx as nx
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
        ebunch: list / array-like
            Each tuple can be of two possible shape:
                1. (u, v): This would add an edge from u to v without setting any parameter
                           for the edge.
                2. (u, v, parameter): This would add an edge from u to v and set the edge
                            parameter to `parameter`.

        latents: list / array-like
            List of nodes which are latent. By default all others are considered observed.

        err_corr: dict
            Dict of correlation between the error terms in the model.

        Examples
        --------
        """
        super(SEM, self).__init__()

        # Check and make ebunch uniform (len 3 tuples)
        if ebunch:
            u_ebunch = []
            for t in ebunch:
                if len(t) == 3:
                    u_ebunch.append(t)
                elif len(t) == 2:
                    u_ebunch.append((t[0], t[1], np.NaN))
                else:
                    raise ValueError("Expected tuple length: 2 or 3. Got {t} of len {shape}".format(
                                                            t=t, shape=len(t)))

        # Initialize attributes latents and observed
        self.latents = set(latents)
        self.observed = set()

        # Create the full graph structure. Adds new latent variable whenever observed --> latent
        # edge to convert to standard LISREL notation.
        self.graph = nx.DiGraph()
        if u_ebunch:
            mapping = {}
            for u, v, w in u_ebunch:
                self.graph.add_edge(u, v, weight=w)
                if (u not in self.latents) and (v in self.latents):
                    mapping[u] = '_l_' + u
                elif (u not in self.latents) and (v not in self.latents):
                    mapping[u] = '_l_' + u
            self.latents.update(mapping.values())
            self.graph = nx.relabel_nodes(self.graph, mapping, copy=False)
            for u, v in mapping.items():
                self.graph.add_edge(v, u, weight=1.0)

        for node, di in self.graph.node.items():
            if node in self.latents:
                di['latent'] = True
            else:
                di['latent'] = False
                self.observed.add(node)

        # Create the latent variable structure
        self.latent_struct = self.graph.subgraph(self.latents)

        # Assign variables \eta, \xi
        self.eta = []
        self.xi = []
        latent_indegree = self.latent_struct.in_degree()

        for node in self.latent_struct.nodes():
            if latent_indegree[node]:
                self.eta.append(node)
            else:
                self.xi.append(node)

        # Assign variables y and x
        self.x = []
        self.y = []
        for exo in self.xi:
            self.x.extend([x for x in self.graph.neighbors(exo) if not x in self.latents])
        for endo in self.eta:
            self.y.extend([x for x in self.graph.neighbors(endo) if not x in self.latents])

        # Create a graph for correlation between error terms
        self.err_graph = nx.Graph(err_corr)
        # Add all the variables from y, x and \eta
        self.err_graph.add_nodes_from(self.observed)
        self.err_graph.add_nodes_from(self.eta)

    def _masks(self, weight):
        p, q, m, n = (len(self.y), len(self.x), len(self.eta), len(self.xi))

        nodelist = self.y + self.x + self.eta + self.xi
        adj_matrix = nx.to_numpy_matrix(self.graph, nodelist=nodelist, weight=weight)

        B_mask = adj_matrix[p+q:p+q+m, p+q:p+q+m]
        gamma_mask = adj_matrix[p+q+m:, p+q:p+q+m]
        wedge_y_mask = adj_matrix[p+q:p+q+m, 0:p]
        wedge_x_mask = adj_matrix[p+q+m:, p:p+q]
        phi_mask = np.ones((n, 1))

        err_nodelist = self.y + self.x + self.eta
        err_adj_matrix = nx.to_numpy_matrix(self.err_graph, nodelist=err_nodelist,
                                            weight=weight)
        theta_e_mask = err_adj_matrix[:p, :p]
        theta_del_mask = err_adj_matrix[p:p+q, p:p+q]
        psi_mask = err_adj_matrix[p+q:, p+q:]

        return (B_mask.T, gamma_mask.T, wedge_y_mask.T, wedge_x_mask.T, phi_mask.T,
                theta_e_mask.T, theta_del_mask.T, psi_mask.T)

    def get_fixed_masks(self):
        """
        Returns a fixed mask of all the algebriac parameters of the model.
        A fixed mask has the fixed value when the parameter is fixed otherwise
        has a value of 0.
        """
        return(self._masks(weight='weight'))

    def get_masks(self):
        """
        Returns masks of all the algebriac parameters of the model.
        A mask is a matrix with a value of 0's and 1's where 0 signifies
        no edge between the variables and 1 signifies an edge.

        Returns
        -------
        B_mask: np.ndarray (shape m x m)
            Mask for B matrix.

        gamma_mask: np.ndarray (shape m x n)
            Mask for \Gamma matrix.

        wedge_y_mask: np.ndarray (shape p x m)
            Mask for \wedge_y matrix.

        wedge_x_mask: np.ndarray (shape q x n)
            Mask for \wedge_x matrix.

        phi_mask: np.ndarray (shape n x 1)
            Mask for \phi matrix.

        theta_e_mask: np.ndarray (shape p x p)
            Mask for \theta_\epsilon matrix.

        theta_del_mask: np.ndarray (shape q x q)
            Mask for \theta_\delta matrix.

        psi_mask: np.ndarray (shape m x m)
            Mask for \psi matrix
        """
        # Arrage the adjecency matrix in order y, x, eta, xi and then slice masks from it.
        #       y(p)   x(q)   eta(m)  xi(n)
        # y     B   \Gamma  \wedge_y
        # x                         \wedge_x
        # eta
        # xi
        masks_arr = []
        for mask, fixed_mask in zip(self._masks(weight=None), self._masks(weight='weight')):
            masks_arr.append(np.multiply(np.where(fixed_mask != 0, 0.0, 1.0), mask))
        return tuple(masks_arr)

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
