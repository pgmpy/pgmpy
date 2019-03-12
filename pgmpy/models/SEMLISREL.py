from itertools import chain

import networkx as nx
import numpy as np


class SEMLISREL:
    """
    Base class for algebraic representation of Structural Equation Models(SEMs).
    """
    def __init__(self, str_model=None, var_names=None, params=None, fixed_masks=None):
        r"""
        Initializes SEMLISREL model. The LISREL notation is defined as:
        ..math::
            \mathbf{\eta} = \mathbf{B \eta} + \mathbf{\Gamma \xi} + mathbf{\zeta} \\
            \mathbf{y} = \mathbf{\wedge_y \eta} + \mathbf{\epsilon} \\
            \mathbf{x} = \mathbf{\wedge_x \xi} + \mathbf{\delta}

        where :math:`\mathbf{\eta}` is the set of endogenous variables, :math:`\mathbf{\xi}`
        is the set of exogeneous variables, :math:`\mathbf{y}` and :math:`\mathbf{x}` are the
        set of measurement variables for :math:`\mathbf{\eta}` and :math:`\mathbf{\xi}`
        respectively. :math:`\mathbf{\zeta}`, :math:`\mathbf{\epsilon}`, and :math:`\mathbf{\delta}`
        are the error terms for :math:`\mathbf{\eta}`, :math:`\mathbf{y}`, and :math:`\mathbf{x}`
        respectively.

        Parameters
        ----------
        str_model: str (default: None)
            A `lavaan` style multiline set of regression equation representing the model.
            Refer http://lavaan.ugent.be/tutorial/syntax1.html for details.

            If None requires `var_names` and `params` to be specified.

        var_names: dict (default: None)
            A dict with the keys: eta, xi, y, and x. Each keys should have a list as the value
            with the name of variables.

        params: dict (default: None)
            A dict of LISREL representation non-zero parameters. Must contain the following
            keys: B, gamma, wedge_y, wedge_x, phi, theta_e, theta_del, and psi.

            If None `str_model` must be specified.

        fixed_params: dict (default: None)
            A dict of fixed values for parameters. The shape of the parameters should be same
            as params.

            If None all the parameters are learnable.
        """
        if str_model:
            raise NotImplementedError("Specification of model as a string is not supported yet")

        param_names = ['B', 'gamma', 'wedge_y', 'wedge_x', 'phi', 'theta_e', 'theta_del', 'psi']

        # Check if params has all the keys and sanitize fixed params.
        for p_name in param_names:
            if p_name not in params.keys():
                raise ValueError("params must have the parameter {p_name}".format(p_name=p_name))
            if p_name not in fixed_masks.keys():
                fixed_masks[p_name] = np.zeros(params[p_name].shape)

        self.var_names = var_names
        self.adjecency = params
        self.fixed_masks = fixed_masks

        # Masks represent the parameters which need to be learnt while training.
        self.masks = {}
        for key in self.adjecency.keys():
            self.masks[key] = np.multiply(np.where(self.fixed_masks[key] != 0, 0.0, 1.0), self.adjecency[key])

    def __to_minimal_graph(self, graph, err_graph):
        """
        Takes a standard LISREL graph structure and removes all the extra added latents.

        graph: nx.DiGraph
            The graph structure except for the error terms.

        err_graph: nx.Graph
            The error graph structure.
        """
        mapping = {}
        for node in chain(self.var_names['xi'], self.var_names['eta']):
            if node.startswith('_l_'):
                mapping[node] = node[3:]

        for u, v in mapping.items():
            graph.remove_node(v)
            graph.remove_edge(u, v)

        return nx.relabel_nodes(graph, mapping, copy=True)

    def to_SEMGraph(self):
        """
        Creates a graph structure from the LISREL represntation.
        """
        # Construct the full ajceceny matrix.
        # Construct the full error adjececny matrix.
        # Construct graphs
        # Minimal graph
        # Return

#         """
#         Sets the params in the graph structure. The model is defined as:
# 
#         eta = B eta + Gamma xi + zeta
#         y = wedge_y eta + epsilon
#         x = wedge_x xi + delta
#         Psi = COV(eta)
#         Phi = COV(xi)
#         theta_e = COV(y)
#         theta_{del} = COV(x)
# 
#         Parameters
#         ----------
#         params: dict
#             A dictionary with the parameters to set to the model. Each array should assume that
#             the variable lists eta, xi, y and x are sorted.
# 
#             Expected keys in `params`:
#                 1. B: array (m x m)
#                 2. gamma: array (m x n)
#                 3. wedge_y: array (p x m)
#                 4. wedge_x: array (q x n)
#                 5. phi: array (n x n)
#                 6. theta_e: array (p x p)
#                 7. theta_del: array (q x q)
#                 8. psi: array (m x m)
#             m = len(self.eta)
#             n = len(self.xi)
#             p = len(self.y)
#             q = len(self.x)
# 
#         Returns
#         -------
#         None
# 
#         Examples
#         --------
#         """
#         # Edges to use to inialize SEMGraph.
#         ebunch = []
#         err_ebunch = []
#         err_var = {}
# 
#         # Sort the variable names so that it gets correct values assigned
#         eta = self.var_names['eta']
#         xi = self.var_names['xi']
#         y = self.var_names['y']
#         x = self.var_names['x']
# 
#         # Add edges and weights
#         for i, u_var in enumerate(eta):
#             for j, v_var in enumerate(eta):
#                 if self.params['B'][i, j] == 1:
#                     ebunch.append((v_var, u_var,
#                                    np.NaN if self.fixed_params['B'][i, j]==0 else self.fixed_params['B'][i, j]))
# 
#         for i, u_var in enumerate(eta):
#             for j, v_var in enumerate(xi):
#                 if self.params['gamma'][i, j] == 1:
#                     ebunch.append(
#                         (v_var, u_var,
#                          np.NaN if self.fixed_params['gamma'][i, j]==0 else self.fixed_params['gamma'][i, j]))
# 
#         for i, u_var in enumerate(y):
#             for j, v_var in enumerate(eta):
#                 if self.parms['wedge_y'][i, j] == 1:
#                     ebunch.append((v_var, u_var, np.NaN if self.fixed_params['wedge_y'][i, j]==0
#                                    else self.fixed_params['wedge_y'][i, j]))
# 
#         for i, u_var in enumerate(x):
#             for j, v_var in enumerate(xi):
#                 if self.params['wedge_x'][i, j] == 1:
#                     ebunch.append((v_var, u_var, np.NaN if self.fixed_params['wedge_x'][i, j]==0
#                                    else self.fixed_params['wedge_x'][i, j]))
# 
#         for i, u_var in enumerate(xi):
#             for j, v_var in enumerate(xi):
#                 if i == j:
#                     err_var[u_var] = (np.NaN if self.fixed_params['phi'][i, j]==0 else
#                                       self.fixed_params['phi'][i, j])
#                 else:
#                     err_ebunch.append(u_var, v_var, np.NaN if self.fixed_params['phi'][i, j]==0 else
#                                       self.fixed_params['phi'][i, j])
# 
#         for i, u_var in enumerate(eta):
#             for j, v_var in enumerate(eta):
#                 if i == j:
#                     err_var[u_var] = (np.NaN if self.fixed_params['psi'][i, j]==0 else
#                                       self.fixed_params['psi'][i, j])
#                 else:
#                     err_ebunch.append(u_var, v_var, np.NaN if self.fixed_params['psi'][i, j]==0 else
#                                       self.fixed_params['psi'][i, j])
# 
#         for i, u_var in enumerate(y):
#             for j, v_var in enumerate(y):
#                 if i == j:
#                     err_var[u_var] = (np.NaN if self.fixed_params['theta_e'][i, j]==0 else
#                                       self.fixed_params['theta_e'][i, j])
#                 else:
#                     err_ebunch.append(u_var, v_var, np.NaN if self.fixed_params['theta_e'][i, j]==0 else
#                                       self.fixed_params['theta_e'][i, j])
# 
#         for i, u_var in enumerate(x):
#             for j, v_var in enumerate(x):
#                 if i == j:
#                     err_var[u_var] = (np.NaN if self.fixed_params['theta_del'][i, j]==0 else
#                                       self.fixed_params['theta_del'][i, j])
#                 else:
#                     err_ebunch.append(u_var, v_var, np.NaN if self.fixed_params['theta_del'][i, j]==0 else
#                                       self.fixed_params['theta_del'][i, j])
# 
#         # Construct the SEMGraph object and return it.
#         from pgmpy.models import SEMGraph
#         sem_graph = SEMGraph(ebunch=ebunch, latents=eta+xi, err_corr=err_ebunch)
#         for node in sem_graph.err_graph.nodes():
#             sem_graph.err_graph[node]['var'] = err_var[node]
# 
#         return sem_graph
