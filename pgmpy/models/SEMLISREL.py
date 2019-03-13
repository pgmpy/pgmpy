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

        Returns
        -------
        pgmpy.models.SEMLISREL instance: An instance of the object with initalized values.

        Examples
        --------
        >>> from pgmpy.models import SEMLISREL
        # TODO: Finish this example
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

    def __to_minimal_graph(self, graph):
        """
        Takes a standard LISREL graph structure and removes all the extra added latents.

        Parameters
        ----------
        graph: nx.DiGraph
            The graph structure except for the error terms.

        err_graph: nx.Graph
            The error graph structure.

        Returns
        -------
        nx.DiGraph: The graph structure after removing all the reducible latent terms
                    which start with `_l_`.

        Examples
        --------
        >>> from pgmpy.models import SEMGraph
        # TODO: Finish this example.
        """
        mapping = {}
        for node in chain(self.var_names['xi'], self.var_names['eta']):
            if node.startswith('_l_'):
                mapping[node] = node[3:]

        for u, v in mapping.items():
            graph.remove_node(v)

        return nx.relabel_nodes(graph, mapping, copy=True)

    def to_SEMGraph(self):
        """
        Creates a graph structure from the LISREL representation.

        Returns
        -------
        pgmpy.models.SEMGraph instance: A path model of the model.

        Examples
        --------
        >>> from pgmpy.models import SEMLISREL
        >>> model = SEMLISREL()
        # TODO: Finish this example
        """
        y_vars, x_vars, eta_vars, xi_vars = (self.var_names['y'], self.var_names['x'],
                                             self.var_names['eta'], self.var_names['xi'])

        p, q, m, n = (len(y_vars), len(x_vars), len(eta_vars), len(xi_vars))

        graph_adj_matrix = np.zeros((p+q+m+n, p+q+m+n))
        err_graph_adj_matrix = np.zeros((p+q+m+n, p+q+m+n))

        # Replacing non fixed edges with np.NaN as networkx assumes that as edges as well
        # and sets weights to NaN.
        adj_sub_matrices = {}
        for key in self.masks.keys():
            adj_sub_matrices[key] = np.where((self.masks[key] + self.fixed_masks[key]) == 1,
                                             np.NaN, self.masks[key] + self.fixed_masks[key])

        node_dict = {i: node_name for i, node_name in enumerate(chain(y_vars, x_vars, eta_vars, xi_vars))}
        graph_adj_matrix[p+q:p+q+m, p+q:p+q+m] = adj_sub_matrices['B']
        graph_adj_matrix[p+q:p+q+m, p+q+m:] = adj_sub_matrices['gamma']
        graph_adj_matrix[0:p, p+q:p+q+m] = adj_sub_matrices['wedge_y']
        graph_adj_matrix[p:p+q, p+q+m:] = adj_sub_matrices['wedge_x']
        graph = nx.convert_matrix.from_numpy_matrix(graph_adj_matrix.T, create_using=nx.DiGraph)
        graph = nx.relabel_nodes(graph, mapping=node_dict)

        err_graph_adj_matrix[:p, :p] = adj_sub_matrices['theta_e']
        err_graph_adj_matrix[p:p+q, p:p+q] = adj_sub_matrices['theta_del']
        err_graph_adj_matrix[p+q:p+q+m, p+q:p+q+m] = adj_sub_matrices['psi']
        err_graph_adj_matrix[p+q+m:, p+q+m:] = adj_sub_matrices['phi']
        # To remove self edges on error terms because of variance fill diagonal with 0.
        np.fill_diagonal(err_graph_adj_matrix, 0)

        err_graph = nx.convert_matrix.from_numpy_matrix(err_graph_adj_matrix.T, create_using=nx.Graph)
        err_graph = nx.relabel_nodes(err_graph, mapping=node_dict)

        minimal_graph = self.__to_minimal_graph(graph)

        err_var = {node_dict[i]: err_graph_adj_matrix[i, i] for i in range(p+q+m+n)}

        from pgmpy.models import SEMGraph
        sem_graph = SEMGraph(ebunch=minimal_graph.edges(),
                             latents=list(filter(lambda t: not t.startswith('_l_'), eta_vars+xi_vars)),
                             err_corr=err_graph.edges(),
                             err_var=err_var)
        return sem_graph

