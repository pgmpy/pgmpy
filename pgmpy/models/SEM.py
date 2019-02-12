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
            self.x.extend([x for x in self.graph.neighbors(exo) if x not in self.latents])
        for endo in self.eta:
            self.y.extend([x for x in self.graph.neighbors(endo) if x not in self.latents])

        # Remove duplicate elements from self.y and self.x. Also remove elements from self.x which are in self.y
        self.y = list(set(self.y))
        self.x = list(set(self.x) - set(self.x).intersection(set(self.y)))

        # Create a graph for correlation between error terms and
        # add all variables from y, x and \eta
        self.err_graph = nx.Graph(err_corr)
        self.err_graph.add_nodes_from(self.observed)
        self.err_graph.add_nodes_from(self.eta)
        self.err_graph.add_nodes_from(self.xi)

        # Set error correlations to np.NaN if not specified to be fixed.
        for edge in self.err_graph.edges:
            if not 'weight' in self.err_graph.edges[edge].keys():
                self.err_graph.edges[edge]['weight'] = np.NaN

    def _masks(self, weight, sort_vars=False):
        """
        This method is called by `get_fixed_masks` and `get_masks` methods.

        Parameters
        ----------
        weight: None | 'weight'
            If None: Returns a 1.0 for an edge in the graph else 0.0
            If 'weight': Returns the weight if a weight is assigned to an edge
                    else 0.0

        sort_vars: Boolean (default: False)
            If True: Individually sorts variables in x, y, eta and xi, and then
                    creates the adjecency matrix.
            If False: Directly uses the order of variables in self.x, self.y,
                    self.eta and self.xi to compute the adjecency matrix.

        Returns
        -------
        np.ndarray: Adjecency matrix of model's graph structure.

        Variable Name Reference
        -----------------------
        B: Effect matrix of eta on eta
        \gamma: Effect matrix of xi on eta
        \wedge_y: Effect matrix of eta on y
        \wedge_x: Effect matrix of xi on x
        \phi: Covariance matrix of xi
        \psi: Covariance matrix of eta errors
        \theta_e: Covariance matrix of y errors
        \theta_del: Covariance matrix of x errors
        """
        # Arrage the adjecency matrix in order y, x, eta, xi and then slice masks from it.
        #       y(p)   x(q)   eta(m)  xi(n)
        # y
        # x
        # eta \wedge_y          B
        # xi         \wedge_x \Gamma
        # 
        # But here we are slicing from the transpose of adjecency because we want incoming
        # edges instead of outgoing because parameters come before variables in equations.
        # 
        #       y(p)   x(q)   eta(m)  xi(n)
        # y                  \wedge_y
        # x                          \wedge_x
        # eta                   B    \Gamma
        # xi
        if sort_vars:
            y_vars, x_vars, eta_vars, xi_vars = (sorted(self.y), sorted(self.x),
                                                 sorted(self.eta), sorted(self.xi))
        else:
            y_vars, x_vars, eta_vars, xi_vars = self.y, self.x, self.eta, self.xi

        p, q, m, n = (len(y_vars), len(x_vars), len(eta_vars), len(xi_vars))

        nodelist = y_vars + x_vars + eta_vars + xi_vars
        adj_matrix = nx.to_numpy_matrix(self.graph, nodelist=nodelist, weight=weight).T

        B_mask = adj_matrix[p+q:p+q+m, p+q:p+q+m]
        gamma_mask = adj_matrix[p+q:p+q+m, p+q+m:]
        wedge_y_mask = adj_matrix[0:p, p+q:p+q+m]
        wedge_x_mask = adj_matrix[p:p+q, p+q+m:]

        # if weight is None:
        #     phi_mask = np.ones((n, 1))
        # elif weight == 'weight':
        #     phi_mask = np.zeros((n, 1))

        err_nodelist = y_vars + x_vars + eta_vars + xi_vars
        err_adj_matrix = nx.to_numpy_matrix(self.err_graph, nodelist=err_nodelist,
                                            weight=weight)

        if not weight == 'weight':
            np.fill_diagonal(err_adj_matrix, 1.0)

        theta_e_mask = err_adj_matrix[:p, :p]
        theta_del_mask = err_adj_matrix[p:p+q, p:p+q]
        psi_mask = err_adj_matrix[p+q:p+q+m, p+q:p+q+m]
        phi_mask = err_adj_matrix[p+q+m:, p+q+m:]

        return (B_mask, gamma_mask, wedge_y_mask, wedge_x_mask, phi_mask,
                theta_e_mask, theta_del_mask, psi_mask)

    def get_fixed_masks(self, sort_vars=False):
        """
        Returns a fixed mask of all the algebriac parameters of the model.
        A fixed mask has the fixed value when the parameter is fixed otherwise
        has a value of 0.

        Parameters
        ----------
        sort_vars: Boolean
            If True: Individually sorts variables in x, y, eta and xi, and then
                    creates the adjecency matrix.
            If False: Directly uses the order of variables in self.x, self.y,
                    self.eta and self.xi to compute the adjecency matrix.

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

        Examples
        --------
        """
        return(self._masks(weight='weight', sort_vars=sort_vars))

    def get_masks(self, sort_vars=False):
        """
        Returns masks of all the algebriac parameters of the model.
        A mask is a matrix with a value of 0's and 1's where 0 signifies
        no edge or a fixed parameter edge  between the variables and 1
        signifies an edge without a fixed value.

        While learning only the parameters with corresponing values of 1 in the
        mask will be learned.

        Parameters
        ----------
        sort_vars: Boolean
            If True: Individually sorts variables in x, y, eta and xi, and then
                    creates the adjecency matrix.
            If False: Directly uses the order of variables in self.x, self.y,
                    self.eta and self.xi to compute the adjecency matrix.

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

        Examples
        --------
        """
        masks_arr = []
        for mask, fixed_mask in zip(self._masks(weight=None, sort_vars=sort_vars),
                                    self._masks(weight='weight', sort_vars=sort_vars)):
            masks_arr.append(np.multiply(np.where(fixed_mask != 0, 0.0, 1.0), mask))
        return tuple(masks_arr)

    def _iv_transformations(self, X, Y, indicators={}):
        graph_copy = self.graph.copy()
        err_graph_copy = self.err_graph.copy()

        x_parent = set(graph_copy.predecessors(X))
        y_parent = set(graph_copy.predecessors(Y))
        common_parents = x_parent.intersection(y_parent)

        if common_parents:
            graph_copy.remove_edges_from([(parent, Y) for parent in common_parents])
            err_graph_copy.add_edge(X, Y)

        else:
            parent_latent = y_parent.pop()
            graph_copy.remove_edge(parent_latent, Y)
            y_parent_parent = set(self.latent_struct.predecessors(parent_latent))
            err_graph_copy.add_edges_from([(indicators[p], Y) for p in y_parent_parent])
            err_graph_copy.add_edge(parent_latent, Y)

        return graph_copy, err_graph_copy

    def _get_ancestral_iv(self, X, Y):
        pass

    def active_trail_nodes(self, variables, observed=None):
        """
        Returns a dictionary with the given variables as keys and all the nodes reachable
        from that respective variable as values.

        Parameters
        ----------

        variables: str or array like
            variables whose active trails are to be found.

        observed : List of nodes (optional)
            If given the active trails would be computed assuming these nodes to be observed.

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> student = BayesianModel()
        >>> student.add_nodes_from(['diff', 'intel', 'grades'])
        >>> student.add_edges_from([('diff', 'grades'), ('intel', 'grades')])
        >>> student.active_trail_nodes('diff')
        {'diff': {'diff', 'grades'}}
        >>> student.active_trail_nodes(['diff', 'intel'], observed='grades')
        {'diff': {'diff', 'intel'}, 'intel': {'diff', 'intel'}}

        References
        ----------
        Details of the algorithm can be found in 'Probabilistic Graphical Model
        Principles and Techniques' - Koller and Friedman
        Page 75 Algorithm 3.1
        """
        if observed:
            observed_list = observed if isinstance(observed, (list, tuple)) else [observed]
        else:
            observed_list = []

        ancestors_list = set()
        for node in observed_list:
            ancestors_list = ancestors_list.union(nx.algorithms.dag.ancestors(self.graph, node))

        # Direction of flow of information
        # up ->  from parent to child
        # down -> from child to parent

        active_trails = {}
        for start in variables if isinstance(variables, (list, tuple)) else [variables]:
            visit_list = set()
            visit_list.add((start, 'up'))
            traversed_list = set()
            active_nodes = set()
            while visit_list:
                node, direction = visit_list.pop()
                if (node, direction) not in traversed_list:
                    if (node not in observed_list) and (node not in self.latents):
                        active_nodes.add(node)
                    traversed_list.add((node, direction))
                    if direction == 'up' and node not in observed_list:
                        for parent in self.graph.predecessors(node):
                            visit_list.add((parent, 'up'))
                        for child in self.graph.successors(node):
                            visit_list.add((child, 'down'))
                    elif direction == 'down':
                        if node not in observed_list:
                            for child in self.graph.successors(node):
                                visit_list.add((child, 'down'))
                        if node in ancestors_list:
                            for parent in self.graph.predecessors(node):
                                visit_list.add((parent, 'up'))
            active_trails[start] = active_nodes
        return active_trails

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
