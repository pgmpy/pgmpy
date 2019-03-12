import networkx as nx
import numpy as np
import warnings
import itertools

from networkx.algorithms.dag import descendants

from pgmpy.base import DirectedGraph
from pgmpy.global_vars import HAS_PANDAS


if HAS_PANDAS:
    import pandas as pd


class SEMGraph(DirectedGraph):
    """
    Base class for graphical representation of Structural Equation Models(SEMs).

    All variables are by default assumed to have an associated error latent variable, therefore
    they don't need to be specified in the graph.

    Attributes
    ----------
    latents: list
        List of all the latent variables in the model except the error terms.

    observed: list
        List of all the observed variables in the model.

    graph: nx.DirectedGraph
        The graphical structure of the latent and observed variables except the error terms.

    err_corr: nx.Graph
        An undirected graph representing the relations between the error terms of the model.
        The error terms use the same name as variables themselves.
    """
    def __init__(self, ebunch=[], latents=[], err_corr=[], err_var={}):
        """
        Initializes `SEMGraph` object.

        Parameters
        ----------
        ebunch: list/array-like
            List of edges in form of tuples. Each tuple can be of two possible shape:
                1. (u, v): This would add an edge from u to v without setting any parameter
                           for the edge.
                2. (u, v, parameter): This would add an edge from u to v and set the edge's
                            parameter to `parameter`.

        latents: list/array-like
            List of nodes which are latent. All other variables are considered observed.

        err_corr: list/array-like
            List of tuples representing edges between error terms. It can be of the following forms:
                1. (u, v): Add correlation between error terms of `u` and `v`. Doesn't set any variance or
                           covariance values.
                2. (u, v, covar): Adds correlation between the error terms of `u` and `v` and sets the
                                  parameter to `covar`.

        err_var: dict
            Dict of the form (var: variance).

        Examples
        --------
        Defining a model (Union sentiment model[1]) without setting any paramaters.
        >>> from pgmpy.models import SEMGraph
        >>> sem = SEMGraph(ebunch=[('deferenc', 'unionsen'), ('laboract', 'unionsen'),
        ...                        ('yrsmill', 'unionsen'), ('age', 'deferenc'),
        ...                        ('age', 'laboract'), ('deferenc', 'laboract')],
        ...                latents=[],
        ...                err_corr=[('yrsmill', 'age')],
        ...                err_var={})

        Defining a model (Education [2]) with all the parameters set. For not setting any
        parameter `np.NaN` can be explicitly passed.
        >>> sem_edu = SEMGraph(ebunch=[('intelligence', 'academic', 0.8), ('intelligence', 'scale_1', 0.7),
        ...                            ('intelligence', 'scale_2', 0.64), ('intelligence', 'scale_3', 0.73),
        ...                            ('intelligence', 'scale_4', 0.82), ('academic', 'SAT_score', 0.98),
        ...                            ('academic', 'High_school_gpa', 0.75), ('academic', 'ACT_score', 0.87)],
        ...                    latents=['intelligence', 'academic'],
        ...                    err_corr=[]
        ...                    err_var={})

        References
        ----------
        [1] McDonald, A, J., & Clelland, D. A. (1984). Textile Workers and Union Sentiment.
            Social Forces, 63(2), 502–521
        [2] https://en.wikipedia.org/wiki/Structural_equation_modeling#/
            media/File:Example_Structural_equation_model.svg
        """
        super(SEMGraph, self).__init__()

        # Construct the graph and set the parameters.
        self.graph = nx.DiGraph()
        for t in ebunch:
            if len(t) == 3:
                self.graph.add_edge(t[0], t[1], weight=t[2])
            elif len(t) == 2:
                self.graph.add_edge(t[0], t[1], weight=np.NaN)
            else:
                raise ValueError("Expected tuple length: 2 or 3. Got {t} of len {shape}".format(
                                                        t=t, shape=len(t)))

        self.latents = set(latents)
        self.observed = set(self.graph.nodes()) - self.latents

        # Construct the error graph and set the parameters.
        self.err_graph = nx.Graph()
        self.err_graph.add_nodes_from(self.graph.nodes())
        for t in err_corr:
            if len(t) == 2:
                self.err_graph.add_edge(t[0], t[1], weight=np.NaN)
            elif len(t) == 3:
                self.err_graph.add_edge(t[0], t[1], weight=t[2])
            else:
                raise ValueError("Expected tuple length: 2 or 3. Got {t} of len {shape}".format(
                                                        t=t, shape=len(t)))
        for var in self.err_graph.nodes():
            self.err_graph.nodes[var]['weight'] = err_var[var] if var in err_var.keys() else np.NaN

        self.full_graph_struct = self._get_full_graph_struct()

    def _get_full_graph_struct(self):
        """
        Creates a directed graph by joining `self.graph` and `self.err_graph`.
        Adds new nodes to replace undirected edges (u <--> v) with two directed
        edges (u <-- ..uv) and (..uv --> v).

        Returns
        -------
        nx.DiGraph: A full directed graph strucuture with error nodes starting
                    with `.` and bidirected edges replaced with common cause
                    nodes starting with `..`.

        Examples
        --------
        >>> from pgmpy.models import SEMGraph
        >>> sem = SEMGraph(ebunch=[('deferenc', 'unionsen'), ('laboract', 'unionsen'),
        ...                        ('yrsmill', 'unionsen'), ('age', 'deferenc'),
        ...                        ('age', 'laboract'), ('deferenc', 'laboract')],
        ...                latents=[],
        ...                err_corr=[('yrsmill', 'age')])
        >>> sem.get_full_graph_struct()
        """
        full_graph = self.graph.copy()

        mapping_dict = {'.'+node: node for node in self.err_graph.nodes}
        full_graph.add_edges_from([(u, v) for u, v in mapping_dict.items()])
        for u, v in self.err_graph.edges:
            cov_node = '..' + u + v
            full_graph.add_edges_from([(cov_node, '.' + u), (cov_node, '.'+ v)])

        return full_graph

    def get_scaling_indicators(self):
        """
        Returns a random scaling indicator for each latent variable in the model.
        """
        scaling_indicators = {}
        for node in self.latents:
            for neighbor in self.graph.neighbors(node):
                if neighbor in self.observed:
                    scaling_indicators[node] = neighbor
                    break
        return scaling_indicators

    def active_trail_nodes(self, variables, observed=[], struct='full'):
        """
        Finds all the observed variables which are d-connected to `variables` in the `graph_struct`
        when `observed` variables are observed.

        Parameters
        ----------
        variables: str or array like
            Observed variables whose d-connected variables are to be found.

        observed : list/array-like
            If given the active trails would be computed assuming these nodes to be observed.

        struct: str or nx.DiGraph instance
            If "full", is used considers correlation between error terms for computing d-connection.
            If "non_error", doesn't condised error correlations for computing d-connection.
            If nx.DiGraph, finds d-connected variables on the given graph.

        Examples
        --------

        References
        ----------
        Details of the algorithm can be found in 'Probabilistic Graphical Model
        Principles and Techniques' - Koller and Friedman
        Page 75 Algorithm 3.1
        """
        if struct == 'full':
            graph_struct = self.full_graph_struct
        elif struct == 'non_error':
            graph_struct = self.graph
        elif isinstance(struct, nx.DiGraph):
            graph_struct = struct
        else:
            raise ValueError("Expected struct to be str or nx.DiGraph. Got {t}".format(t=type(struct)))

        ancestors_list = set()
        for node in observed:
            ancestors_list = ancestors_list.union(nx.algorithms.dag.ancestors(graph_struct, node))

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
                    if (node not in observed) and (not node.startswith('.')) and (node not in self.latents):
                        active_nodes.add(node)
                    traversed_list.add((node, direction))
                    if direction == 'up' and node not in observed:
                        for parent in graph_struct.predecessors(node):
                            visit_list.add((parent, 'up'))
                        for child in graph_struct.successors(node):
                            visit_list.add((child, 'down'))
                    elif direction == 'down':
                        if node not in observed:
                            for child in graph_struct.successors(node):
                                visit_list.add((child, 'down'))
                        if node in ancestors_list:
                            for parent in graph_struct.predecessors(node):
                                visit_list.add((parent, 'up'))
            active_trails[start] = active_nodes
        return active_trails

    def _iv_transformations(self, X, Y, scaling_indicators={}):
        """
        Transforms the graph structure of SEM so that the d-separation criterion is
        applicable for finding IVs. The method transforms the graph for finding MIIV
        for the estimation of X \rightarrow Y given the scaling indicator for all the
        parent latent variables.

        Parameters
        ----------
        X: node
            The explantory variable.
        Y: node
            The dependent variable.

        Returns
        -------
        nx.DiGraph, nx.Graph: The transformed latent graph and the transformed error
                              graph.

        Examples
        --------
        """
        full_graph = self.full_graph_struct.copy()
        x_parent = set(self.graph.predecessors(X))
        y_parent = set(self.graph.predecessors(Y))
        common_parents = x_parent.intersection(y_parent)

        if common_parents:
            full_graph.remove_edges_from([(parent, Y) for parent in common_parents])
            full_graph.add_edge('.'+X, Y)

        else:
            parent_latent = y_parent.pop()
            full_graph.remove_edge(parent_latent, Y)
            y_parent_parent = set(self.latent_struct.predecessors(parent_latent))
            full_graph.add_edges_from([('.'+scaling_indicators[p], Y) for p in y_parent_parent])
            full_graph.add_edge('.'+parent_latent, Y)

        return full_graph

    def get_ivs(self, X, Y, scaling_indicators={}):
        """
        Returns the Instrumental variables for the relation X -> Y

        Parameters
        ----------
        X: node
            The observed variable name
        Y: node
            The observed variable name
        scaling_indicators: dict
            A dict representing which observed variable to use as scaling indicator for
            the latent variables.

        Returns
        -------
        set: The set of Instrumental Variables for the predicted value.

        Examples
        --------
        """
        transformed_graph = self._iv_transformations(X, Y, scaling_indicators=scaling_indicators)
        d_connected = self.active_trail_nodes([X, Y], struct=transformed_graph)
        return (d_connected[X] - d_connected[Y])

    def moralize(self, graph='full'):
        """
        Removes all the immoralities in the DirectedGraph and creates a moral
        graph (UndirectedGraph).

        A v-structure X->Z<-Y is an immorality if there is no directed edge
        between X and Y.

        Parameters
        ----------
        graph:

        Examples
        --------
        """
        if graph == 'full':
            graph = self.full_graph_struct
        else:
            graph = self.graph

        moral_graph = graph.to_undirected()

        for node in graph.nodes():
            moral_graph.add_edges_from(
                itertools.combinations(graph.predecessors(node), 2))

        return moral_graph

    def _nearest_separator(self, G, Y, Z):
        W = set()
        for path in nx.all_simple_paths(G, Y, Z):
            path_set = set(path)
            if (len(path) >= 3) and not (W & path_set):
                for index in range(1, len(path)-1):
                    if (path[index] in self.observed) or (path[index].startswith('_l_')):
                        W.add(path[index])
                        break
        if Y not in self.active_trail_nodes([Z], observed=W)[Z]:
            return W
        else:
            return None

    def get_conditional_ivs(self, X, Y, scaling_indicators={}):
        """
        Returns the conditional IVs for the relation X -> Y

        Parameters
        ----------
        X: node
            The observed variable's name

        Y: node
            The oberved variable's name

        scaling_indicators: dict
            A dict representing which observed variable to use as scaling indicator for
            the latent variables.

        Returns
        -------
        set: Set of 2-tuples representing tuple[0] is an IV for X -> Y given tuple[1].

        References
        ----------
        .. [1] Van Der Zander, B., Textor, J., & Liskiewicz, M. (2015, June). Efficiently finding
               conditional instruments for causal inference. In Twenty-Fourth International Joint
               Conference on Artificial Intelligence.

        Examples
        --------
        """
        transformed_graph = self._iv_transformations(X, Y, scaling_indicators=scaling_indicators)

        if (X, Y) in transformed_graph.edges:
            G_c = transformed_graph.removed_edge(X, Y)
        else:
            G_c = transformed_graph

        instruments = []
        for Z in (self.observed - {X, Y}):
            W = self._nearest_separator(self.moralize(graph=G_c), Y, Z)
            if (W is None) or (W.intersection(descendants(G_c, Y))) or (X in W):
                continue
            elif X in self.active_trail_nodes([Z], observed=W, struct=G_c)[Z]:
                instruments.append((Z, W))
            else:
                continue
        return instruments

    @staticmethod
    def __masks(graph, err_graph, weight, var):
        """
        This method is called by `get_fixed_masks` and `get_masks` methods.

        Parameters
        ----------
        weight: None | 'weight'
            If None: Returns a 1.0 for an edge in the graph else 0.0
            If 'weight': Returns the weight if a weight is assigned to an edge
                    else 0.0

        var: dict
            Dict with keys eta, xi, y, and x representing the variables in them.

        Returns
        -------
        np.ndarray: Adjecency matrix of model's graph structure.

        Notes
        -----
        B: Effect matrix of eta on eta
        \gamma: Effect matrix of xi on eta
        \wedge_y: Effect matrix of eta on y
        \wedge_x: Effect matrix of xi on x
        \phi: Covariance matrix of xi
        \psi: Covariance matrix of eta errors
        \theta_e: Covariance matrix of y errors
        \theta_del: Covariance matrix of x errors

        Examples
        --------
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
        y_vars, x_vars, eta_vars, xi_vars = var['y'], var['x'], var['eta'], var['xi']

        p, q, m, n = (len(y_vars), len(x_vars), len(eta_vars), len(xi_vars))

        nodelist = y_vars + x_vars + eta_vars + xi_vars
        adj_matrix = nx.to_numpy_matrix(graph, nodelist=nodelist, weight=weight).T

        B_mask = adj_matrix[p+q:p+q+m, p+q:p+q+m]
        gamma_mask = adj_matrix[p+q:p+q+m, p+q+m:]
        wedge_y_mask = adj_matrix[0:p, p+q:p+q+m]
        wedge_x_mask = adj_matrix[p:p+q, p+q+m:]

        err_nodelist = y_vars + x_vars + eta_vars + xi_vars
        err_adj_matrix = nx.to_numpy_matrix(err_graph, nodelist=err_nodelist,
                                            weight=weight)

        if not weight == 'weight':
            np.fill_diagonal(err_adj_matrix, 1.0)

        theta_e_mask = err_adj_matrix[:p, :p]
        theta_del_mask = err_adj_matrix[p:p+q, p:p+q]
        psi_mask = err_adj_matrix[p+q:p+q+m, p+q:p+q+m]
        phi_mask = err_adj_matrix[p+q+m:, p+q+m:]

        return {'B': B_mask, 'gamma': gamma_mask, 'wedge_y': wedge_y_mask, 'wedge_x': wedge_x_mask,
                'phi': phi_mask, 'theta_e': theta_e_mask, 'theta_del': theta_del_mask, 'psi': psi_mask}

#     @staticmethod
#     def __get_masks(graph, err_graph, var):
#         """
#         Returns masks of all the algebriac parameters of the model.
#         A mask is a matrix with a value of 0's and 1's where 0 signifies
#         no edge or a fixed parameter edge  between the variables and 1
#         signifies an edge without a fixed value.
# 
#         While learning only the parameters with corresponing values of 1 in the
#         mask will be learned.
# 
#         Parameters
#         ----------
#         sort_vars: Boolean
#             If True: Individually sorts variables in x, y, eta and xi, and then
#                     creates the adjecency matrix.
#             If False: Directly uses the order of variables in self.x, self.y,
#                     self.eta and self.xi to compute the adjecency matrix.
# 
#         Returns
#         -------
#         B_mask: np.ndarray (shape m x m)
#             Mask for B matrix.
# 
#         gamma_mask: np.ndarray (shape m x n)
#             Mask for \Gamma matrix.
# 
#         wedge_y_mask: np.ndarray (shape p x m)
#             Mask for \wedge_y matrix.
# 
#         wedge_x_mask: np.ndarray (shape q x n)
#             Mask for \wedge_x matrix.
# 
#         phi_mask: np.ndarray (shape n x 1)
#             Mask for \phi matrix.
# 
#         theta_e_mask: np.ndarray (shape p x p)
#             Mask for \theta_\epsilon matrix.
# 
#         theta_del_mask: np.ndarray (shape q x q)
#             Mask for \theta_\delta matrix.
# 
#         psi_mask: np.ndarray (shape m x m)
#             Mask for \psi matrix
# 
#         Examples
#         --------
#         """
#         pass
#         # masks_arr = []
#         # for mask, fixed_mask in zip(SEMGraph.__masks(
#         #                                 graph=graph, err_graph=err_graph, weight=None, var=var),
#         #                             SEMGraph.__masks(
#         #                                 graph=graph, err_graph=err_graph, weight='weight', var=var)):
#         #     masks_arr.append(np.multiply(np.where(fixed_mask != 0, 0.0, 1.0), mask))
#         # return tuple(masks_arr)
# 
#     @staticmethod
#     def __get_fixed_masks(graph, err_graph, var):
#         """
#         Returns a fixed mask of all the algebriac parameters of the model.
#         A fixed mask has the fixed value when the parameter is fixed otherwise
#         has a value of 0.
# 
#         Parameters
#         ----------
#         sort_vars: Boolean
#             If True: Individually sorts variables in x, y, eta and xi, and then
#                     creates the adjecency matrix.
#             If False: Directly uses the order of variables in self.x, self.y,
#                     self.eta and self.xi to compute the adjecency matrix.
# 
#         Returns
#         -------
#         B_mask: np.ndarray (shape m x m)
#             Mask for B matrix.
# 
#         gamma_mask: np.ndarray (shape m x n)
#             Mask for \Gamma matrix.
# 
#         wedge_y_mask: np.ndarray (shape p x m)
#             Mask for \wedge_y matrix.
# 
#         wedge_x_mask: np.ndarray (shape q x n)
#             Mask for \wedge_x matrix.
# 
#         phi_mask: np.ndarray (shape n x 1)
#             Mask for \phi matrix.
# 
#         theta_e_mask: np.ndarray (shape p x p)
#             Mask for \theta_\epsilon matrix.
# 
#         theta_del_mask: np.ndarray (shape q x q)
#             Mask for \theta_\delta matrix.
# 
#         psi_mask: np.ndarray (shape m x m)
#             Mask for \psi matrix
# 
#         Examples
#         --------
#         """
#         return(self.__masks(graph=graph, err_graph=err_graph, weight='weight', var=var))

    def __to_standard_lisrel(self):
        """
        Converts the model structure into the standard LISREL notation of latent structure and
        measurement structure by adding new latent variables
        """
        lisrel_err_graph = self.err_graph.copy()
        lisrel_latents = self.latents.copy()
        lisrel_observed = self.observed.copy()

        # Add new latent nodes to convert it to LISREL format.
        mapping = {}
        for u, v in self.graph.edges:
            if (u not in self.latents) and (v in self.latents):
                mapping[u] = '_l_' + u
            elif (u not in self.latents) and (v not in self.latents):
                mapping[u] = '_l_' + u
        lisrel_latents.update(mapping.values())
        lisrel_graph = nx.relabel_nodes(self.graph, mapping, copy=True)
        for u, v in mapping.items():
            lisrel_graph.add_edge(v, u, weight=1.0)

        # Get values of eta, xi, y, x
        latent_struct = lisrel_graph.subgraph(lisrel_latents)
        latent_indegree = lisrel_graph.in_degree()

        eta = []
        xi = []
        for node in latent_struct.nodes():
            if latent_indegree[node]:
                eta.append(node)
            else:
                xi.append(node)

        x = []
        y = []
        for exo in xi:
            x.extend([x for x in lisrel_graph.neighbors(exo) if x not in lisrel_latents])
        for endo in eta:
            y.extend([y for y in lisrel_graph.neighbors(endo) if y not in lisrel_latents])

        y = list(set(y))
        x = list(set(x) - set(x).intersection(set(y)))

        return (lisrel_graph, lisrel_err_graph, {'eta': eta, 'xi': xi, 'y': y, 'x': x})

    def to_lisrel(self):
        """
        Gets parameters from the graph structure to the standard LISREL matrix representation.

        Returns
        -------
        dict: Dict with the keys B, gamma, wedge_y, wedge_x, theta_e, theta_del, phi and psi.

        Examples
        --------
        """
        lisrel_graph, lisrel_err_graph, var = self.__to_standard_lisrel()

        edges_mask = self.__masks(graph=lisrel_graph, err_graph=lisrel_err_graph, weight=None, var=var)
        fixed_mask = self.__masks(graph=lisrel_graph, err_graph=lisrel_err_graph, weight='weight', var=var)

        from pgmpy.models import SEMLISREL
        return SEMLISREL(var_names=var, params=edges_mask, fixed_params=fixed_mask)
