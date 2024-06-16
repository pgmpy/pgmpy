import itertools

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.dag import descendants
from pyparsing import OneOrMore, Optional, Suppress, Word, alphanums, nums

from pgmpy.base import DAG
from pgmpy.global_vars import logger


class SEMGraph(DAG):
    """
    Base class for graphical representation of Structural Equation Models(SEMs).

    All variables are by default assumed to have an associated error latent variable, therefore
    doesn't need to be specified.

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

    err_var: dict (variable: variance)
        Sets variance for the error terms in the model.

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
    parameter `np.nan` can be explicitly passed.
    >>> sem_edu = SEMGraph(ebunch=[('intelligence', 'academic', 0.8), ('intelligence', 'scale_1', 0.7),
    ...                            ('intelligence', 'scale_2', 0.64), ('intelligence', 'scale_3', 0.73),
    ...                            ('intelligence', 'scale_4', 0.82), ('academic', 'SAT_score', 0.98),
    ...                            ('academic', 'High_school_gpa', 0.75), ('academic', 'ACT_score', 0.87)],
    ...                    latents=['intelligence', 'academic'],
    ...                    err_corr=[],
    ...                    err_var={'intelligence': 1})

    References
    ----------
    [1] McDonald, A, J., & Clelland, D. A. (1984). Textile Workers and Union Sentiment.
        Social Forces, 63(2), 502–521
    [2] https://en.wikipedia.org/wiki/Structural_equation_modeling#/
        media/File:Example_Structural_equation_model.svg

    Attributes
    ----------
    latents: list
        List of all the latent variables in the model except the error terms.

    observed: list
        List of all the observed variables in the model.

    graph: nx.DirectedGraph
        The graphical structure of the latent and observed variables except the error terms.
        The parameters are stored in the `weight` attribute of each edge.

    err_graph: nx.Graph
        An undirected graph representing the relations between the error terms of the model.
        The node of the graph has the same name as the variable but represents the error terms.
        The variance is stored in the `weight` attribute of the node and the covariance are stored
        in the `weight` attribute of the edge.

    full_graph_struct: nx.DiGraph
        Represents the full graph structure. The names of error terms start with `.` and
        new nodes are added for each correlation which starts with `..`.

    """

    def __init__(self, ebunch=[], latents=[], err_corr=[], err_var={}):
        super(SEMGraph, self).__init__()

        # Construct the graph and set the parameters.
        self.graph = nx.DiGraph()
        for t in ebunch:
            if len(t) == 3:
                self.graph.add_edge(t[0], t[1], weight=t[2])
            elif len(t) == 2:
                self.graph.add_edge(t[0], t[1], weight=np.nan)
            else:
                raise ValueError(
                    f"Expected tuple length: 2 or 3. Got {t} of len {len(t)}"
                )

        self.latents = set(latents)
        self.observed = set(self.graph.nodes()) - self.latents

        # Construct the error graph and set the parameters.
        self.err_graph = nx.Graph()
        self.err_graph.add_nodes_from(self.graph.nodes())
        for t in err_corr:
            if len(t) == 2:
                self.err_graph.add_edge(t[0], t[1], weight=np.nan)
            elif len(t) == 3:
                self.err_graph.add_edge(t[0], t[1], weight=t[2])
            else:
                raise ValueError(
                    f"Expected tuple length: 2 or 3. Got {t} of len {len(t)}"
                )

        # Set the error variances
        for var in self.err_graph.nodes():
            self.err_graph.nodes[var]["weight"] = (
                err_var[var] if var in err_var.keys() else np.nan
            )

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
        >>> sem._get_full_graph_struct()
        """
        full_graph = self.graph.copy()

        mapping_dict = {"." + node: node for node in self.err_graph.nodes}
        full_graph.add_edges_from([(u, v) for u, v in mapping_dict.items()])
        for u, v in self.err_graph.edges:
            cov_node = ".." + "".join(sorted([u, v]))
            full_graph.add_edges_from([(cov_node, "." + u), (cov_node, "." + v)])

        return full_graph

    def get_scaling_indicators(self):
        """
        Returns a scaling indicator for each of the latent variables in the model.
        The scaling indicator is chosen randomly among the observed measurement
        variables of the latent variable.

        Examples
        --------
        >>> from pgmpy.models import SEMGraph
        >>> model = SEMGraph(ebunch=[('xi1', 'eta1'), ('xi1', 'x1'), ('xi1', 'x2'),
        ...                          ('eta1', 'y1'), ('eta1', 'y2')],
        ...                  latents=['xi1', 'eta1'])
        >>> model.get_scaling_indicators()
        {'xi1': 'x1', 'eta1': 'y1'}

        Returns
        -------
        dict: Returns a dict with latent variables as the key and their value being the
                scaling indicator.
        """
        scaling_indicators = {}
        for node in self.latents:
            for neighbor in self.graph.neighbors(node):
                if neighbor in self.observed:
                    scaling_indicators[node] = neighbor
                    break
        return scaling_indicators

    def active_trail_nodes(self, variables, observed=[], avoid_nodes=[], struct="full"):
        """
        Finds all the observed variables which are d-connected to `variables` in the `graph_struct`
        when `observed` variables are observed.

        Parameters
        ----------
        variables: str or array like
            Observed variables whose d-connected variables are to be found.

        observed : list/array-like
            If given the active trails would be computed assuming these nodes to be observed.

        avoid_nodes: list/array-like
            If specificed, the algorithm doesn't account for paths that have influence flowing
            through the avoid node.

        struct: str or nx.DiGraph instance
            If "full", considers correlation between error terms for computing d-connection.
            If "non_error", doesn't condised error correlations for computing d-connection.
            If instance of nx.DiGraph, finds d-connected variables on the given graph.

        Examples
        --------
        >>> from pgmpy.models import SEM
        >>> model = SEMGraph(ebunch=[('yrsmill', 'unionsen'), ('age', 'laboract'),
        ...                          ('age', 'deferenc'), ('deferenc', 'laboract'),
        ...                          ('deferenc', 'unionsen'), ('laboract', 'unionsen')],
        ...                  latents=[],
        ...                  err_corr=[('yrsmill', 'age')])
        >>> model.active_trail_nodes('age')

        Returns
        -------
        dict: {str: list}
            Returns a dict with `variables` as the key and a list of d-connected variables as the
            value.

        References
        ----------
        Details of the algorithm can be found in 'Probabilistic Graphical Model
        Principles and Techniques' - Koller and Friedman
        Page 75 Algorithm 3.1
        """
        if struct == "full":
            graph_struct = self.full_graph_struct
        elif struct == "non_error":
            graph_struct = self.graph
        elif isinstance(struct, nx.DiGraph):
            graph_struct = struct
        else:
            raise ValueError(
                f"Expected struct to be str or nx.DiGraph. Got {type(struct)}"
            )

        ancestors_list = set()
        for node in observed:
            ancestors_list = ancestors_list.union(
                nx.algorithms.dag.ancestors(graph_struct, node)
            )

        # Direction of flow of information
        # up ->  from parent to child
        # down -> from child to parent

        active_trails = {}
        for start in variables if isinstance(variables, (list, tuple)) else [variables]:
            visit_list = set()
            visit_list.add((start, "up"))
            traversed_list = set()
            active_nodes = set()
            while visit_list:
                node, direction = visit_list.pop()
                if node in avoid_nodes:
                    continue
                if (node, direction) not in traversed_list:
                    if (
                        (node not in observed)
                        and (not node.startswith("."))
                        and (node not in self.latents)
                    ):
                        active_nodes.add(node)
                    traversed_list.add((node, direction))
                    if direction == "up" and node not in observed:
                        for parent in graph_struct.predecessors(node):
                            visit_list.add((parent, "up"))
                        for child in graph_struct.successors(node):
                            visit_list.add((child, "down"))
                    elif direction == "down":
                        if node not in observed:
                            for child in graph_struct.successors(node):
                                visit_list.add((child, "down"))
                        if node in ancestors_list:
                            for parent in graph_struct.predecessors(node):
                                visit_list.add((parent, "up"))
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

        scaling_indicators: dict
            Scaling indicator for each latent variable in the model.

        Returns
        -------
        nx.DiGraph: The transformed full graph structure.

        Examples
        --------
        >>> from pgmpy.models import SEMGraph
        >>> model = SEMGraph(ebunch=[('xi1', 'eta1'), ('xi1', 'x1'), ('xi1', 'x2'),
        ...                          ('eta1', 'y1'), ('eta1', 'y2')],
        ...                  latents=['xi1', 'eta1'])
        >>> model._iv_transformations('xi1', 'eta1',
        ...                           scaling_indicators={'xi1': 'x1', 'eta1': 'y1'})
        """
        full_graph = self.full_graph_struct.copy()

        if not (X, Y) in full_graph.edges():
            raise ValueError(f"The edge from {X} -> {Y} doesn't exist in the graph")

        if (X in self.observed) and (Y in self.observed):
            full_graph.remove_edge(X, Y)
            return full_graph, Y

        elif Y in self.latents:
            full_graph.add_edge("." + Y, scaling_indicators[Y])
            dependent_var = scaling_indicators[Y]
        else:
            dependent_var = Y

        for parent_y in self.graph.predecessors(Y):
            # Remove edge even when the parent is observed ????
            full_graph.remove_edge(parent_y, Y)
            if parent_y in self.latents:
                full_graph.add_edge("." + scaling_indicators[parent_y], dependent_var)

        return full_graph, dependent_var

    def get_ivs(self, X, Y, scaling_indicators={}):
        """
        Returns the Instrumental variables(IVs) for the relation X -> Y

        Parameters
        ----------
        X: node
            The variable name (observed or latent)

        Y: node
            The variable name (observed or latent)

        scaling_indicators: dict (optional)
            A dict representing which observed variable to use as scaling indicator for
            the latent variables.
            If not given the method automatically selects one of the measurement variables
            at random as the scaling indicator.

        Returns
        -------
        set: {str}
            The set of Instrumental Variables for X -> Y.

        Examples
        --------
        >>> from pgmpy.models import SEMGraph
        >>> model = SEMGraph(ebunch=[('I', 'X'), ('X', 'Y')],
        ...                  latents=[],
        ...                  err_corr=[('X', 'Y')])
        >>> model.get_ivs('X', 'Y')
        {'I'}
        """
        if not scaling_indicators:
            scaling_indicators = self.get_scaling_indicators()

        if (X in scaling_indicators.keys()) and (scaling_indicators[X] == Y):
            logger.warning(
                f"{Y} is the scaling indicator of {X}. Please specify `scaling_indicators`"
            )

        transformed_graph, dependent_var = self._iv_transformations(
            X, Y, scaling_indicators=scaling_indicators
        )
        if X in self.latents:
            explanatory_var = scaling_indicators[X]
        else:
            explanatory_var = X

        d_connected_x = self.active_trail_nodes(
            [explanatory_var], struct=transformed_graph
        )[explanatory_var]

        # Condition on X to block any paths going through X.
        d_connected_y = self.active_trail_nodes(
            [dependent_var], avoid_nodes=[explanatory_var], struct=transformed_graph
        )[dependent_var]

        # Remove {X, Y} because they can't be IV for X -> Y
        return d_connected_x - d_connected_y - {dependent_var, explanatory_var}

    def moralize(self, graph="full"):
        """
        TODO: This needs to go to a parent class.
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
        if graph == "full":
            graph = self.full_graph_struct
        elif isinstance(graph, nx.DiGraph):
            graph = graph
        else:
            graph = self.graph

        moral_graph = graph.to_undirected()

        for node in graph.nodes():
            moral_graph.add_edges_from(
                itertools.combinations(graph.predecessors(node), 2)
            )

        return moral_graph

    def _nearest_separator(self, G, Y, Z):
        """
        Finds the set of the nearest separators for `Y` and `Z` in `G`.

        Parameters
        ----------
        G: nx.DiGraph instance
            The graph in which to the find the nearest separation for `Y` and `Z`.

        Y: str
            The variable name for which the separators are needed.

        Z: str
            The other variable for which the separators are needed.

        Returns
        -------
        set or None: If there is a nearest separator returns the set of separators else returns None.
        """
        W = set()
        ancestral_G = G.subgraph(
            nx.ancestors(G, Y).union(nx.ancestors(G, Z)).union({Y, Z})
        ).copy()

        # Optimization: Remove all error nodes which don't have any correlation as it doesn't add any new path. If not removed it can create a lot of
        # extra paths resulting in a much higher runtime.
        err_nodes_to_remove = set(self.err_graph.nodes()) - set(
            [node for edge in self.err_graph.edges() for node in edge]
        )
        ancestral_G.remove_nodes_from(["." + node for node in err_nodes_to_remove])

        M = self.moralize(graph=ancestral_G)
        visited = set([Y])
        to_visit = list(M.neighbors(Y))

        # Another optimization over the original algo. Rather than going through all the paths does
        # a DFS search to find a markov blanket of observed variables. This doesn't ensure minimal observed
        # set.
        while to_visit:
            node = to_visit.pop()
            if node == Z:
                return None
            visited.add(node)
            if node in self.observed:
                W.add(node)
            else:
                to_visit.extend(
                    [node for node in M.neighbors(node) if node not in visited]
                )
        # for path in nx.all_simple_paths(M, Y, Z):
        #     path_set = set(path)
        #     if (len(path) >= 3) and not (W & path_set):
        #         for index in range(1, len(path)-1):
        #             if path[index] in self.observed:
        #                 W.add(path[index])
        #                 break
        if Y not in self.active_trail_nodes([Z], observed=W, struct=ancestral_G)[Z]:
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

        scaling_indicators: dict (optional)
            A dict representing which observed variable to use as scaling indicator for
            the latent variables.
            If not provided, automatically finds scaling indicators by randomly selecting
            one of the measurement variables of each latent variable.

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
        >>> from pgmpy.models import SEMGraph
        >>> model = SEMGraph(ebunch=[('I', 'X'), ('X', 'Y'), ('W', 'I')],
        ...                  latents=[],
        ...                  err_corr=[('W', 'Y')])
        >>> model.get_ivs('X', 'Y')
        [('I', {'W'})]
        """
        if not scaling_indicators:
            scaling_indicators = self.get_scaling_indicators()

        if (X in scaling_indicators.keys()) and (scaling_indicators[X] == Y):
            logger.warning(
                f"{Y} is the scaling indicator of {X}. Please specify `scaling_indicators`"
            )

        transformed_graph, dependent_var = self._iv_transformations(
            X, Y, scaling_indicators=scaling_indicators
        )
        if (X, Y) in transformed_graph.edges:
            G_c = transformed_graph.remove_edge(X, Y)
        else:
            G_c = transformed_graph

        instruments = []
        for Z in self.observed - {X, Y}:
            W = self._nearest_separator(G_c, Y, Z)
            # Condition to check if W d-separates Y from Z
            if (not W) or (W.intersection(descendants(G_c, Y))) or (X in W):
                continue

            # Condition to check if X d-connected to I after conditioning on W.
            elif X in self.active_trail_nodes([Z], observed=W, struct=G_c)[Z]:
                instruments.append((Z, W))
            else:
                continue
        return instruments

    def to_lisrel(self):
        """
        Converts the model from a graphical representation to an equivalent algebraic
        representation. This converts the model into a Reticular Action Model (RAM) model
        representation which is implemented by `pgmpy.models.SEMAlg` class.

        Returns
        -------
        SEMAlg instance: Instance of `SEMAlg` representing the model.

        Examples
        --------
        >>> from pgmpy.models import SEM
        >>> sem = SEM.from_graph(ebunch=[('deferenc', 'unionsen'), ('laboract', 'unionsen'),
        ...                              ('yrsmill', 'unionsen'), ('age', 'deferenc'),
        ...                              ('age', 'laboract'), ('deferenc', 'laboract')],
        ...                      latents=[],
        ...                      err_corr=[('yrsmill', 'age')],
        ...                      err_var={})
        >>> sem.to_lisrel()
        # TODO: Complete this.

        See Also
        --------
        to_standard_lisrel: Converts to the standard lisrel format and returns the parameters.
        """
        nodelist = list(self.observed) + list(self.latents)
        graph_adj = nx.to_numpy_array(self.graph, nodelist=nodelist, weight=None)
        graph_fixed = nx.to_numpy_array(self.graph, nodelist=nodelist, weight="weight")

        err_adj = nx.to_numpy_array(self.err_graph, nodelist=nodelist, weight=None)
        np.fill_diagonal(err_adj, 1.0)  # Variance exists for each error term.
        err_fixed = nx.to_numpy_array(
            self.err_graph, nodelist=nodelist, weight="weight"
        )

        # Add the variance of the error terms.
        for index, node in enumerate(nodelist):
            try:
                err_fixed[index, index] = self.err_graph.nodes[node]["weight"]
            except KeyError:
                err_fixed[index, index] = 0.0

        wedge_y = np.zeros((len(self.observed), len(nodelist)), dtype=int)
        for index, obs_var in enumerate(self.observed):
            wedge_y[index][nodelist.index(obs_var)] = 1.0

        from pgmpy.models import SEMAlg

        return SEMAlg(
            eta=nodelist,
            B=graph_adj.T,
            zeta=err_adj.T,
            wedge_y=wedge_y,
            fixed_values={"B": graph_fixed.T, "zeta": err_fixed.T},
        )

    @staticmethod
    def __standard_lisrel_masks(graph, err_graph, weight, var):
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
        np.ndarray: Adjacency matrix of model's graph structure.

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
        # Arrange the adjacency matrix in order y, x, eta, xi and then slice masks from it.
        #       y(p)   x(q)   eta(m)  xi(n)
        # y
        # x
        # eta \wedge_y          B
        # xi         \wedge_x \Gamma
        #
        # But here we are slicing from the transpose of adjacency because we want incoming
        # edges instead of outgoing because parameters come before variables in equations.
        #
        #       y(p)   x(q)   eta(m)  xi(n)
        # y                  \wedge_y
        # x                          \wedge_x
        # eta                   B    \Gamma
        # xi
        y_vars, x_vars, eta_vars, xi_vars = var["y"], var["x"], var["eta"], var["xi"]

        p, q, m, n = (len(y_vars), len(x_vars), len(eta_vars), len(xi_vars))

        nodelist = y_vars + x_vars + eta_vars + xi_vars
        adj_matrix = nx.to_numpy_array(graph, nodelist=nodelist, weight=weight).T

        B_mask = adj_matrix[p + q : p + q + m, p + q : p + q + m]
        gamma_mask = adj_matrix[p + q : p + q + m, p + q + m :]
        wedge_y_mask = adj_matrix[0:p, p + q : p + q + m]
        wedge_x_mask = adj_matrix[p : p + q, p + q + m :]

        err_nodelist = y_vars + x_vars + eta_vars + xi_vars
        err_adj_matrix = nx.to_numpy_array(
            err_graph, nodelist=err_nodelist, weight=weight
        )

        if not weight == "weight":
            np.fill_diagonal(err_adj_matrix, 1.0)

        theta_e_mask = err_adj_matrix[:p, :p]
        theta_del_mask = err_adj_matrix[p : p + q, p : p + q]
        psi_mask = err_adj_matrix[p + q : p + q + m, p + q : p + q + m]
        phi_mask = err_adj_matrix[p + q + m :, p + q + m :]

        return {
            "B": B_mask,
            "gamma": gamma_mask,
            "wedge_y": wedge_y_mask,
            "wedge_x": wedge_x_mask,
            "phi": phi_mask,
            "theta_e": theta_e_mask,
            "theta_del": theta_del_mask,
            "psi": psi_mask,
        }

    def to_standard_lisrel(self):
        r"""
        Transforms the model to the standard LISREL representation of latent and measurement
        equations. The standard LISREL representation is given as:

        ..math::
            \mathbf{\eta} = \mathbf{B \eta} + \mathbf{\Gamma \xi} + \mathbf{\zeta} \\
            \mathbf{y} = \mathbf{\wedge_y \eta} + \mathbf{\epsilon} \\
            \mathbf{x} = \mathbf{\wedge_x \xi} + \mathbf{\delta} \\
            \mathbf{\Theta_e} = COV(\mathbf{\epsilon}) \\
            \mathbf{\Theta_\delta} = COV(\mathbf{\delta}) \\
            \mathbf{\Psi} = COV(\mathbf{\eta}) \\
            \mathbf{\Phi} = COV(\mathbf{\xi}) \\

        Since the standard LISREL representation has restrictions on the types of model,
        this method adds extra latent variables with fixed loadings of `1` to make the model
        consistent with the restrictions.

        Returns
        -------
        var_names: dict (keys: eta, xi, y, x)
            Returns the variable names in :math:`\mathbf{\eta}`, :math:`\mathbf{\xi}`,
            :math:`\mathbf{y}`, :math:`\mathbf{x}`.

        params: dict (keys: B, gamma, wedge_y, wedge_x, theta_e, theta_del, phi, psi)
            Returns a boolean matrix for each of the parameters. A 1 in the matrix
            represents that there is an edge in the model, 0 represents there is no edge.

        fixed_values: dict (keys: B, gamma, wedge_y, wedge_x, theta_e, theta_del, phi, psi)
            Returns a matrix for each of the parameters. A value in the matrix represents the
            set value for the parameter in the model else it is 0.

        See Also
        --------
        to_lisrel: Converts the model to `pgmpy.models.SEMAlg` instance.

        Examples
        --------
        TODO: Finish this.
        """
        lisrel_err_graph = self.err_graph.copy()
        lisrel_latents = self.latents.copy()
        lisrel_observed = self.observed.copy()

        # Add new latent nodes to convert it to LISREL format.
        mapping = {}
        for u, v in self.graph.edges:
            if (u not in self.latents) and (v in self.latents):
                mapping[u] = "_l_" + u
            elif (u not in self.latents) and (v not in self.latents):
                mapping[u] = "_l_" + u
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

        x = set()
        y = set()
        for exo in xi:
            x.update(
                [x for x in lisrel_graph.neighbors(exo) if x not in lisrel_latents]
            )
        for endo in eta:
            y.update(
                [y for y in lisrel_graph.neighbors(endo) if y not in lisrel_latents]
            )

        # If some node has edges from both eta and xi, replace it with another latent variable
        # otherwise it won't get included in any of the matrices.
        # TODO: Patchy work. Find a better solution.
        common_elements = set(x).intersection(set(y))
        if common_elements:
            mapping = {}
            for var in common_elements:
                mapping[var] = "_l_" + var
            lisrel_graph = nx.relabel_nodes(lisrel_graph, mapping, copy=True)
            for v, u in mapping.items():
                lisrel_graph.add_edge(u, v, weight=1.0)
            eta.extend(mapping.values())
            x = list(set(x) - common_elements)
            y.update(common_elements)

        var_names = {"eta": eta, "xi": xi, "y": list(y), "x": list(x)}
        edges_masks = self.__standard_lisrel_masks(
            graph=lisrel_graph, err_graph=lisrel_err_graph, weight=None, var=var_names
        )
        fixed_masks = self.__standard_lisrel_masks(
            graph=lisrel_graph,
            err_graph=lisrel_err_graph,
            weight="weight",
            var=var_names,
        )
        return (var_names, edges_masks, fixed_masks)


class SEMAlg:
    """
    Base class for algebraic representation of Structural Equation Models(SEMs). The model is
    represented using the Reticular Action Model (RAM).
    """

    def __init__(self, eta=None, B=None, zeta=None, wedge_y=None, fixed_values=None):
        r"""
        Initializes SEMAlg model. The model is represented using the Reticular Action Model(RAM)
        which is given as:
        ..math::
            \mathbf{\eta} = \mathbf{B \eta} + \mathbf{\zeta}
            \mathbf{y} = \mathbf{\wedge_y \eta}

        where :math:`\mathbf{\eta}` is the set of all the observed and latent variables in the
        model, :math:`\mathbf{y}` are the set of observed variables, :math:`\mathbf{\zeta}` is
        the error terms for :math:`\mathbf{\eta}`, and \mathbf{\wedge_y} is a boolean array to
        select the observed variables from :math:`\mathbf{\eta}`.

        Parameters
        ----------
        The following set of parameters are used to set the learnable parameters in the model.
        To specify the values of the parameter use the `fixed_values` parameter. Either `eta`,
        `B`, `zeta`, and `wedge_y`, or `fixed_values` need to be specified.

        eta: list/array-like
            The name of the variables in the model.

        B: 2-D array (boolean)
            The learnable parameters in the `B` matrix.

        zeta: 2-D array (boolean)
            The learnable parameters in the covariance matrix of the error terms.

        wedge_y: 2-D array
            The `wedge_y` matrix.

        fixed_params: dict (default: None)
            A dict of fixed values for parameters.

            If None all the parameters specified by `B`, and `zeta` are learnable.

        Returns
        -------
        pgmpy.models.SEMAlg instance: An instance of the object with initalized values.

        Examples
        --------
        >>> from pgmpy.models import SEMAlg
        # TODO: Finish this example
        """
        self.eta = eta
        self.B = np.array(B)
        self.zeta = np.array(zeta)
        self.wedge_y = wedge_y

        # Get the observed variables
        self.y = []
        for row_i in range(self.wedge_y.shape[0]):
            for index, val in enumerate(self.wedge_y[row_i]):
                if val:
                    self.y.append(self.eta[index])

        if fixed_values:
            self.B_fixed_mask = fixed_values["B"]
            self.zeta_fixed_mask = fixed_values["zeta"]
        else:
            self.B_fixed_mask = np.zeros(self.B.shape)
            self.zeta_fixed_mask = np.zeros(self.zeta.shape)

        # Masks represent the parameters which need to be learnt while training.
        self.B_mask = np.multiply(np.where(self.B_fixed_mask != 0, 0.0, 1.0), self.B)
        self.zeta_mask = np.multiply(
            np.where(self.zeta_fixed_mask != 0, 0.0, 1.0), self.zeta
        )

    def to_SEMGraph(self):
        """
        Creates a graph structure from the LISREL representation.

        Returns
        -------
        pgmpy.models.SEMGraph instance: A path model of the model.

        Examples
        --------
        >>> from pgmpy.models import SEMAlg
        >>> model = SEMAlg()
        # TODO: Finish this example
        """

        err_var = {var: np.diag(self.zeta)[i] for i, var in enumerate(self.eta)}
        graph = nx.relabel_nodes(
            nx.from_numpy_array(self.B.T, create_using=nx.DiGraph),
            mapping={i: self.eta[i] for i in range(self.B.shape[0])},
        )
        # Fill zeta diagonal with 0's as they represent variance and would add self loops in the graph.
        zeta = self.zeta.copy()
        np.fill_diagonal(zeta, 0)
        err_graph = nx.relabel_nodes(
            nx.from_numpy_array(zeta.T, create_using=nx.Graph),
            mapping={i: self.eta[i] for i in range(self.zeta.shape[0])},
        )

        latents = set(self.eta) - set(self.y)

        from pgmpy.models import SEMGraph

        # TODO: Add edge weights
        sem_graph = SEMGraph(
            ebunch=graph.edges(),
            latents=latents,
            err_corr=err_graph.edges(),
            err_var=err_var,
        )
        return sem_graph

    def set_params(self, B, zeta):
        """
        Sets the fixed parameters of the model.

        Parameters
        ----------
        B: 2D array
            The B matrix.

        zeta: 2D array
            The covariance matrix.
        """
        self.B_fixed_mask = B
        self.zeta_fixed_mask = zeta

    def generate_samples(self, n_samples=100):
        """
        Generates random samples from the model.

        Parameters
        ----------
        n_samples: int
            The number of samples to generate.

        Returns
        -------
        pd.DataFrame: The generated samples.
        """
        if (self.B_fixed_mask is None) or (self.zeta_fixed_mask is None):
            raise ValueError("Parameters for the model has not been specified.")

        B_inv = np.linalg.inv(np.eye(self.B_fixed_mask.shape[0]) - self.B_fixed_mask)
        implied_cov = (
            self.wedge_y @ B_inv @ self.zeta_fixed_mask @ B_inv.T @ self.wedge_y.T
        )

        # Check if implied covariance matrix is positive definite.
        if not np.all(np.linalg.eigvals(implied_cov) > 0):
            raise ValueError(
                "The implied covariance matrix is not positive definite."
                + "Please check model parameters."
            )

        # Get the order of observed variables
        x_index, y_index = np.nonzero(self.wedge_y)
        observed = [self.eta[i] for i in y_index]

        # Generate samples and return a dataframe.
        samples = np.random.multivariate_normal(
            mean=[0 for i in range(implied_cov.shape[0])],
            cov=implied_cov,
            size=n_samples,
        )
        return pd.DataFrame(samples, columns=observed)


class SEM(SEMGraph):
    """
    Class for representing Structural Equation Models. This class is a wrapper over
    `SEMGraph` and `SEMAlg` to provide a consistent API over the different representations.

    Attributes
    ----------
    model: SEMGraph instance
        A graphical representation of the model.
    """

    def __init__(self, syntax, **kwargs):
        """
        Initialize a `SEM` object. Preferred way to initialize the object is to use one of
        the `from_lavaan`, `from_graph`, `from_lisrel`, or `from_RAM` methods.

        There are three possible ways to initialize the model:
            1. Lavaan syntax: `lavaan_str` needs to be specified.
            2. Graph structure: `ebunch`, `latents`, `err_corr`, and `err_var` need to be specified.
            3. LISREL syntax: `var_names`, `params`, and `fixed_masks` need to be specified.
            4. Reticular Action Model (RAM/all-y) syntax: `var_names`, `B`, `zeta`, and `wedge_y`
                                                            need to be specified.

        Parameters
        ----------
        syntax: str (lavaan|graph|lisrel|ram)
            The syntax used to initialize the model.

        kwargs:
            For parameter details, check docstrings for `from_lavaan`, `from_graph`, `from_lisrel`,
            and `from_RAM` methods.

        See Also
        --------
        from_lavaan: Initialize a model using lavaan syntax.
        from_graph: Initialize a model using graph structure.
        from_lisrel: Initialize a model using LISREL syntax.
        from_RAM: Initialize a model using Reticular Action Model(RAM/all-y) syntax.
        """
        if syntax.lower() == "lavaan":
            # Create a SEMGraph model using the lavaan str.

            # Step 1: Define the grammar for each type of string.
            var = Word(alphanums)
            reg_gram = (
                OneOrMore(
                    var.setResultsName("predictors", listAllMatches=True)
                    + Optional(Suppress("+"))
                )
                + "~"
                + OneOrMore(
                    var.setResultsName("covariates", listAllMatches=True)
                    + Optional(Suppress("+"))
                )
            )
            intercept_gram = var("inter_var") + "~" + Word("1")
            covar_gram = (
                var("covar_var1")
                + "~~"
                + OneOrMore(
                    var.setResultsName("covar_var2", listAllMatches=True)
                    + Optional(Suppress("+"))
                )
            )
            latent_gram = (
                var("latent")
                + "=~"
                + OneOrMore(
                    var.setResultsName("obs", listAllMatches=True)
                    + Optional(Suppress("+"))
                )
            )

            # Step 2: Preprocess string to lines
            lines = kwargs["lavaan_str"]

            # Step 3: Initialize arguments and fill them by parsing each line.
            ebunch = []
            latents = []
            err_corr = []
            err_var = []
            for line in lines:
                line = line.strip()
                if (line != "") and (not line.startswith("#")):
                    if intercept_gram.matches(line):
                        continue
                    elif reg_gram.matches(line):
                        results = reg_gram.parseString(line, parseAll=True)
                        for pred in results["predictors"]:
                            ebunch.extend(
                                [
                                    (covariate, pred)
                                    for covariate in results["covariates"]
                                ]
                            )
                    elif covar_gram.matches(line):
                        results = covar_gram.parseString(line, parseAll=True)
                        for var in results["covar_var2"]:
                            err_corr.append((results["covar_var1"], var))

                    elif latent_gram.matches(line):
                        results = latent_gram.parseString(line, parseAll=True)
                        latents.append(results["latent"])
                        ebunch.extend(
                            [(results["latent"], obs) for obs in results["obs"]]
                        )

            # Step 4: Call the parent __init__ with the arguments
            super(SEM, self).__init__(ebunch=ebunch, latents=latents, err_corr=err_corr)

        elif syntax.lower() == "graph":
            super(SEM, self).__init__(
                ebunch=kwargs["ebunch"],
                latents=kwargs["latents"],
                err_corr=kwargs["err_corr"],
                err_var=kwargs["err_var"],
            )

        elif syntax.lower() == "lisrel":
            model = SEMAlg(
                var_names=var_names, params=params, fixed_masks=fixed_masks
            ).to_SEMGraph()
            # Initialize an empty SEMGraph instance and set the properties.
            # TODO: Boilerplate code, find a better way to do this.
            super(SEM, self).__init__(ebunch=[], latents=[], err_corr=[], err_var={})
            self.graph = model.graph
            self.latents = model.latents
            self.obseved = model.observed
            self.err_graph = model.err_graph
            self.full_graph_struct = model.full_graph_struct

        elif syntax.lower() == "ram":
            model = SEMAlg(
                eta=kwargs["var_names"],
                B=kwargs["B"],
                zeta=kwargs["zeta"],
                wedge_y=kwargs["wedge_y"],
                fixed_values=fixed_masks,
            )

    @classmethod
    def from_lavaan(cls, string=None, filename=None):
        """
        Initializes a `SEM` instance using lavaan syntax.

        Parameters
        ----------
        string: str (default: None)
            A `lavaan` style multiline set of regression equation representing the model.
            Refer http://lavaan.ugent.be/tutorial/syntax1.html for details.

        filename: str (default: None)
            The filename of the file containing the model in lavaan syntax.

        Examples
        --------
        """
        if filename:
            with open(filename, "r") as f:
                lavaan_str = f.readlines()
        elif string:
            lavaan_str = string.split("\n")
        else:
            raise ValueError("Either `filename` or `string` need to be specified")

        return cls(syntax="lavaan", lavaan_str=lavaan_str)

    @classmethod
    def from_graph(cls, ebunch, latents=[], err_corr=[], err_var={}):
        """
        Initializes a `SEM` instance using graphical structure.

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
        >>> from pgmpy.models import SEM
        >>> sem = SEM.from_graph(ebunch=[('deferenc', 'unionsen'), ('laboract', 'unionsen'),
        ...                              ('yrsmill', 'unionsen'), ('age', 'deferenc'),
        ...                              ('age', 'laboract'), ('deferenc', 'laboract')],
        ...                      latents=[],
        ...                      err_corr=[('yrsmill', 'age')],
        ...                      err_var={})

        Defining a model (Education [2]) with all the parameters set. For not setting any
        parameter `np.nan` can be explicitly passed.
        >>> sem_edu = SEM.from_graph(ebunch=[('intelligence', 'academic', 0.8), ('intelligence', 'scale_1', 0.7),
        ...                                  ('intelligence', 'scale_2', 0.64), ('intelligence', 'scale_3', 0.73),
        ...                                  ('intelligence', 'scale_4', 0.82), ('academic', 'SAT_score', 0.98),
        ...                                  ('academic', 'High_school_gpa', 0.75), ('academic', 'ACT_score', 0.87)],
        ...                          latents=['intelligence', 'academic'],
        ...                          err_corr=[],
        ...                          err_var={})

        References
        ----------
        [1] McDonald, A, J., & Clelland, D. A. (1984). Textile Workers and Union Sentiment.
            Social Forces, 63(2), 502–521
        [2] https://en.wikipedia.org/wiki/Structural_equation_modeling#/
            media/File:Example_Structural_equation_model.svg
        """
        return cls(
            syntax="graph",
            ebunch=ebunch,
            latents=latents,
            err_corr=err_corr,
            err_var=err_var,
        )

    @classmethod
    def from_lisrel(cls, var_names, params, fixed_masks=None):
        r"""
        Initializes a `SEM` instance using LISREL notation. The LISREL notation is defined as:
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
        pgmpy.models.SEM instance: An instance of the object with initalized values.

        Examples
        --------
        >>> from pgmpy.models import SEMAlg
        # TODO: Finish this example
        """
        eta = var_names["y"] + var_names["x"] + var_names["eta"] + var_names["xi"]
        m, n, p, q = (
            len(var_names["y"]),
            len(var_names["x"]),
            len(var_names["eta"]),
            len(var_names["xi"]),
        )

        B = np.block(
            [
                [np.zeros((m, m + n)), params["wedge_y"], np.zeros((m, q))],
                [np.zeros((n, m + n + p)), params["wedge_x"]],
                [np.zeros((p, m + n)), params["B"], params["gamma"]],
                [np.zeros((q, m + n + p + q))],
            ]
        )
        zeta = np.block(
            [
                [params["theta_e"], np.zeros((m, n + p + q))],
                [np.zeros((n, m)), params["theta_del"], np.zeros((n, p + q))],
                [np.zeros((p, m + n)), params["psi"], np.zeros((p, q))],
                [np.zeros((q, m + n + p)), params["phi"]],
            ]
        )

        B = np.block(
            [
                [np.zeros((m, m + n)), fixed_params["wedge_y"], np.zeros((m, q))],
                [np.zeros((n, m + n + p)), fixed_params["wedge_x"]],
                [np.zeros((p, m + n)), fixed_params["B"], fixed_params["gamma"]],
                [np.zeros((q, m + n + p + q))],
            ]
        )
        zeta = np.block(
            [
                [fixed_params["theta_e"], np.zeros((m, n + p + q))],
                [np.zeros((n, m)), fixed_params["theta_del"], np.zeros((n, p + q))],
                [np.zeros((p, m + n)), fixed_params["psi"], np.zeros((p, q))],
                [np.zeros((q, m + n + p)), fixed_params["phi"]],
            ]
        )
        observed = var_names["y"] + var_names["x"]

        return cls.from_RAM(
            variables=eta,
            B=B,
            zeta=zeta,
            observed=observed,
            fixed_values={"B": B, "zeta": zeta},
        )

    @classmethod
    def from_RAM(
        cls, variables, B, zeta, observed=None, wedge_y=None, fixed_values=None
    ):
        r"""
        Initializes a `SEM` instance using Reticular Action Model(RAM) notation. The model
        is defined as:

        ..math::

            \mathbf{\eta} = \mathbf{B \eta} + \mathbf{\epsilon} \\
            \mathbf{\y} = \wedge_y \mathbf{\eta}
            \zeta = COV(\mathbf{\epsilon})

        where :math:`\mathbf{\eta}` is the set of variables (both latent and observed),
        :math:`\mathbf{\epsilon}` are the error terms, :math:`\mathbf{y}` is the set
        of observed variables, :math:`\wedge_y` is a boolean array of the shape (no of
        observed variables, no of total variables).

        Parameters
        ----------
        variables: list, array-like
            List of variables (both latent and observed) in the model.

        B: 2-D boolean array (shape: `len(variables)` x `len(variables)`)
            The non-zero parameters in :math:`B` matrix. Refer model definition in docstring for details.

        zeta: 2-D boolean array (shape: `len(variables)` x `len(variables)`)
            The non-zero parameters in :math:`\zeta` (error covariance) matrix. Refer model definition
            in docstring for details.

        observed: list, array-like (optional: Either `observed` or `wedge_y` needs to be specified)
            List of observed variables in the model.

        wedge_y: 2-D array (shape: no. observed x total vars) (optional: Either `observed` or `wedge_y`)
            The :math:`\wedge_y` matrix. Refer model definition in docstring for details.

        fixed_values: dict (optional)
            If specified, fixes the parameter values and are not changed during estimation.
            A dict with the keys B, zeta.

        Returns
        -------
        pgmpy.models.SEM instance: An instance of the object with initialized values.

        Examples
        --------
        >>> from pgmpy.models import SEM
        >>> SEM.from_RAM  # TODO: Finish this
        """
        if observed:
            wedge_y = np.zeros((len(variables), len(observed)))
            obs_dict = {var: index for index, var in enumerate(observed)}
            all_dict = {var: index for index, var in enumerate(variables)}
            for var in observed:
                wedge_y[obs_dict[var], all_dict[var]] = 1

        return cls(
            syntax="ram",
            var_names=variables,
            B=B,
            zeta=zeta,
            wedge_y=wedge_y,
            fixed_values=fixed_values,
        )

    def fit(self):
        pass
