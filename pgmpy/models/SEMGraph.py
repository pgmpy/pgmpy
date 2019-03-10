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
    def __init__(self, ebunch=[], latents=[], err_corr=[]):
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
                2. ((u, u_var), (v, v_var), covar): Adds correlation between the error terms of `u` and `v`.
                           Also sets the variance of `u`'s and `v`'s error term to `u_var` and `v_var`
                           respectively, and the covariance to `covar`. Pass `np.NaN` for any of these values
                           in case it shouldn't be set.

        Examples
        --------
        Defining a model (Union sentiment model[1]) without setting any paramaters.
        >>> from pgmpy.models import SEMGraph
        >>> sem = SEMGraph(ebunch=[('deferenc', 'unionsen'), ('laboract', 'unionsen'),
        ...                        ('yrsmill', 'unionsen'), ('age', 'deferenc'),
        ...                        ('age', 'laboract'), ('deferenc', 'laboract')],
        ...                latents=[],
        ...                err_corr=[('yrsmill', 'age')])

        Defining a model (Education [2]) with all the parameters set. For not setting any
        parameter `np.NaN` can be explicitly passed.
        >>> sem_edu = SEMGraph(ebunch=[('intelligence', 'academic', 0.8), ('intelligence', 'scale_1', 0.7),
        ...                            ('intelligence', 'scale_2', 0.64), ('intelligence', 'scale_3', 0.73),
        ...                            ('intelligence', 'scale_4', 0.82), ('academic', 'SAT_score', 0.98),
        ...                            ('academic', 'High_school_gpa', 0.75), ('academic', 'ACT_score', 0.87)],
        ...                    latents=['intelligence', 'academic'],
        ...                    err_corr=[])

        References
        ----------
        [1] McDonald, A, J., & Clelland, D. A. (1984). Textile Workers and Union Sentiment.
            Social Forces, 63(2), 502–521
        [2] https://en.wikipedia.org/wiki/Structural_equation_modeling#/
            media/File:Example_Structural_equation_model.svg
        """
        super(SEM, self).__init__()

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
        self.observed = set(self.graph.nodes()) - self.latents()

        # Construct the error graph and set the parameters.
        self.err_graph = nx.Graph()
        self.err_graph.add_nodes_from(self.graph.nodes())
        for t in err_corr:
            if len(t) == 2:
                self.err_graph.add_edge(t[0], t[1])
            elif len(t) == 3:
                try:
                    if isinstance(t[0], str) and isinstance(t[1], str):
                        self.err_graph.nodes[t[0][0]]['var'] = np.NaN
                        self.err_graph.nodes[t[1][0]]['var'] = np.NaN
                    else:
                        self.err_graph.nodes[t[0][0]]['var'] = t[0][1]
                        self.err_graph.nodes[t[1][0]]['var'] = t[1][1]

                    self.err_graph.add_edge(t[0][0], t[1][0], weight=t[2])
                except KeyError:
                    raise ValueError("{t} not in expected shape. Please refer to the documentation".format(t=t))
            else:
                raise ValueError("Expected tuple length: 2 or 3. Got {t} of len {shape}".format(
                                                        t=t, shape=len(t)))

        self.full_graph = self._get_full_graph_struct()


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
        for node in observed_list:
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
                    if (node not in observed_list) and (not node.startswith('.')) and (node not in self.latents):
                        active_nodes.add(node)
                    traversed_list.add((node, direction))
                    if direction == 'up' and node not in observed_list:
                        for parent in graph_struct.predecessors(node):
                            visit_list.add((parent, 'up'))
                        for child in graph_struct.successors(node):
                            visit_list.add((child, 'down'))
                    elif direction == 'down':
                        if node not in observed_list:
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
        d_connected = self.active_trail_nodes([X, Y], graph_struct=transformed_graph)
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

    def to_lisrel(self):
        """
        Gets parameters from the graph structure to the standard LISREL matrix representation.

        Returns
        -------
        dict: Dict with the keys B, gamma, wedge_y, wedge_x, theta_e, theta_del, phi and psi.

        Examples
        --------
        """
        eta, m = sorted(self.eta), len(self.eta)
        xi, n = sorted(self.xi), len(self.xi)
        y, p = sorted(self.y), len(self.y)
        x, q = sorted(self.x), len(self.x)

        # Set values in relation matrices.
        B = np.zeros((m, m))
        gamma = np.zeros((m, n))
        wedge_y = np.zeros((p, m))
        wedge_x = np.zeros((q, n))

        for u, v in self.graph.edges:
            if u in eta and v in eta:
                B[eta.index(v), eta.index(u)] = self.graph.edges[u, v]['weight']
            elif u in xi and v in eta:
                gamma[eta.index(v), xi.index(u)] = self.graph.edges[u, v]['weight']
            elif u in xi and v in x:
                wedge_x[x.index(v), xi.index(u)] = self.graph.edges[u, v]['weight']
            elif u in eta and v in y:
                wedge_y[y.index(v), eta.index(u)] = self.graph.edges[u, v]['weight']

        # Set values in covariance matrices.
        psi = np.zeros((m, m))
        phi = np.zeros((n, n))
        theta_e = np.zeros((p, p))
        theta_del = np.zeros((q, q))

        for node in self.err_graph.nodes:
            if node in eta:
                index = eta.index(node)
                psi[index, index] = self.err_graph.nodes[node]['var']
            elif node in xi:
                index = xi.index(node)
                phi[index, index] = self.err_graph.nodes[node]['var']
            elif node in y:
                index = y.index(node)
                theta_e[index, index] = self.err_graph.nodes[node]['var']
            elif node in x:
                index = x.index(node)
                theta_del[index, index] = self.err_graph.nodes[node]['var']

        for u, v in self.err_graph.edges:
            if u in eta and v in eta:
                u_index, v_index = eta.index(u), eta.index(v)
                psi[u_index, v_index] = self.err_graph.edges[u, v]['weight']
                psi[v_index, u_index] = self.err_graph.edges[u, v]['weight']
            elif u in xi and v in xi:
                u_index, v_index = xi.index(u), xi.index(v)
                phi[u_index, v_index] = self.err_graph.edges[u, v]['weight']
                phi[v_index, u_index] = self.err_graph.edges[u, v]['weight']
            elif u in y and v in y:
                u_index, v_index = y.index(u), y.index(v)
                theta_e[u_index, v_index] = self.err_graph.edges[u, v]['weight']
                theta_e[v_index, u_index] = self.err_graph.edges[u, v]['weight']
            elif u in x and v in x:
                u_index, v_index = x.index(u), x.index(v)
                theta_del[u_index, v_index] = self.err_graph.edges[u, v]['weight']
                theta_del[v_index, u_index] = self.err_graph.edges[u, v]['weight']

        return SEMLISREL(B=B, gamma=gamma, wedge_y=wedge_y, wedge_x=wedge_x,
                         psi=psi, phi=phi, theta_e=theta_e, theta_del=theta_del)
