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
    they don't need to be specified.

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
        Initializes a `SEMGraph` object.

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
        parameter `np.NaN` can be explicitly passed.
        >>> sem_edu = SEMGraph(ebunch=[('intelligence', 'academic', 0.8), ('intelligence', 'scale_1', 0.7),
        ...                            ('intelligence', 'scale_2', 0.64), ('intelligence', 'scale_3', 0.73),
        ...                            ('intelligence', 'scale_4', 0.82), ('academic', 'SAT_score', 0.98),
        ...                            ('academic', 'High_school_gpa', 0.75), ('academic', 'ACT_score', 0.87)],
        ...                    latents=['intelligence', 'academic'],
        ...                    err_corr=[]
        ...                    err_var={'intelligence': 1})

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

        # Set the error variances
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
            cov_node = '..' + ''.join(sorted([u, v]))
            full_graph.add_edges_from([(cov_node, '.' + u), (cov_node, '.'+ v)])

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

    def active_trail_nodes(self, variables, observed=[], avoid_nodes=[], struct='full'):
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
                            if parent not in avoid_nodes:
                                visit_list.add((parent, 'up'))
                        for child in graph_struct.successors(node):
                            visit_list.add((child, 'down'))
                    elif direction == 'down':
                        if node not in observed:
                            for child in graph_struct.successors(node):
                                visit_list.add((child, 'down'))
                        if node in ancestors_list:
                            for parent in graph_struct.predecessors(node):
                                if parent not in avoid_nodes:
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
            raise ValueError("The edge from {X} -> {Y} doesn't exist in the graph".format(
                                                                                    X=X, Y=Y))

        if (X in self.observed) and (Y in self.observed):
            full_graph.remove_edge(X, Y)
            return full_graph, Y

        elif Y in self.latents:
            full_graph.add_edge('.'+Y, scaling_indicators[Y])
            dependent_var = scaling_indicators[Y]
        else:
            dependent_var = Y

        for parent_y in self.graph.predecessors(Y):
            # Remove edge even when the parent is observed ????
            full_graph.remove_edge(parent_y, Y)
            if parent_y in self.latents:
                full_graph.add_edge('.'+scaling_indicators[parent_y], dependent_var)

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
        ...                  err_corr=['X', 'Y'])
        >>> model.get_ivs('X', 'Y')
        {'I'}
        """
        if not scaling_indicators:
            scaling_indicators = self.get_scaling_indicators()

        transformed_graph, dependent_var = self._iv_transformations(X, Y, scaling_indicators=scaling_indicators)
        if X in self.latents:
            explanatory_var = scaling_indicators[X]
        else:
            explanatory_var = X

        d_connected_x = self.active_trail_nodes([explanatory_var], struct=transformed_graph)[explanatory_var]

        # Condition on X to block any paths going through X.
        d_connected_y = self.active_trail_nodes([dependent_var], avoid_nodes=[explanatory_var],
                                                struct=transformed_graph)[dependent_var]

        # Remove {X, Y} because they can't be IV for X -> Y
        return (d_connected_x - d_connected_y - {dependent_var, explanatory_var})

    def moralize(self, graph='full'):
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
                    if path[index] in self.observed:
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
        transformed_graph, dependent_var = self._iv_transformations(X, Y, scaling_indicators=scaling_indicators)

        if (X, Y) in transformed_graph.edges:
            G_c = transformed_graph.remove_edge(X, Y)
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
        Converts the model from a graphical representation to an equivalent algebraic
        representation.

        Returns
        -------
        SEMLISREL instance: Instance of `SEMLISREL` representing the model.

        Examples
        --------
        """
        nodelist = list(self.observed) + list(self.latents)
        graph_adj = nx.to_numpy_matrix(self.graph, nodelist=nodelist, weight=None)
        graph_fixed = nx.to_numpy_matrix(self.graph, nodelist=nodelist, weight='weight')

        err_adj = nx.to_numpy_matrix(self.err_graph, nodelist=nodelist, weight=None)
        err_fixed = nx.to_numpy_matrix(self.err_graph, nodelist=nodelist, weight='weight')

        wedge_y = np.zeros((len(self.observed), len(nodelist)), dtype=int)
        for index, obs_var in enumerate(self.observed):
            wedge_y[index][nodelist.index(obs_var)] = 1.0

        from pgmpy.models import SEMLISREL
        return SEMLISREL(eta=nodelist, B=graph_adj.T, zeta=err_adj.T, wedge_y=wedge_y,
                         fixed_values={'B': graph_fixed, 'zeta': err_fixed})


class SEMLISREL:
    """
    Base class for algebraic representation of Structural Equation Models(SEMs). The model is
    represented using the Reticular Action Model (RAM).
    """
    def __init__(self, eta=None, B=None, zeta=None, wedge_y=None, fixed_values=None):
        r"""
        Initializes SEMLISREL model. The model is represented using the Reticular Action Model(RAM)
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
        pgmpy.models.SEMLISREL instance: An instance of the object with initalized values.

        Examples
        --------
        >>> from pgmpy.models import SEMLISREL
        # TODO: Finish this example
        """
        self.eta = eta
        self.B = np.array(B)
        self.zeta = np.array(zeta)
        self.wedge_y = wedge_y

        if fixed_values:
            self.B_fixed_mask = fixed_values['B']
            self.zeta_fixed_mask = fixed_values['zeta']
        else:
            self.B_fixed_mask = np.zeros(self.B.shape)
            self.zeta_fixed_mask = np.zeros(self.zeta.shape)

        # Masks represent the parameters which need to be learnt while training.
        self.B_mask = np.multiply(np.where(self.B_fixed_mask != 0, 0.0, 1.0), self.B)
        self.zeta_mask = np.multiply(np.where(self.zeta_fixed_mask != 0, 0.0, 1.0), self.zeta)

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
        graph = nx.relabel_nodes(nx.from_numpy_matrix(self.B.T, create_using=nx.DiGraph),
                                 mapping={i: self.eta[i] for i in range(self.B.shape[0])})
        err_graph = nx.relabel_nodes(nx.from_numpy_matrix(self.zeta.T, create_using=nx.Graph),
                                     mapping={i: self.eta[i] for i in range(self.zeta.shape[0])})

        # Extract observed variables from `eta` using `wedge_y`
        observed = []
        for row_i in range(self.wedge_y.shape[0]):
            for index, val in enumerate(self.wedge_y[row_i]):
                if val:
                    observed.append(self.eta[index])
        latents = set(self.eta) - set(observed)

        from pgmpy.models import SEMGraph
        # TODO: Add edge weights
        sem_graph = SEMGraph(ebunch=graph.edges(),
                             latents=latents,
                             err_corr=err_graph.edges(),
                             err_var={var: np.diag(self.zeta)[i]
                                      for i, var in enumerate(self.eta)})
        return sem_graph

    def set_params(self, B, zeta):
        self.B_fixed_mask = B
        self.zeta_fixed_mask = zeta


class SEM(SEMGraph):
    """
    Class for representing Structural Equation Models. This class is a wrapper over
    `SEMGraph` and `SEMLISREL` to provide a consistent API over the different representations.

    Attributes
    ----------
    model: SEMGraph instance
        A graphical representation of the model.
    """
    def __init__(self, lavaan_str=None, ebunch=[], latents=[], err_corr=[],
                 err_var={}, var_names=None, params=None, fixed_masks=None):
        """
        Initialize a `SEM` object. Prefered way to initialize the object is to use one of
        the `from_lavaan`, `from_graph`, or `from_lisrel` methods.

        There are three possible ways to initialize the model:
            1. Lavaan syntax: `lavaan_str` needs to be specified.
            2. Graph structure: `ebunch`, `latents`, `err_corr`, and `err_var` need to specified.
            3. LISREL syntax: `var_names`, `params`, and `fixed_masks` need to be specified.

        Parameters
        ----------
        For parameter details, check docstrings for `from_lavaan`, `from_graph`, and `from_lisrel`
        methods.

        See Also
        --------
        from_lavaan: Initialize a model using lavaan syntax.
        from_graph: Initialize a model using graph structure.
        from_lisrel: Initialize a model using LISREL syntax.
        """
        if lavaan_str:
            # Create a SEMGraph model using the lavaan str.
            raise NotImplementedError("Lavaan syntax is not supported yet.")
        elif ebunch:
            super(SEM, self).__init__(ebunch=ebunch, latents=latents,
                                      err_corr=err_corr, err_var=err_var)
        elif var_names:
            model = SEMLISREL(var_names=var_names, params=params, fixed_masks=fixed_masks).to_SEMGraph()
            # Initialize an empty SEMGraph instance and set the properties.
            # TODO: Boilerplate code, find a better way to do this.
            super(SEM, self).__init__(ebunch=[], latents=[], err_corr=[], err_var={})
            self.graph = model.graph
            self.latents = model.latents
            self.obseved = model.observed
            self.err_graph = model.err_graph
            self.full_graph_struct = model.full_graph_struct

    @classmethod
    def from_lavaan(cls, lavaan_str):
        """
        Initializes a `SEM` instance using lavaan syntax.

        Parameters
        ----------
        str_model: str (default: None)
            A `lavaan` style multiline set of regression equation representing the model.
            Refer http://lavaan.ugent.be/tutorial/syntax1.html for details.

            If None requires `var_names` and `params` to be specified.

        Examples
        --------
        """
        return cls(lavaan_str=lavaan_str)

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
        parameter `np.NaN` can be explicitly passed.
        >>> sem_edu = SEM.from_graph(ebunch=[('intelligence', 'academic', 0.8), ('intelligence', 'scale_1', 0.7),
        ...                                  ('intelligence', 'scale_2', 0.64), ('intelligence', 'scale_3', 0.73),
        ...                                  ('intelligence', 'scale_4', 0.82), ('academic', 'SAT_score', 0.98),
        ...                                  ('academic', 'High_school_gpa', 0.75), ('academic', 'ACT_score', 0.87)],
        ...                          latents=['intelligence', 'academic'],
        ...                          err_corr=[]
        ...                          err_var={})

        References
        ----------
        [1] McDonald, A, J., & Clelland, D. A. (1984). Textile Workers and Union Sentiment.
            Social Forces, 63(2), 502–521
        [2] https://en.wikipedia.org/wiki/Structural_equation_modeling#/
            media/File:Example_Structural_equation_model.svg
        """
        return cls(ebunch=ebunch, latents=latents, err_corr=err_corr, err_var=err_var)

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
        >>> from pgmpy.models import SEMLISREL
        # TODO: Finish this example
        """
        return cls(var_names=var_names, params=params, fixed_masks=fixed_masks)

    def fit(self):
        pass
