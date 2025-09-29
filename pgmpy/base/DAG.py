import itertools
from os import PathLike
from typing import Hashable, Iterable, Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base._mixin_roles import _GraphRolesMixin
from pgmpy.global_vars import logger
from pgmpy.independencies import Independencies
from pgmpy.utils.parser import parse_dagitty, parse_lavaan


class DAG(_GraphRolesMixin, nx.DiGraph):
    """Directed Graphical Model, graph with vertex roles.

    Each node in the graph can represent either a random variable, ``Factor``,
    or a cluster of random variables. Edges in the graph represent the
    dependencies between these.

    Abstract roles can be assigned to nodes in the graph, such as
    exposure, outcome, adjustment set, etc. These roles are used, or created,
    by algorithms that use the graph, such as causal inference,
    causal discovery, causal prediction.

    Parameters
    ----------
    ebunch : input graph (optional, default: None)
        Data to initialize graph. If None (default) an empty
        graph is created.  The data can be any format that is supported
        by the to_networkx_graph() function, currently including edge list,
        dict of dicts, dict of lists, NetworkX graph, 2D NumPy array, SciPy
        sparse matrix, or PyGraphviz graph.

    latents : set of nodes (default: empty set)
        A set of latent variables in the graph. These are not observed
        variables but are used to represent unobserved confounding or
        other latent structures.

    roles : dict, optional (default: None)
        A dictionary mapping roles to node names.
        The keys are roles, and the values are role names (strings or iterables of str).
        If provided, this will automatically assign roles to the nodes in the graph.
        Passing a key-value pair via ``roles`` is equivalent to calling
        ``with_role(role, variables)`` for each key-value pair in the dictionary.

    Examples
    --------
    Create an empty DAG with no nodes and no edges

    >>> from pgmpy.base import DAG
    >>> G = DAG()

    Edges and vertices can be passed to the constructor as an edge list.

    >>> G = DAG(ebunch=[("a", "b"), ("b", "c")])

    G can be also grown incrementally, in several ways:

    **Nodes:**

    Add one node at a time:

    >>> G.add_node(node="a")

    Add the nodes from any container (a list, set or tuple or the nodes
    from another graph).

    >>> G.add_nodes_from(nodes=["a", "b"])

    **Edges:**

    G can also be grown by adding edges.

    Add one edge,

    >>> G.add_edge(u="a", v="b")

    a list of edges,

    >>> G.add_edges_from(ebunch=[("a", "b"), ("b", "c")])

    If some edges connect nodes not yet in the model, the nodes
    are added automatically. There are no errors when adding
    nodes or edges that already exist.

    **Shortcuts:**

    Many common graph features allow python syntax for speed reporting.

    >>> "a" in G  # check if node in graph
    True
    >>> len(G)  # number of nodes in graph
    3

    Roles can be assigned to nodes in the graph at construction or using methods.

    At construction:

    >>> G = DAG(
    ...     ebunch=[("U", "X"), ("X", "M"), ("M", "Y"), ("U", "Y")],
    ...     roles={"exposure": "X", "outcome": "Y"},
    ... )

    Roles can also be assigned after creation using the ``with_role`` method.

    >>> G = G.with_role("adjustment", {"U", "M"})

    Vertices of a specific role can be retrieved using the ``get_role`` method.

    >>> G.get_role("exposure")
    ['X']
    >>> G.get_role("adjustment")
    ['U', 'M']

    **Latents:**
        Latent variables can be managed using the `latents` parameter at
        initialization or by assigning the "latents" role to nodes. The
        `latents` parameter is a convenient shortcut for `roles={'latents': ...}`.

    Create a graph with initial latent variables 'U' and 'V':

    >>> from pgmpy.base import DAG
    >>> G = DAG(
    ...     ebunch=[("U", "X"), ("X", "M"), ("M", "Y"), ("U", "Y"), ("V", "M")],
    ...     latents={"U", "V"},
    ... )
    >>> sorted(G.latents)
    ['U', 'V']

    Add a new latent variable 'Z' using the role system:

    >>> G.add_node("Z")
    >>> G.with_role(role="latents", variables="Z", inplace=True)
    >>> sorted(G.latents)
    ['U', 'V', 'Z']

    You can also check for latents using the `get_role` method:

    >>> sorted(G.get_role(role="latents"))
    ['U', 'V', 'Z']

    Remove a latent variable from the role:

    >>> G.without_role(role="latents", variables="V", inplace=True)
    >>> sorted(G.latents)
    ['U', 'Z']
    """

    def __init__(
        self,
        ebunch: Optional[Iterable[tuple[Hashable, Hashable]]] = None,
        latents: set[Hashable] = set(),
        roles=None,
    ):
        super().__init__(ebunch)

        self._check_cycles()

        self.latents = set(latents)

        if roles is None:
            roles = {}
        elif not isinstance(roles, dict):
            raise TypeError("Roles must be provided as a dictionary.")

        # set the roles to the vertices as networkx attributes
        for role, vars in roles.items():
            self.with_role(role=role, variables=vars, inplace=True)

    def _check_cycles(self):
        """Checks if the graph has cycles.

        Raises
        ------
        ValueError
            If the graph has cycles.
        """
        cycles = []
        try:
            cycles = list(nx.find_cycle(self))
        except nx.NetworkXNoCycle:
            pass
        else:
            out_str = "Cycles are not allowed in a DAG."
            out_str += "\nEdges indicating the path taken for a loop: "
            out_str += "".join([f"({u},{v}) " for (u, v) in cycles])
            raise ValueError(out_str)

    @classmethod
    def from_lavaan(
        cls,
        string: Optional[str] = None,
        filename: Optional[str | PathLike] = None,
    ) -> "DAG":
        """
        Initializes a `DAG` instance using lavaan syntax.

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
        ebunch, latents, err_corr, _ = parse_lavaan(lavaan_str)
        if err_corr:
            logger.warning(
                f"Residual correlations {err_corr} are ignored in DAG. Use the SEM class to keep them."
            )
        return cls(ebunch=ebunch, latents=latents)

    @classmethod
    def from_dagitty(cls, string=None, filename=None) -> "DAG":
        """
        Initializes a `DAG` instance using DAGitty syntax.

        Creates a `DAG` from the dagitty string. If parameter `beta` is specified in the DAGitty
        string, the method returns a `LinearGaussianBayesianNetwork` instead of a plain `DAG`.

        Parameters
        ----------
        string: str (default: None)
            A `DAGitty` style multiline set of regression equation representing the model.
            Refer https://www.dagitty.net/manual-3.x.pdf#page=3.58 and
            https://github.com/jtextor/dagitty/blob/7a657776dc8f5e5ba4e323edb028e2c2aaf29327/gui/js/dagitty.js#L3417

        filename: str (default: None)
            The filename of the file containing the model in DAGitty syntax.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> dag = DAG.from_dagitty(
        ...     "dag{'carry matches' [latent] cancer [outcome] smoking -> 'carry matches' [beta=0.2]",
        ...     "smoking -> cancer [beta=0.5] 'carry matches' -> cancer }",
        ... )

        Creating a Linear Gaussian Bayesian network from dagitty:

        >>> from pgmpy.base import DAG
        >>> from pgmpy.models import LinearGaussianBayesianNetwork as LGBN

        # Specifying beta creates a LinearGaussianBayesianNetwork instance
        >>> dag = DAG.from_dagitty("dag{X -> Y [beta=0.3] Y -> Z [beta=0.1]}")
        >>> data = dag.simulate(n_samples=int(1e4))

        >>> from pgmpy.base import DAG
        >>> from pgmpy.models import LinearGaussianBayesianNetwork as LGBN
        """
        if filename:
            with open(filename, "r") as f:
                dagitty_str = f.readlines()
        elif string:
            dagitty_str = string.split("\n")
        else:
            raise ValueError("Either `filename` or `string` need to be specified")

        ebunch, latents, coefs, nodes = parse_dagitty(dagitty_str)
        if len(coefs) == 0:
            dag = cls(ebunch=ebunch, latents=latents)
            dag.add_nodes_from(nodes)
            return dag
        else:
            from pgmpy.factors.continuous import LinearGaussianCPD
            from pgmpy.models import LinearGaussianBayesianNetwork

            lgbn = LinearGaussianBayesianNetwork(ebunch=ebunch, latents=latents)
            lgbn.add_nodes_from(nodes)

            std = 1
            intercept = 0

            cpds = []
            for i, var in enumerate(lgbn.nodes()):
                parents = lgbn.get_parents(var)
                if var not in coefs:
                    coefs[var] = {}

                rng = np.random.default_rng()

                beta = rng.normal(loc=0, scale=1, size=(len(parents) + 1))
                beta[0] = intercept

                for i, ev in enumerate(parents):
                    if ev in coefs[var]:
                        beta[i + 1] = coefs[var][ev]

                cpd = LinearGaussianCPD(
                    variable=var,
                    beta=beta,
                    std=std,
                    evidence=parents,
                )

                cpds.append(cpd)
            lgbn.add_cpds(*cpds)

            return lgbn

    def add_node(
        self,
        node: Hashable,
        weight: Optional[float] = None,
        **kwargs,
    ):
        """
        Adds a single node to the Graph.

        Parameters
        ----------
        node: str, int, or any hashable python object.
            The node to add to the graph.

        weight: int, float
            The weight of the node.
        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG()
        >>> G.add_node(node="A")
        >>> sorted(G.nodes())
        ['A']
        """

        super().add_node(node, weight=weight, **kwargs)

    def add_nodes_from(
        self,
        nodes: Iterable[Hashable],
        weights: Optional[list[float] | tuple[float]] = None,
    ):
        """
        Add multiple nodes to the Graph.

        **The behviour of adding weights is different than in networkx.

        Parameters
        ----------
        nodes: iterable container
            A container (list, dict, set) of nodes (str, int or any hashable python
            object).

        weights: list, tuple (default=None)
            A container of weights (int, float). The weight value at index i
            is associated with the variable at index i.


        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG()
        >>> G.add_nodes_from(nodes=["A", "B", "C"])
        >>> G.nodes()
        NodeView(('A', 'B', 'C'))

        Adding nodes with weights:

        >>> G.add_nodes_from(nodes=["D", "E"], weights=[0.3, 0.6])
        >>> G.nodes["D"]
        {'weight': 0.3}
        >>> G.nodes["E"]
        {'weight': 0.6}
        >>> G.nodes["A"]
        {'weight': None}
        """
        nodes = list(nodes)

        if weights:
            if len(nodes) != len(weights):
                raise ValueError(
                    "The number of elements in nodes and weights" "should be equal."
                )
            for index in range(len(nodes)):
                self.add_node(node=nodes[index], weight=weights[index])
        else:
            for index in range(len(nodes)):
                self.add_node(node=nodes[index])

    def add_edge(self, u: Hashable, v: Hashable, weight: Optional[int | float] = None):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are not already in the graph.

        Parameters
        ----------
        u, v : nodes
            Nodes can be any hashable Python object.

        weight: int, float (default=None)
            The weight of the edge

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG()
        >>> G.add_nodes_from(nodes=["Alice", "Bob", "Charles"])
        >>> G.add_edge(u="Alice", v="Bob")
        >>> G.nodes()
        NodeView(('Alice', 'Bob', 'Charles'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob')])

        When the node is not already present in the graph:

        >>> G.add_edge(u="Alice", v="Ankur")
        >>> G.nodes()
        NodeView(('Alice', 'Ankur', 'Bob', 'Charles'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob'), ('Alice', 'Ankur')])

        Adding edges with weight:

        >>> G.add_edge("Ankur", "Maria", weight=0.1)
        >>> G.edge["Ankur"]["Maria"]
        {'weight': 0.1}
        """
        super().add_edge(u, v, weight=weight)

    def add_edges_from(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable]],
        weights: list[float] | tuple[float] | None = None,
    ):
        """
        Add all the edges in ebunch.

        If nodes referred in the ebunch are not already present, they
        will be automatically added. Node names can be any hashable python
        object.

        **The behavior of adding weights is different than networkx.

        Parameters
        ----------
        ebunch : container of edges
            Each edge given in the container will be added to the graph.
            The edges must be given as 2-tuples (u, v).

        weights: list, tuple (default=None)
            A container of weights (int, float). The weight value at index i
            is associated with the edge at index i.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG()
        >>> G.add_nodes_from(nodes=["Alice", "Bob", "Charles"])
        >>> G.add_edges_from(ebunch=[("Alice", "Bob"), ("Bob", "Charles")])
        >>> G.nodes()
        NodeView(('Alice', 'Bob', 'Charles'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob'), ('Bob', 'Charles')])

        When the node is not already in the model:

        >>> G.add_edges_from(ebunch=[("Alice", "Ankur")])
        >>> G.nodes()
        NodeView(('Alice', 'Bob', 'Charles', 'Ankur'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob'), ('Bob', 'Charles'), ('Alice', 'Ankur')])

        Adding edges with weights:

        >>> G.add_edges_from(
        ...     [("Ankur", "Maria"), ("Maria", "Mason")], weights=[0.3, 0.5]
        ... )
        >>> G.edge["Ankur"]["Maria"]
        {'weight': 0.3}
        >>> G.edge["Maria"]["Mason"]
        {'weight': 0.5}

        or

        >>> G.add_edges_from([("Ankur", "Maria", 0.3), ("Maria", "Mason", 0.5)])
        """
        ebunch = list(ebunch)

        if weights:
            if len(ebunch) != len(weights):
                raise ValueError(
                    "The number of elements in ebunch and weights" "should be equal"
                )
            for index in range(len(ebunch)):
                self.add_edge(ebunch[index][0], ebunch[index][1], weight=weights[index])
        else:
            for edge in ebunch:
                if len(edge) == 2:
                    self.add_edge(edge[0], edge[1])
                else:
                    self.add_edge(edge[0], edge[1], edge[2])

    def get_parents(self, node: Hashable):
        """
        Returns a list of parents of node.

        Throws an error if the node is not present in the graph.

        Parameters
        ----------
        node: string, int or any hashable python object.
            The node whose parents would be returned.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG(ebunch=[("diff", "grade"), ("intel", "grade")])
        >>> G.get_parents(node="grade")
        ['diff', 'intel']
        """
        return list(self.predecessors(node))

    def moralize(self):
        """
        Removes all the immoralities in the DAG and creates a moral
        graph (UndirectedGraph).

        A v-structure X->Z<-Y is an immorality if there is no directed edge
        between X and Y.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG(ebunch=[("diff", "grade"), ("intel", "grade")])
        >>> moral_graph = G.moralize()
        >>> moral_graph.edges()
        EdgeView([('intel', 'grade'), ('intel', 'diff'), ('grade', 'diff')])
        """
        from pgmpy.base import UndirectedGraph

        moral_graph = UndirectedGraph()
        moral_graph.add_nodes_from(self.nodes())
        moral_graph.add_edges_from(self.to_undirected().edges())

        for node in self.nodes():
            moral_graph.add_edges_from(
                itertools.combinations(self.get_parents(node), 2)
            )

        return moral_graph

    def get_leaves(self):
        """
        Returns a list of leaves of the graph.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> graph = DAG([("A", "B"), ("B", "C"), ("B", "D")])
        >>> graph.get_leaves()
        ['C', 'D']
        """
        return [node for node, out_degree in self.out_degree_iter() if out_degree == 0]

    def out_degree_iter(self, nbunch=None, weight=None):
        return iter(self.out_degree(nbunch, weight))

    def in_degree_iter(self, nbunch=None, weight=None):
        return iter(self.in_degree(nbunch, weight))

    def get_roots(self):
        """
        Returns a list of roots of the graph.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> graph = DAG([("A", "B"), ("B", "C"), ("B", "D"), ("E", "B")])
        >>> graph.get_roots()
        ['A', 'E']
        """
        return [
            node for node, in_degree in dict(self.in_degree()).items() if in_degree == 0
        ]

    def get_children(self, node: Hashable):
        """
        Returns a list of children of node.
        Throws an error if the node is not present in the graph.

        Parameters
        ----------
        node: string, int or any hashable python object.
            The node whose children would be returned.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> g = DAG(
        ...     ebunch=[
        ...         ("A", "B"),
        ...         ("C", "B"),
        ...         ("B", "D"),
        ...         ("B", "E"),
        ...         ("B", "F"),
        ...         ("E", "G"),
        ...     ]
        ... )
        >>> g.get_children(node="B")
        ['D', 'E', 'F']
        """
        return list(self.successors(node))

    def get_independencies(
        self, latex=False, include_latents=False
    ) -> Independencies | list[str]:
        """
        Computes independencies in the DAG, by checking minimal d-seperation.

        Parameters
        ----------
        latex: boolean
            If latex=True then latex string of the independence assertion
            would be created.

        include_latents: boolean
            If True, includes latent variables in the independencies. Otherwise,
            only generates independencies on observed variables.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> chain = DAG([("X", "Y"), ("Y", "Z")])
        >>> chain.get_independencies()
        (X \u27c2 Z | Y)
        """
        nodes = set(self.nodes())
        if not include_latents:
            nodes -= self.latents

        independencies = Independencies()
        for x, y in itertools.combinations(nodes, 2):
            if not self.has_edge(x, y) and not self.has_edge(y, x):
                minimal_separator = self.minimal_dseparator(
                    start=x, end=y, include_latents=include_latents
                )
                if minimal_separator is not None:
                    independencies.add_assertions([x, y, minimal_separator])

        independencies = independencies.reduce()

        if not latex:
            return independencies
        else:
            return independencies.latex_string()

    def local_independencies(
        self, variables: list[Hashable] | tuple[Hashable, ...] | str
    ):
        """
        Returns an instance of Independencies containing the local independencies
        of each of the variables.

        Parameters
        ----------
        variables: str or array like
            variables whose local independencies are to be found.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> student = DAG()
        >>> student.add_edges_from(
        ...     [
        ...         ("diff", "grade"),
        ...         ("intel", "grade"),
        ...         ("grade", "letter"),
        ...         ("intel", "SAT"),
        ...     ]
        ... )
        >>> ind = student.local_independencies("grade")
        >>> ind
        (grade \u27c2 SAT | diff, intel)
        """

        independencies = Independencies()
        for variable in (
            variables if isinstance(variables, (list, tuple)) else [variables]
        ):
            non_descendents = (
                set(self.nodes())
                - {variable}
                - set(nx.dfs_preorder_nodes(self, variable))
            )
            parents = set(self.get_parents(variable))
            if non_descendents - parents:
                independencies.add_assertions(
                    [variable, non_descendents - parents, parents]
                )
        return independencies

    def is_iequivalent(self, model: "DAG"):
        """
        Checks whether the given model is I-equivalent

        Two graphs G1 and G2 are said to be I-equivalent if they have same skeleton
        and have same set of immoralities.

        Parameters
        ----------
        model : A DAG object, for which you want to check I-equivalence

        Returns
        --------
        I-equivalence: boolean
            True if both are I-equivalent, False otherwise

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG()
        >>> G.add_edges_from([("V", "W"), ("W", "X"), ("X", "Y"), ("Z", "Y")])
        >>> G1 = DAG()
        >>> G1.add_edges_from([("W", "V"), ("X", "W"), ("X", "Y"), ("Z", "Y")])
        >>> G.is_iequivalent(G1)
        True

        """
        if not isinstance(model, DAG):
            raise TypeError(
                f"Model must be an instance of DAG. Got type: {type(model)}"
            )

        if (self.to_undirected().edges() == model.to_undirected().edges()) and (
            self.get_immoralities() == model.get_immoralities()
        ):
            return True
        return False

    def get_immoralities(self) -> dict[Hashable, list[tuple[Hashable, Hashable]]]:
        """
        Finds all the immoralities in the model
        A v-structure X -> Z <- Y is an immorality if there is no direct edge between X and Y .

        Returns
        -------
        Immoralities: set
            A set of all the immoralities in the model

        Examples
        ---------
        >>> from pgmpy.base import DAG
        >>> student = DAG()
        >>> student.add_edges_from(
        ...     [
        ...         ("diff", "grade"),
        ...         ("intel", "grade"),
        ...         ("intel", "SAT"),
        ...         ("grade", "letter"),
        ...     ]
        ... )
        >>> student.get_immoralities()
        {('diff', 'intel')}
        """
        immoralities = dict()
        for node in self.nodes():
            parent_pairs = []
            for parents in itertools.combinations(self.predecessors(node), 2):
                if not self.has_edge(parents[0], parents[1]) and not self.has_edge(
                    parents[1], parents[0]
                ):
                    parent_pairs.append(tuple(sorted(parents)))
            immoralities[node] = parent_pairs
        return immoralities

    def is_dconnected(
        self,
        start: Hashable,
        end: Hashable,
        observed: Optional[Sequence[Hashable]] = None,
        include_latents=False,
    ):
        """
        Returns True if there is an active trail (i.e. d-connection) between
        `start` and `end` node given that `observed` is observed.

        Parameters
        ----------
        start, end : int, str, any hashable python object.
            The nodes in the DAG between which to check the d-connection/active trail.

        observed : list, array-like (optional)
            If given the active trail would be computed assuming these nodes to
            be observed.

        include_latents: boolean (default: False)
            If true, latent variables are return as part of the active trail.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> student = DAG()
        >>> student.add_nodes_from(["diff", "intel", "grades", "letter", "sat"])
        >>> student.add_edges_from(
        ...     [
        ...         ("diff", "grades"),
        ...         ("intel", "grades"),
        ...         ("grades", "letter"),
        ...         ("intel", "sat"),
        ...     ]
        ... )
        >>> student.is_dconnected("diff", "intel")
        False
        >>> student.is_dconnected("grades", "sat")
        True
        """
        if (
            end
            in self.active_trail_nodes(
                variables=start, observed=observed, include_latents=include_latents
            )[start]
        ):
            return True
        else:
            return False

    def minimal_dseparator(
        self, start: Hashable, end: Hashable, include_latents=False
    ) -> set[Hashable]:
        """
        Finds the minimal d-separating set for `start` and `end`.

        Parameters
        ----------
        start: node
            The first node.

        end: node
            The second node.

        include_latents: boolean (default: False)
            If true, latent variables are consider for minimal d-seperator.

        Examples
        --------
        >>> dag = DAG([("A", "B"), ("B", "C")])
        >>> dag.minimal_dseparator(start="A", end="C")
        {'B'}

        References
        ----------
        [1] Algorithm 4, Page 10: Tian, Jin, Azaria Paz, and
          Judea Pearl. Finding minimal d-separators. Computer Science Department,
            University of California, 1998.
        """
        if (end in self.neighbors(start)) or (start in self.neighbors(end)):
            raise ValueError(
                "No possible separators because start and end are adjacent"
            )
        an_graph = self.get_ancestral_graph([start, end])
        separator = set(
            itertools.chain(self.predecessors(start), self.predecessors(end))
        )

        if not include_latents:
            # If any of the parents were latents, take the latent's parent
            while separator.intersection(self.latents):
                separator_copy = separator.copy()
                for u in separator:
                    if u in self.latents:
                        separator_copy.remove(u)
                        separator_copy.update(set(self.predecessors(u)))
                separator = separator_copy

        # Remove the start and end nodes in case it reaches there while removing latents.
        separator.difference_update({start, end})

        # If the initial set is not able to d-separate, no d-separator is possible.
        if an_graph.is_dconnected(start, end, observed=separator):
            return None

        # Go through the separator set, remove one element and check if it remains
        # a dseparating set.
        minimal_separator = separator.copy()

        for u in separator:
            if not an_graph.is_dconnected(start, end, observed=minimal_separator - {u}):
                minimal_separator.remove(u)

        return minimal_separator

    def get_markov_blanket(self, node: Hashable) -> list[Hashable]:
        """
        Returns a markov blanket for a random variable. In the case
        of Bayesian Networks, the markov blanket is the set of
        node's parents, its children and its children's other parents.

        Returns
        -------
        Markov Blanket: list
            List of nodes in the markov blanket of `node`.

        Parameters
        ----------
        node: string, int or any hashable python object.
              The node whose markov blanket would be returned.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> G = DAG(
        ...     [
        ...         ("x", "y"),
        ...         ("z", "y"),
        ...         ("y", "w"),
        ...         ("y", "v"),
        ...         ("u", "w"),
        ...         ("s", "v"),
        ...         ("w", "t"),
        ...         ("w", "m"),
        ...         ("v", "n"),
        ...         ("v", "q"),
        ...     ]
        ... )
        >>> G.get_markov_blanket("y")
        ['s', 'w', 'x', 'u', 'z', 'v']
        """
        children = self.get_children(node)
        parents = self.get_parents(node)
        blanket_nodes = children + parents
        for child_node in children:
            blanket_nodes.extend(self.get_parents(child_node))
        blanket_nodes = set(blanket_nodes)
        blanket_nodes.discard(node)
        return list(blanket_nodes)

    def active_trail_nodes(
        self,
        variables: list[Hashable] | Hashable,
        observed: Optional[
            Hashable | list[Hashable] | tuple[Hashable, Hashable]
        ] = None,
        include_latents=False,
    ) -> dict[Hashable, set[Hashable]]:
        """
        Returns a dictionary with the given variables as keys and all the nodes reachable
        from that respective variable as values.

        Parameters
        ----------
        variables: str or array like
            variables whose active trails are to be found.

        observed : List of nodes (optional)
            If given the active trails would be computed assuming these nodes to be
            observed.

        include_latents: boolean (default: False)
            Whether to include the latent variables in the returned active trail nodes.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> student = DAG()
        >>> student.add_nodes_from(["diff", "intel", "grades"])
        >>> student.add_edges_from([("diff", "grades"), ("intel", "grades")])
        >>> student.active_trail_nodes("diff")
        {'diff': {'diff', 'grades'}}
        >>> student.active_trail_nodes(["diff", "intel"], observed="grades")
        {'diff': {'diff', 'intel'}, 'intel': {'diff', 'intel'}}

        References
        ----------
        Details of the algorithm can be found in 'Probabilistic Graphical Model
        Principles and Techniques' - Koller and Friedman
        Page 75 Algorithm 3.1
        """
        observed_list: list[Hashable] | tuple[Hashable, Hashable]
        if observed:
            if isinstance(observed, set):
                observed = list(observed)

            observed_list = (
                observed if isinstance(observed, (list, tuple)) else [observed]
            )
        else:
            observed_list = []
        ancestors_list = self._get_ancestors_of(observed_list)

        # Direction of flow of information
        # up ->  from parent to child
        # down -> from child to parent

        active_trails = {}
        for start in variables if isinstance(variables, list) else [variables]:
            visit_list = set()
            visit_list.add((start, "up"))
            traversed_list = set()
            active_nodes = set()
            while visit_list:
                node, direction = visit_list.pop()
                if (node, direction) not in traversed_list:
                    if node not in observed_list:
                        active_nodes.add(node)
                    traversed_list.add((node, direction))
                    if direction == "up" and node not in observed_list:
                        for parent in self.predecessors(node):
                            visit_list.add((parent, "up"))
                        for child in self.successors(node):
                            visit_list.add((child, "down"))
                    elif direction == "down":
                        if node not in observed_list:
                            for child in self.successors(node):
                                visit_list.add((child, "down"))
                        if node in ancestors_list:
                            for parent in self.predecessors(node):
                                visit_list.add((parent, "up"))
            if include_latents:
                active_trails[start] = active_nodes
            else:
                active_trails[start] = active_nodes - self.latents

        return active_trails

    def _get_ancestors_of(
        self, nodes: str | tuple[Hashable, Hashable] | Iterable[Hashable]
    ) -> set[Hashable]:
        """
        Returns a dictionary of all ancestors of all the observed nodes including the
        node itself.

        Parameters
        ----------
        nodes: string, list-type
            name of all the observed nodes

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> model = DAG([("D", "G"), ("I", "G"), ("G", "L"), ("I", "L")])
        >>> model._get_ancestors_of("G")
        {'D', 'G', 'I'}
        >>> model._get_ancestors_of(["G", "I"])
        {'D', 'G', 'I'}
        """
        if not isinstance(nodes, (list, tuple)):
            nodes = [nodes]

        for node in nodes:
            if node not in self.nodes():
                raise ValueError(f"Node {node} not in graph")

        ancestors_list = set()
        for node in nodes:
            ancestors_list.update(nx.ancestors(self, node))

        ancestors_list.update(nodes)
        return ancestors_list

    def to_pdag(self):
        """
        Returns the CPDAG (Completed Partial DAG) of the DAG representing the equivalence class
        that the given DAG belongs to.

        Returns
        -------
        CPDAG: pgmpy.base.PDAG
            An instance of pgmpy.base.PDAG representing the CPDAG of the given DAG.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> dag = DAG([("A", "B"), ("B", "C"), ("C", "D")])
        >>> pdag = dag.to_pdag()
        >>> pdag.directed_edges
        {('A', 'B'), ('B', 'C'), ('C', 'D')}

        References
        ----------
        [1] Chickering, David Maxwell. "Learning equivalence classes of Bayesian-network structures."
          Journal of machine learning research 2.Feb (2002): 445-498. Figure 4 and 5.
        """
        # Perform a topological sort on the nodes
        topo_order = list(nx.topological_sort(self))
        node_order = {node: i for i, node in enumerate(topo_order)}

        # Initialize edge ordering
        i = 0
        edge_order = {}
        unordered_edges = set(self.edges())

        # While there are unordered edges
        while unordered_edges:
            # Find lowest ordered node with unordered edges incident into it
            nodes_with_unordered_edges = {edge[1] for edge in unordered_edges}
            y = min(nodes_with_unordered_edges, key=lambda x: node_order[x])

            # Find highest ordered node for which x->y is not ordered
            unordered_edges_into_y = {edge for edge in unordered_edges if edge[1] == y}
            x = max(
                (edge[0] for edge in unordered_edges_into_y),
                key=lambda x: node_order[x],
            )

            # Label x->y with order i
            edge_order[(x, y)] = i
            i += 1
            unordered_edges.remove((x, y))

        # Label every edge as "unknown"
        edge_labels = {edge: "unknown" for edge in self.edges()}

        # While there are edges labeled "unknown"
        while any(label == "unknown" for label in edge_labels.values()):
            # Let x -> y be the lowest ordered edge that is labeled "unknown"
            unknown_edges = [
                (edge, edge_order[edge])
                for edge, label in edge_labels.items()
                if label == "unknown"
            ]
            x, y = min(unknown_edges, key=lambda x: x[1])[0]

            # Check compelled parents
            compelled_parents = [
                w for w in self.get_parents(x) if edge_labels.get((w, x)) == "compelled"
            ]
            for w in compelled_parents:
                if not self.has_edge(w, y):
                    # Label x -> y and every edge incident into y with "compelled"
                    edge_labels[(x, y)] = "compelled"
                    for z in self.get_parents(y):
                        if edge_labels.get((z, y)) == "unknown":
                            edge_labels[(z, y)] = "compelled"
                    break
                else:
                    # Label w -> y with "compelled"
                    edge_labels[(w, y)] = "compelled"

            # Check for v-structures
            if edge_labels.get((x, y)) != "compelled":
                v_structure_exists = False
                for z in self.get_parents(y):
                    if z != x and not self.has_edge(z, x):
                        v_structure_exists = True
                        break

                if v_structure_exists:
                    # Label x -> y and all "unknown" edges incident into y with "compelled"
                    edge_labels[(x, y)] = "compelled"
                    for z in self.get_parents(y):
                        if edge_labels.get((z, y)) == "unknown":
                            edge_labels[(z, y)] = "compelled"
                else:
                    # Label x -> y and all "unknown" edges incident into y with "reversible"
                    edge_labels[(x, y)] = "reversible"
                    for z in self.get_parents(y):
                        if edge_labels.get((z, y)) == "unknown":
                            edge_labels[(z, y)] = "reversible"

        # Create PDAG with directed and undirected edges
        directed_edges = [
            edge for edge, label in edge_labels.items() if label == "compelled"
        ]
        undirected_edges = [
            edge for edge, label in edge_labels.items() if label == "reversible"
        ]

        from pgmpy.base import PDAG

        return PDAG(
            directed_ebunch=directed_edges,
            undirected_ebunch=undirected_edges,
            latents=self.latents,
        )

    def do(
        self,
        nodes: Hashable | Iterable[Hashable] | tuple[Hashable, Hashable],
        inplace=False,
    ):
        """
        Applies the do operator to the graph and returns a new DAG with the
        transformed graph.

        The do-operator, do(X = x) has the effect of removing all edges from
        the parents of X and setting X to the given value x.

        Parameters
        ----------
        nodes : list, array-like
            The names of the nodes to apply the do-operator for.

        inplace: boolean (default: False)
            If inplace=True, makes the changes to the current object,
            otherwise returns a new instance.

        Returns
        -------
        Modified DAG: pgmpy.base.DAG
            A new instance of DAG modified by the do-operator

        Examples
        --------
        Initialize a DAG

        >>> graph = DAG()
        >>> graph.add_edges_from([("X", "A"), ("A", "Y"), ("A", "B")])
        >>> # Applying the do-operator will return a new DAG with the desired structure.
        >>> graph_do_A = graph.do("A")
        >>> # Which we can verify is missing the edges we would expect.
        >>> graph_do_A.edges
        OutEdgeView([('A', 'B'), ('A', 'Y')])

        References
        ----------
        Causality: Models, Reasoning, and Inference, Judea Pearl (2000). p.70.
        """
        dag = self if inplace else self.copy()

        if isinstance(nodes, (str, int)):
            nodes = [nodes]
        else:
            nodes = list(nodes)

        if not set(nodes).issubset(set(self.nodes())):
            raise ValueError(
                f"Nodes not found in the model: {set(nodes) - set(self.nodes())}"
            )

        for node in nodes:
            parents = list(dag.predecessors(node))
            for parent in parents:
                dag.remove_edge(parent, node)
        return dag

    def get_ancestral_graph(self, nodes: Iterable[Hashable]):
        """
        Returns the ancestral graph of the given `nodes`. The ancestral graph only
        contains the nodes which are ancestors of at least one of the variables in
        node.

        Parameters
        ----------
        node: iterable
            List of nodes whose ancestral graph needs to be computed.

        Returns
        -------
        Ancestral Graph: pgmpy.base.DAG

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> dag = DAG([("A", "C"), ("B", "C"), ("D", "A"), ("D", "B")])
        >>> anc_dag = dag.get_ancestral_graph(nodes=["A", "B"])
        >>> anc_dag.edges()
        OutEdgeView([('D', 'A'), ('D', 'B')])
        """
        return self.subgraph(nodes=self._get_ancestors_of(nodes=nodes))

    def to_daft(
        self,
        node_pos: str | dict[Hashable, tuple[int, int]] = "circular",
        latex=True,
        pgm_params={},
        edge_params={},
        node_params={},
        plot_edge_strength=False,
    ):
        """
        Returns a daft (https://docs.daft-pgm.org/en/latest/) object which can be rendered for
        publication quality plots. The returned object's render method can be called to see the plots.

        Parameters
        ----------
        node_pos: str or dict (default: circular)
            If str: Must be one of the following: circular, kamada_kawai, planar, random, shell, sprint,
                spectral, spiral. Please refer:
                  https://networkx.org/documentation/stable//reference/drawing.html#module-networkx.drawing.layout
                    for details on these layouts.

            If dict should be of the form {node: (x coordinate, y coordinate)} describing the x and y coordinate of each
            node.

            If no argument is provided uses circular layout.

        latex: boolean
            Whether to use latex for rendering the node names.

        pgm_params: dict (optional)
            Any additional parameters that need to be passed to `daft.PGM` initializer.
            Should be of the form: {param_name: param_value}

        edge_params: dict (optional)
            Any additional edge parameters that need to be passed to `daft.add_edge` method.
            Should be of the form: {(u1, v1): {param_name: param_value}, (u2, v2): {...} }

        node_params: dict (optional)
            Any additional node parameters that need to be passed to `daft.add_node` method.
            Should be of the form: {node1: {param_name: param_value}, node2: {...} }

        plot_edge_strength: bool (default: False)
            If True, displays edge strength values as labels on edges.
            Requires edge strengths to be computed first using the edge_strength() method.

        Returns
        -------
        Daft object: daft.PGM object
            Daft object for plotting the DAG.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> dag = DAG([("a", "b"), ("b", "c"), ("d", "c")])
        >>> dag.to_daft(node_pos={"a": (0, 0), "b": (1, 0), "c": (2, 0), "d": (1, 1)})
        <daft.PGM at 0x7fc756e936d0>
        >>> dag.to_daft(node_pos="circular")
        <daft.PGM at 0x7f9bb48c5eb0>
        >>> dag.to_daft(node_pos="circular", pgm_params={"observed_style": "inner"})
        <daft.PGM at 0x7f9bb48b0bb0>
        >>> dag.to_daft(
        ...     node_pos="circular",
        ...     edge_params={("a", "b"): {"label": 2}},
        ...     node_params={"a": {"shape": "rectangle"}},
        ... )
        <daft.PGM at 0x7f9bb48b0bb0>
        """
        try:
            from daft import PGM
        except ImportError as e:
            raise ImportError(
                f"{e}. Package `daft` is required for plotting probabilistic graphical models.\n"
                "Please install it using: pip install daft-pgm\n"
                "Documentation: https://docs.daft-pgm.org/en/latest/"
            ) from None

        # Check edge strength existence if plotting is requested
        if plot_edge_strength:
            missing_strengths = []
            for u, v in self.edges():
                if "strength" not in self.edges[(u, v)]:
                    missing_strengths.append((u, v))

            if missing_strengths:
                raise ValueError(
                    f"Edge strength plotting requested but strengths not found for edges: {missing_strengths}. "
                    "Use edge_strength() method to compute strengths first."
                )

        if isinstance(node_pos, str):
            supported_layouts = {
                "circular": nx.circular_layout,
                "kamada_kawai": nx.kamada_kawai_layout,
                "planar": nx.planar_layout,
                "random": nx.random_layout,
                "shell": nx.shell_layout,
                "spring": nx.spring_layout,
                "spectral": nx.spectral_layout,
                "spiral": nx.spiral_layout,
            }
            if node_pos not in supported_layouts:
                raise ValueError(
                    "Unknown node_pos argument. Please refer docstring "
                    "for accepted values"
                )
            else:
                node_pos = supported_layouts[node_pos](self)
        elif isinstance(node_pos, dict):
            for node in self.nodes():
                if node not in node_pos:
                    raise ValueError(f"No position specified for {node}.")
        else:
            raise ValueError(
                "Argument node_pos not valid. Please refer to the docstring."
            )

        daft_pgm = PGM(**pgm_params)
        for node in self.nodes():
            observed = node in self.observed
            extra_params = node_params.get(node, dict())

            if latex:
                daft_pgm.add_node(
                    node,
                    rf"${node}$",
                    node_pos[node][0],
                    node_pos[node][1],
                    observed=observed,
                    **extra_params,
                )
            else:
                daft_pgm.add_node(
                    node,
                    f"{node}",
                    node_pos[node][0],
                    node_pos[node][1],
                    observed=observed,
                    **extra_params,
                )

        for u, v in self.edges():
            try:
                extra_params = edge_params[(u, v)]
            except KeyError:
                extra_params = dict()

            # Add edge strength as label if requested
            if plot_edge_strength:
                strength_value = self.edges[(u, v)]["strength"]
                strength_label = f"{strength_value: .3f}"
                if "label" not in extra_params:
                    extra_params["label"] = strength_label

            daft_pgm.add_edge(u, v, **extra_params)

        return daft_pgm

    @staticmethod
    def get_random(
        n_nodes=5,
        edge_prob=0.5,
        node_names: Optional[list[Hashable]] = None,
        latents=False,
        seed: Optional[int] = None,
    ) -> "DAG":
        """
        Returns a randomly generated DAG with `n_nodes` number of nodes with
        edge probability being `edge_prob`.

        Parameters
        ----------
        n_nodes: int
            The number of nodes in the randomly generated DAG.

        edge_prob: float
            The probability of edge between any two nodes in the topologically
            sorted DAG.

        node_names: list (default: None)
            A list of variables names to use in the random graph.
            If None, the node names are integer values starting from 0.

        latents: bool (default: False)
            If True, includes latent variables in the generated DAG.

        seed: int (default: None)
            The seed for the random number generator.

        Returns
        -------
        Random DAG: pgmpy.base.DAG
            The randomly generated DAG.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> random_dag = DAG.get_random(n_nodes=10, edge_prob=0.3)
        >>> random_dag.nodes()
        NodeView((0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
        >>> random_dag.edges()
        OutEdgeView([(0, 6), (1, 6), (1, 7), (7, 9), (2, 5), (2, 7), (2, 8), (5, 9), (3, 7)])
        """
        # Step 1: Generate a matrix of 0 and 1. Prob of choosing 1 = edge_prob
        gen = np.random.default_rng(seed=seed)
        adj_mat = gen.choice(
            [0, 1], size=(n_nodes, n_nodes), p=[1 - edge_prob, edge_prob]
        )

        # Step 2: Use the upper triangular part of the matrix as adjacency.
        if node_names is None:
            node_names = list([f"X_{i}" for i in range(n_nodes)])

        adj_pd = pd.DataFrame(
            np.triu(adj_mat, k=1), columns=node_names, index=node_names
        )
        nx_dag = nx.from_pandas_adjacency(adj_pd, create_using=nx.DiGraph)

        dag = DAG(nx_dag)
        dag.add_nodes_from(node_names)

        if latents:
            dag.latents = set(
                gen.choice(dag.nodes(), gen.integers(low=0, high=len(dag.nodes())))
            )
        return dag

    def to_graphviz(self, plot_edge_strength=False):
        """
        Retuns a pygraphviz object for the DAG. pygraphviz is useful for
        visualizing the network structure.

        Parameters
        ----------
        plot_edge_strength: bool (default: False)
            If True, displays edge strength values as labels on edges.
            Requires edge strengths to be computed first using the edge_strength() method.

        Returns
        -------
        AGraph object: pygraphviz.AGraph
            pygraphviz object for plotting the DAG.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model
        >>> model = get_example_model("alarm")
        >>> model.to_graphviz()
        <AGraph <Swig Object of type 'Agraph_t *' at 0x7fdea4cde040>>
        >>> model.draw("model.png", prog="neato")
        """
        if plot_edge_strength:
            missing_strengths = []
            for u, v in self.edges():
                if "strength" not in self.edges[(u, v)]:
                    missing_strengths.append((u, v))

            if missing_strengths:
                raise ValueError(
                    f"Edge strength plotting requested but strengths not found for edges: {missing_strengths}. "
                    "Use edge_strength() method to compute strengths first."
                )

        agraph = nx.nx_agraph.to_agraph(self)

        if plot_edge_strength:
            for u, v in self.edges():
                strength_value = self.edges[(u, v)]["strength"]
                strength_label = f"{strength_value: .3f}"
                agraph.get_edge(u, v).attr["label"] = strength_label

        return agraph

    def to_lavaan(self) -> str:
        """
        Convert the DAG to lavaan syntax representation.

        The lavaan syntax represents structural equations where each line
        shows a dependent variable regressed on its parents using the ~ operator.
        Isolated nodes (nodes with no parents) are not included in the output.

        Returns
        -------
        str
            String representation of the DAG in lavaan syntax format.
            Each line represents a regression equation where the dependent
            variable is regressed on its parents.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> dag = DAG([("X", "Y"), ("Z", "Y")])
        >>> print(dag.to_lavaan())
        Y ~ X + Z

        >>> dag2 = DAG([("A", "B"), ("B", "C")])
        >>> print(dag2.to_lavaan())
        B ~ A
        C ~ B

        >>> # Empty DAG returns empty string
        >>> empty_dag = DAG()
        >>> print(empty_dag.to_lavaan())
        ""

        Notes
        -----
        - Node names are converted to string representations using str().
        - If node names contain spaces or special characters, they will be used as-is.
        - Users should ensure node names are valid in R/lavaan context if needed.

        References
        ----------
        lavaan syntax: http://lavaan.ugent.be/tutorial/syntax1.html
        """
        lavaan_statements = []

        # Create regression equations for nodes with parents in the format "Y ~ X + Z"
        for node in sorted(self.nodes(), key=str):
            parents = self.get_parents(node)
            if parents:
                node_str = str(node)
                parent_strs = sorted([str(parent) for parent in parents], key=str)
                parents_str = " + ".join(parent_strs)
                lavaan_statements.append(f"{node_str} ~ {parents_str}")

        return "\n".join(lavaan_statements)

    def to_dagitty(self) -> str:
        """
        Convert the DAG to dagitty syntax representation.

        The dagitty syntax represents directed acyclic graphs using
        the dag { statements } format with -> for directed edges.
        Isolated nodes (nodes with no edges) are included as standalone nodes.

        Returns
        -------
        str
            String representation of the DAG in dagitty syntax format.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> dag = DAG([("X", "Y"), ("Z", "Y")])
        >>> print(dag.to_dagitty())
        dag {
        X -> Y
        Z -> Y
        }

        >>> dag2 = DAG([("A", "B"), ("B", "C")])
        >>> print(dag2.to_dagitty())
        dag {
        A -> B
        B -> C
        }

        >>> # DAG with isolated node
        >>> dag3 = DAG()
        >>> dag3.add_nodes_from(["A", "B"])
        >>> dag3.add_edge("A", "B")
        >>> dag3.add_node("C")  # Isolated node
        >>> print(dag3.to_dagitty())
        dag {
        A -> B
        C
        }

        Notes
        -----
        - Node names are converted to string representations using str().
        - If node names contain spaces or special characters, they will be used as-is.
        - Users should ensure node names are valid in R/dagitty context if needed.

        References
        ----------
        dagitty syntax: https://cran.r-project.org/web/packages/dagitty/dagitty.pdf
        """
        statements = []

        # Create edge statements in "X -> Y" format and add isolated nodes
        if self.edges():
            edge_statements = []
            for parent, child in sorted(
                self.edges(), key=lambda x: (str(x[0]), str(x[1]))
            ):
                parent_str = str(parent)
                child_str = str(child)
                edge_statements.append(f"{parent_str} -> {child_str}")
            statements.extend(edge_statements)

        for node in sorted(nx.isolates(self), key=str):
            statements.append(str(node))

        content = "\n".join(statements)
        if content:
            return f"dag {{\n{content}\n}}"
        else:
            return "dag {\n}"

    def _variable_name_contains_non_string(self):
        """
        Checks if the variable names contain any non-string values. Used only for CausalInference class.
        """
        for node in list(self.nodes()):
            if not isinstance(node, str):
                return (node, type(node))
        return False

    def copy(self):
        """Returns a copy of the DAG object."""
        dag = DAG(ebunch=self.edges(), latents=self.latents)
        dag.add_nodes_from(self.nodes())

        for role, vars in self.get_role_dict().items():
            dag.with_role(role=role, variables=vars, inplace=True)

        return dag

    def __eq__(self, other):
        """
        Checks if two DAGs are equal. Two DAGs are considered equal if they
        have the same nodes, edges, latent variables, and variable roles.

        Parameters
        ----------
        other: DAG object
            The other DAG to compare with.

        Returns
        -------
        bool
            True if the DAGs are equal, False otherwise.
        """
        if not isinstance(other, DAG):
            return False

        return (
            set(self.nodes()) == set(other.nodes())
            and set(self.edges()) == set(other.edges())
            and self.latents == other.latents
            and self.get_role_dict() == other.get_role_dict()
        )

    def edge_strength(self, data, edges=None):
        """
        Computes the strength of each edge in `edges`. The strength is bounded
        between 0 and 1, with 1 signifying strong effect.

        The edge strength is defined as the effect size measure of a
        Conditional Independence test using the parents as the conditional set.
        The strength quantifies the effect of edge[0] on edge[1] after
        controlling for any other influence paths. We use a residualization-based
        CI test[1] to compute the strengths.

        Interpretation:
        - The strength is the Pillai's Trace effect size of partial correlation.
        - Measures the strength of linear relationship between the residuals.
        - Works for any mixture of categorical and continuous variables.
        - The value is bounded between 0 and 1:
        - Strength close to 1 → strong dependence.
        - Strength close to 0 → conditional independence.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataset to compute edge strengths on.

        edges : tuple, list, or None (default: None)
            - None: Compute for all DAG edges.
            - Tuple (X, Y): Compute for edge X → Y.
            - List of tuples: Compute for selected edges.

        Returns
        -------
        dict
            Dictionary mapping edges to their strength values.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork as LGBN
        >>> # Create a linear Gaussian Bayesian network
        >>> linear_model = LGBN([("X", "Y"), ("Z", "Y")])
        >>> # Create CPDs with specific beta values
        >>> x_cpd = LinearGaussianCPD(variable="X", beta=[0], std=1)
        >>> y_cpd = LinearGaussianCPD(
        ...     variable="Y", beta=[0, 0.4, 0.6], std=1, evidence=["X", "Z"]
        ... )
        >>> z_cpd = LinearGaussianCPD(variable="Z", beta=[0], std=1)
        >>> # Add CPDs to the model
        >>> linear_model.add_cpds(x_cpd, y_cpd, z_cpd)
        >>> # Simulate data from the model
        >>> data = linear_model.simulate(n_samples=int(1e4))
        >>> # Create DAG and compute edge strengths
        >>> dag = DAG([("X", "Y"), ("Z", "Y")])
        >>> strengths = dag.edge_strength(data)
        {('X', 'Y'): np.float64(0.14587166611282304),
         ('Z', 'Y'): np.float64(0.25683780900125613)}

        References
        ----------
        [1] Ankan, Ankur, and Johannes Textor. "A simple unified approach to testing high-dimensional
        conditional independences for categorical and ordinal data." Proceedings of the AAAI Conference
        on Artificial Intelligence.
        """

        from pgmpy.estimators.CITests import pillai_trace

        # If edges is None, compute for all edges in the DAG
        if edges is None:
            edges_to_compute = list(self.edges())
        # If edges is a single edge tuple
        elif isinstance(edges, tuple) and len(edges) == 2:
            edges_to_compute = [edges]
        # If edges is a list of edge tuples
        elif isinstance(edges, list) and all(
            isinstance(edge, tuple) and len(edge) == 2 for edge in edges
        ):
            edges_to_compute = edges
        else:
            raise ValueError(
                "edges parameter must be either None, a 2-tuple (X, Y), or a list of 2-tuples [(X1, Y1), (X2, Y2), ...]"
            )

        strengths = {}
        skipped_edges = []

        for edge in edges_to_compute:
            x, y = edge

            # Get parents of x and y using get_parents instead of predecessors
            pa_Y = self.get_parents(y)

            # Check if either x or y is a latent node
            if (
                x in self.latents
                or y in self.latents
                or any(parent in self.latents for parent in pa_Y)
            ):
                skipped_edges.append(edge)
                continue

            # Combine parents for conditioning set (excluding x and y themselves)
            conditioning_set = set(pa_Y) - {x, y}

            # Run CI test and get effect size
            effect_size, _ = pillai_trace(
                X=x, Y=y, Z=list(conditioning_set), data=data, boolean=False
            )

            # Store the edge strength
            strengths[edge] = effect_size

            # store the values in the graph as well
            self.edges[edge]["strength"] = effect_size

        if skipped_edges:
            logger.warning(
                f"Skipped computing strengths for edges involving latent variables: {skipped_edges}. "
                "Use CausalInference class for advanced causal effect estimation."
            )

        return strengths

    def __hash__(self):
        """
        Returns a hash value for the DAG object. The hash value is computed based on
        the nodes, edges, latent variables, and variable roles of the DAG.
        """
        return hash(
            (
                frozenset(self.nodes()),
                frozenset(self.edges()),
                frozenset(self.latents),
                frozenset(
                    (role, frozenset(self.get_role(role))) for role in self.get_roles()
                ),
            )
        )
