#!/usr/bin/env python3

import itertools

import networkx as nx
import numpy as np

from pgmpy.base import UndirectedGraph
from pgmpy.global_vars import logger
from pgmpy.independencies import Independencies


class DAG(nx.DiGraph):
    """
    Base class for all Directed Graphical Models.

    Each node in the graph can represent either a random variable, `Factor`,
    or a cluster of random variables. Edges in the graph represent the
    dependencies between these.

    Parameters
    ----------
    data: input graph
        Data to initialize graph. If data=None (default) an empty graph is
        created. The data can be an edge list or any Networkx graph object.

    Examples
    --------
    Create an empty DAG with no nodes and no edges

    >>> from pgmpy.base import DAG
    >>> G = DAG()

    G can be grown in several ways:

    **Nodes:**

    Add one node at a time:

    >>> G.add_node(node='a')

    Add the nodes from any container (a list, set or tuple or the nodes
    from another graph).

    >>> G.add_nodes_from(nodes=['a', 'b'])

    **Edges:**

    G can also be grown by adding edges.

    Add one edge,

    >>> G.add_edge(u='a', v='b')

    a list of edges,

    >>> G.add_edges_from(ebunch=[('a', 'b'), ('b', 'c')])

    If some edges connect nodes not yet in the model, the nodes
    are added automatically. There are no errors when adding
    nodes or edges that already exist.

    **Shortcuts:**

    Many common graph features allow python syntax for speed reporting.

    >>> 'a' in G     # check if node in graph
    True
    >>> len(G)  # number of nodes in graph
    3
    """

    def __init__(self, ebunch=None, latents=set()):
        super(DAG, self).__init__(ebunch)
        self.latents = set(latents)
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

    def add_node(self, node, weight=None, latent=False):
        """
        Adds a single node to the Graph.

        Parameters
        ----------
        node: str, int, or any hashable python object.
            The node to add to the graph.

        weight: int, float
            The weight of the node.

        latent: boolean (default: False)
            Specifies whether the variable is latent or not.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG()
        >>> G.add_node(node='A')
        >>> sorted(G.nodes())
        ['A']

        Adding a node with some weight.
        >>> G.add_node(node='B', weight=0.3)

        The weight of these nodes can be accessed as:
        >>> G.nodes['B']
        {'weight': 0.3}
        >>> G.nodes['A']
        {'weight': None}
        """

        # Check for networkx 2.0 syntax
        if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], dict):
            node, attrs = node
            if attrs.get("weight", None) is not None:
                attrs["weight"] = weight
        else:
            attrs = {"weight": weight}

        if latent:
            self.latents.add(node)

        super(DAG, self).add_node(node, weight=weight)

    def add_nodes_from(self, nodes, weights=None, latent=False):
        """
        Add multiple nodes to the Graph.

        **The behviour of adding weights is different than in networkx.

        Parameters
        ----------
        nodes: iterable container
            A container of nodes (list, dict, set, or any hashable python
            object).

        weights: list, tuple (default=None)
            A container of weights (int, float). The weight value at index i
            is associated with the variable at index i.

        latent: list, tuple (default=False)
            A container of boolean. The value at index i tells whether the
            node at index i is latent or not.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG()
        >>> G.add_nodes_from(nodes=['A', 'B', 'C'])
        >>> G.nodes()
        NodeView(('A', 'B', 'C'))

        Adding nodes with weights:
        >>> G.add_nodes_from(nodes=['D', 'E'], weights=[0.3, 0.6])
        >>> G.nodes['D']
        {'weight': 0.3}
        >>> G.nodes['E']
        {'weight': 0.6}
        >>> G.nodes['A']
        {'weight': None}
        """
        nodes = list(nodes)

        if isinstance(latent, bool):
            latent = [latent] * len(nodes)

        if weights:
            if len(nodes) != len(weights):
                raise ValueError(
                    "The number of elements in nodes and weights" "should be equal."
                )
            for index in range(len(nodes)):
                self.add_node(
                    node=nodes[index], weight=weights[index], latent=latent[index]
                )
        else:
            for index in range(len(nodes)):
                self.add_node(node=nodes[index], latent=latent[index])

    def add_edge(self, u, v, weight=None):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

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
        >>> G.add_nodes_from(nodes=['Alice', 'Bob', 'Charles'])
        >>> G.add_edge(u='Alice', v='Bob')
        >>> G.nodes()
        NodeView(('Alice', 'Bob', 'Charles'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob')])

        When the node is not already present in the graph:
        >>> G.add_edge(u='Alice', v='Ankur')
        >>> G.nodes()
        NodeView(('Alice', 'Ankur', 'Bob', 'Charles'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob'), ('Alice', 'Ankur')])

        Adding edges with weight:
        >>> G.add_edge('Ankur', 'Maria', weight=0.1)
        >>> G.edge['Ankur']['Maria']
        {'weight': 0.1}
        """
        super(DAG, self).add_edge(u, v, weight=weight)

    def add_edges_from(self, ebunch, weights=None):
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
        >>> G.add_nodes_from(nodes=['Alice', 'Bob', 'Charles'])
        >>> G.add_edges_from(ebunch=[('Alice', 'Bob'), ('Bob', 'Charles')])
        >>> G.nodes()
        NodeView(('Alice', 'Bob', 'Charles'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob'), ('Bob', 'Charles')])

        When the node is not already in the model:
        >>> G.add_edges_from(ebunch=[('Alice', 'Ankur')])
        >>> G.nodes()
        NodeView(('Alice', 'Bob', 'Charles', 'Ankur'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob'), ('Bob', 'Charles'), ('Alice', 'Ankur')])

        Adding edges with weights:
        >>> G.add_edges_from([('Ankur', 'Maria'), ('Maria', 'Mason')],
        ...                  weights=[0.3, 0.5])
        >>> G.edge['Ankur']['Maria']
        {'weight': 0.3}
        >>> G.edge['Maria']['Mason']
        {'weight': 0.5}
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
                self.add_edge(edge[0], edge[1])

    def get_parents(self, node):
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
        >>> G = DAG(ebunch=[('diff', 'grade'), ('intel', 'grade')])
        >>> G.get_parents(node='grade')
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
        >>> G = DAG(ebunch=[('diff', 'grade'), ('intel', 'grade')])
        >>> moral_graph = G.moralize()
        >>> moral_graph.edges()
        EdgeView([('intel', 'grade'), ('intel', 'diff'), ('grade', 'diff')])
        """
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
        >>> graph = DAG([('A', 'B'), ('B', 'C'), ('B', 'D')])
        >>> graph.get_leaves()
        ['C', 'D']
        """
        return [node for node, out_degree in self.out_degree_iter() if out_degree == 0]

    def out_degree_iter(self, nbunch=None, weight=None):
        if nx.__version__.startswith("1"):
            return super(DAG, self).out_degree_iter(nbunch, weight)
        else:
            return iter(self.out_degree(nbunch, weight))

    def in_degree_iter(self, nbunch=None, weight=None):
        if nx.__version__.startswith("1"):
            return super(DAG, self).in_degree_iter(nbunch, weight)
        else:
            return iter(self.in_degree(nbunch, weight))

    def get_roots(self):
        """
        Returns a list of roots of the graph.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> graph = DAG([('A', 'B'), ('B', 'C'), ('B', 'D'), ('E', 'B')])
        >>> graph.get_roots()
        ['A', 'E']
        """
        return [
            node for node, in_degree in dict(self.in_degree()).items() if in_degree == 0
        ]

    def get_children(self, node):
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
        >>> g = DAG(ebunch=[('A', 'B'), ('C', 'B'), ('B', 'D'),
                                      ('B', 'E'), ('B', 'F'), ('E', 'G')])
        >>> g.get_children(node='B')
        ['D', 'E', 'F']
        """
        return list(self.successors(node))

    def get_independencies(self, latex=False, include_latents=False):
        """
        Computes independencies in the DAG, by checking d-seperation.

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
        >>> chain = DAG([('X', 'Y'), ('Y', 'Z')])
        >>> chain.get_independencies()
        (X \u27C2 Z | Y)
        (Z \u27C2 X | Y)
        """
        nodes = set(self.nodes())
        if not include_latents:
            nodes = set(self.nodes()) - self.latents

        independencies = Independencies()
        for start in nodes:
            if not include_latents:
                rest = set(self.nodes()) - {start} - self.latents
            else:
                rest = set(self.nodes()) - {start}

            for r in range(len(rest)):
                for observed in itertools.combinations(rest, r):
                    d_seperated_variables = (
                        rest
                        - set(observed)
                        - set(
                            self.active_trail_nodes(
                                start,
                                observed=observed,
                                include_latents=include_latents,
                            )[start]
                        )
                    )
                    if d_seperated_variables:
                        independencies.add_assertions(
                            [start, d_seperated_variables, observed]
                        )
        independencies.reduce()

        if not latex:
            return independencies
        else:
            return independencies.latex_string()

    def local_independencies(self, variables):
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
        >>> student.add_edges_from([('diff', 'grade'), ('intel', 'grade'),
        >>>                         ('grade', 'letter'), ('intel', 'SAT')])
        >>> ind = student.local_independencies('grade')
        >>> ind
        (grade \u27C2 SAT | diff, intel)
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

    def is_iequivalent(self, model):
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
        >>> G.add_edges_from([('V', 'W'), ('W', 'X'),
        ...                   ('X', 'Y'), ('Z', 'Y')])
        >>> G1 = DAG()
        >>> G1.add_edges_from([('W', 'V'), ('X', 'W'),
        ...                    ('X', 'Y'), ('Z', 'Y')])
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

    def get_immoralities(self):
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
        >>> student.add_edges_from([('diff', 'grade'), ('intel', 'grade'),
        ...                         ('intel', 'SAT'), ('grade', 'letter')])
        >>> student.get_immoralities()
        {('diff', 'intel')}
        """
        immoralities = set()
        for node in self.nodes():
            for parents in itertools.combinations(self.predecessors(node), 2):
                if not self.has_edge(parents[0], parents[1]) and not self.has_edge(
                    parents[1], parents[0]
                ):
                    immoralities.add(tuple(sorted(parents)))
        return immoralities

    def is_dconnected(self, start, end, observed=None):
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

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> student = DAG()
        >>> student.add_nodes_from(['diff', 'intel', 'grades', 'letter', 'sat'])
        >>> student.add_edges_from([('diff', 'grades'), ('intel', 'grades'), ('grades', 'letter'),
        ...                         ('intel', 'sat')])
        >>> student.is_dconnected('diff', 'intel')
        False
        >>> student.is_dconnected('grades', 'sat')
        True
        """
        if end in self.active_trail_nodes(start, observed)[start]:
            return True
        else:
            return False

    def minimal_dseparator(self, start, end):
        """
        Finds the minimal d-separating set for `start` and `end`.

        Parameters
        ----------
        start: node
            The first node.

        end: node
            The second node.

        Examples
        --------
        >>> dag = DAG([('A', 'B'), ('B', 'C')])
        >>> dag.minimal_dseparator(start='A', end='C')
        {'B'}

        References
        ----------
        [1] Algorithm 4, Page 10: Tian, Jin, Azaria Paz, and Judea Pearl. Finding minimal d-separators. Computer Science Department, University of California, 1998.
        """
        if (end in self.neighbors(start)) or (start in self.neighbors(end)):
            raise ValueError(
                "No possible separators because start and end are adjacent"
            )
        an_graph = self.get_ancestral_graph([start, end])
        separator = set(
            itertools.chain(self.predecessors(start), self.predecessors(end))
        )
        # If any of the parents were latents, take the latent's parent
        while len(separator.intersection(self.latents)) != 0:
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

    def get_markov_blanket(self, node):
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
        >>> G = DAG([('x', 'y'), ('z', 'y'), ('y', 'w'), ('y', 'v'), ('u', 'w'),
                               ('s', 'v'), ('w', 't'), ('w', 'm'), ('v', 'n'), ('v', 'q')])
        >>> G.get_markov_blanket('y')
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

    def active_trail_nodes(self, variables, observed=None, include_latents=False):
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

    def _get_ancestors_of(self, nodes):
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
        >>> model = DAG([('D', 'G'), ('I', 'G'), ('G', 'L'),
        ...                        ('I', 'L')])
        >>> model._get_ancestors_of('G')
        {'D', 'G', 'I'}
        >>> model._get_ancestors_of(['G', 'I'])
        {'D', 'G', 'I'}
        """
        if not isinstance(nodes, (list, tuple)):
            nodes = [nodes]

        for node in nodes:
            if node not in self.nodes():
                raise ValueError(f"Node {node} not in not in graph")

        ancestors_list = set()
        nodes_list = set(nodes)
        while nodes_list:
            node = nodes_list.pop()
            if node not in ancestors_list:
                nodes_list.update(self.predecessors(node))
            ancestors_list.add(node)
        return ancestors_list

    # TODO: Commented out till the method is implemented.
    #     def to_pdag(self):
    #         """
    #         Returns the PDAG (the equivalence class of DAG; also known as CPDAG) of the DAG.
    #
    #         Returns
    #         -------
    #         Partially oriented DAG: pgmpy.base.PDAG
    #             An instance of pgmpy.base.PDAG.
    #
    #         Examples
    #         --------
    #
    #         """
    #         pass

    def do(self, nodes, inplace=False):
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
        >>> graph.add_edges_from([('X', 'A'),
        ...                       ('A', 'Y'),
        ...                       ('A', 'B')])
        >>> # Applying the do-operator will return a new DAG with the desired structure.
        >>> graph_do_A = graph.do('A')
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
                f"Nodes not found in the model: {set(nodes) - set(self.nodes)}"
            )

        for node in nodes:
            parents = list(dag.predecessors(node))
            for parent in parents:
                dag.remove_edge(parent, node)
        return dag

    def get_ancestral_graph(self, nodes):
        """
        Returns the ancestral graph of the given `nodes`. The ancestral graph only
        contains the nodes which are ancestors of atleast one of the variables in
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
        >>> dag = DAG([('A', 'C'), ('B', 'C'), ('D', 'A'), ('D', 'B')])
        >>> anc_dag = dag.get_ancestral_graph(nodes=['A', 'B'])
        >>> anc_dag.edges()
        OutEdgeView([('D', 'A'), ('D', 'B')])
        """
        return self.subgraph(nodes=self._get_ancestors_of(nodes=nodes))

    def to_daft(
        self,
        node_pos="circular",
        latex=True,
        pgm_params={},
        edge_params={},
        node_params={},
    ):
        """
        Returns a daft (https://docs.daft-pgm.org/en/latest/) object which can be rendered for
        publication quality plots. The returned object's render method can be called to see the plots.

        Parameters
        ----------
        node_pos: str or dict (default: circular)
            If str: Must be one of the following: circular, kamada_kawai, planar, random, shell, sprint,
                spectral, spiral. Please refer: https://networkx.org/documentation/stable//reference/drawing.html#module-networkx.drawing.layout for details on these layouts.

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

        Returns
        -------
        Daft object: daft.PGM object
            Daft object for plotting the DAG.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> dag = DAG([('a', 'b'), ('b', 'c'), ('d', 'c')])
        >>> dag.to_daft(node_pos={'a': (0, 0), 'b': (1, 0), 'c': (2, 0), 'd': (1, 1)})
        <daft.PGM at 0x7fc756e936d0>
        >>> dag.to_daft(node_pos="circular")
        <daft.PGM at 0x7f9bb48c5eb0>
        >>> dag.to_daft(node_pos="circular", pgm_params={'observed_style': 'inner'})
        <daft.PGM at 0x7f9bb48b0bb0>
        >>> dag.to_daft(node_pos="circular",
        ...             edge_params={('a', 'b'): {'label': 2}},
        ...             node_params={'a': {'shape': 'rectangle'}})
        <daft.PGM at 0x7f9bb48b0bb0>
        """
        try:
            from daft import PGM
        except ImportError as e:
            raise ImportError(
                "Package daft required. Please visit: https://docs.daft-pgm.org/en/latest/ for installation instructions."
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
            if node_pos not in supported_layouts.keys():
                raise ValueError(
                    "Unknown node_pos argument. Please refer docstring for accepted values"
                )
            else:
                node_pos = supported_layouts[node_pos](self)
        elif isinstance(node_pos, dict):
            for node in self.nodes():
                if node not in node_pos.keys():
                    raise ValueError(f"No position specified for {node}.")
        else:
            raise ValueError(
                "Argument node_pos not valid. Please refer to the docstring."
            )

        daft_pgm = PGM(**pgm_params)
        for node in self.nodes():
            try:
                extra_params = node_params[node]
            except KeyError:
                extra_params = dict()

            if latex:
                daft_pgm.add_node(
                    node,
                    rf"${node}$",
                    node_pos[node][0],
                    node_pos[node][1],
                    observed=True,
                    **extra_params,
                )
            else:
                daft_pgm.add_node(
                    node,
                    f"{node}",
                    node_pos[node][0],
                    node_pos[node][1],
                    observed=True,
                    **extra_params,
                )

        for u, v in self.edges():
            try:
                extra_params = edge_params[(u, v)]
            except KeyError:
                extra_params = dict()
            daft_pgm.add_edge(u, v, **extra_params)

        return daft_pgm

    @staticmethod
    def get_random(n_nodes=5, edge_prob=0.5, latents=False):
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

        latents: bool (default: False)
            If True, includes latent variables in the generated DAG.

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
        adj_mat = np.random.choice(
            [0, 1], size=(n_nodes, n_nodes), p=[1 - edge_prob, edge_prob]
        )

        # Step 2: Use the upper triangular part of the matrix as adjacency.
        nodes = list(range(n_nodes))
        edges = nx.convert_matrix.from_numpy_array(
            np.triu(adj_mat, k=1), create_using=nx.DiGraph
        ).edges()

        dag = DAG(edges)
        dag.add_nodes_from(nodes)
        if latents:
            dag.latents = set(
                np.random.choice(
                    dag.nodes(), np.random.randint(low=0, high=len(dag.nodes()))
                )
            )
        return dag

    def to_graphviz(self):
        """
        Retuns a pygraphviz object for the DAG. pygraphviz is useful for
        visualizing the network structure.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model
        >>> model = get_example_model('alarm')
        >>> model.to_graphviz()
        <AGraph <Swig Object of type 'Agraph_t *' at 0x7fdea4cde040>>
        >>> model.draw('model.png', prog='neato')
        """
        return nx.nx_agraph.to_agraph(self)


class PDAG(nx.DiGraph):
    """
    Class for representing PDAGs (also known as CPDAG). PDAGs are the equivalence classes of
    DAGs and contain both directed and undirected edges.

    Note: In this class, undirected edges are represented using two edges in both direction i.e.
    an undirected edge between X - Y is represented using X -> Y and X <- Y.
    """

    def __init__(self, directed_ebunch=[], undirected_ebunch=[], latents=[]):
        """
        Initializes a PDAG class.

        Parameters
        ----------
        directed_ebunch: list, array-like of 2-tuples
            List of directed edges in the PDAG.

        undirected_ebunch: list, array-like of 2-tuples
            List of undirected edges in the PDAG.

        latents: list, array-like
            List of nodes which are latent variables.

        Returns
        -------
        An instance of the PDAG object.

        Examples
        --------
        """
        super(PDAG, self).__init__(
            directed_ebunch
            + undirected_ebunch
            + [(Y, X) for (X, Y) in undirected_ebunch]
        )
        self.latents = set(latents)
        self.directed_edges = set(directed_ebunch)
        self.undirected_edges = set(undirected_ebunch)
        # TODO: Fix the cycle issue
        # import pdb; pdb.set_trace()
        # try:
        #     # Filter out undirected edges as they also form a cycle in
        #     # themself when represented using directed edges.
        #     cycles = filter(lambda t: len(t) > 2, nx.simple_cycles(self))
        #     if cycles:
        #         out_str = "Cycles are not allowed in a PDAG. "
        #         out_str += "The following path forms a loop: "
        #         out_str += "".join(["({u},{v}) ".format(u=u, v=v) for (u, v) in cycles])
        #         raise ValueError(out_str)
        # except nx.NetworkXNoCycle:
        #     pass

    def copy(self):
        """
        Returns a copy of the object instance.

        Returns
        -------
        Copy of PDAG: pgmpy.dag.PDAG
            Returns a copy of self.
        """
        return PDAG(
            directed_ebunch=list(self.directed_edges.copy()),
            undirected_ebunch=list(self.undirected_edges.copy()),
            latents=self.latents,
        )

    def to_dag(self, required_edges=[]):
        """
        Returns one possible DAG which is represented using the PDAG.

        Parameters
        ----------
        required_edges: list, array-like of 2-tuples
            The list of edges that should be included in the DAG.

        Returns
        -------
        Returns an instance of DAG.

        Examples
        --------

        """
        # Add required edges if it doesn't form a new v-structure or an opposite edge
        # is already present in the network.
        dag = DAG()
        # Add all the nodes and the directed edges
        dag.add_nodes_from(self.nodes())
        dag.add_edges_from(self.directed_edges)
        dag.latents = self.latents

        pdag = self.copy()
        while pdag.number_of_nodes() > 0:
            # find node with (1) no directed outgoing edges and
            #                (2) the set of undirected neighbors is either empty or
            #                    undirected neighbors + parents of X are a clique
            found = False
            for X in pdag.nodes():
                directed_outgoing_edges = set(pdag.successors(X)) - set(
                    pdag.predecessors(X)
                )
                undirected_neighbors = set(pdag.successors(X)) & set(
                    pdag.predecessors(X)
                )
                neighbors_are_clique = all(
                    (
                        pdag.has_edge(Y, Z)
                        for Z in pdag.predecessors(X)
                        for Y in undirected_neighbors
                        if not Y == Z
                    )
                )

                if not directed_outgoing_edges and (
                    not undirected_neighbors or neighbors_are_clique
                ):
                    found = True
                    # add all edges of X as outgoing edges to dag
                    for Y in pdag.predecessors(X):
                        dag.add_edge(Y, X)
                    pdag.remove_node(X)
                    break

            if not found:
                logger.warning(
                    "PDAG has no faithful extension (= no oriented DAG with the "
                    + "same v-structures as PDAG). Remaining undirected PDAG edges "
                    + "oriented arbitrarily."
                )
                for X, Y in pdag.edges():
                    if not dag.has_edge(Y, X):
                        try:
                            dag.add_edge(X, Y)
                        except ValueError:
                            pass
                break
        return dag

    def to_graphviz(self):
        """
        Retuns a pygraphviz object for the DAG. pygraphviz is useful for
        visualizing the network structure.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model
        >>> model = get_example_model('alarm')
        >>> model.to_graphviz()
        <AGraph <Swig Object of type 'Agraph_t *' at 0x7fdea4cde040>>
        """
        return nx.nx_agraph.to_agraph(self)
