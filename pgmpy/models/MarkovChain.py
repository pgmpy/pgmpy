#!/usr/bin/env python3
import numpy as np
import networkx as nx

from pgmpy.utils import sample_discrete


class MarkovChainMonteCarlo(nx.DiGraph):
    """
    Class to represent a Markov Chain, along with methods to simulate a run.

    Examples:
    ---------
    Create an empty Markov Chain:
    >>> from pgmpy.models import MarkovChainMonteCarlo as MCMC
    >>> model = MCMC()

    And then add edges to it
    >>> model.add_weighted_edges_from([(-1, 0, 0.25), (-1, -1, 0.75), (0, -1, 0.25), (0, 1, 0.25),
    ...     (0, 0, 0.5), (1, 0, 0.25), (1, 1, 0.75)])

    Or directly create a Markov Chain from a list of edges
    >>> model = MCMC([(-1, 0, 0.25), (-1, -1, 0.75), (0, -1, 0.25), (0, 1, 0.25),
    ...     (0, 0, 0.5), (1, 0, 0.25), (1, 1, 0.75)])

    Set a start state
    >>> model.set_start_state(0)

    Check if the Markov Chain and the transition probabilities are valid
    >>> model.check_markov_chain()
    True

    Get the list of added edges
    >>> model.edge
    {-1: {-1: {'weight': 0.75}, 0: {'weight': 0.25}},
    0: {-1: {'weight': 0.25}, 0: {'weight': 0.5}, 1: {'weight': 0.25}},
    1: {0: {'weight': 0.25}, 1: {'weight': 0.75}}}

    Sample from it
    >>> model.sample(5)
    array([ 0., -1.,  0.,  1.,  1.])

    """
    def __init__(self, ebunch=None, start_state=None):
        """
        Parameters:
        -----------
        ebunch: list of 3-tuples (A, B, w) representing the directed weighted edge A->B with weight w.
        start_state: the node of the Markov Chain that will be used as the starting state when simulating a
            run/sampling.
        """
        super().__init__()
        if ebunch is not None:
            super().add_weighted_edges_from(ebunch)
        self._weights = {node: 0 for node in self.nodes()}
        self.start_state = None
        if start_state is not None:
            if start_state not in self.nodes():
                raise ValueError('Start state should be an existing node in the Markov Chain')
            self.weights[start_state] = 1
            self.start_state = start_state

    def add_node(self, node, **kwargs):
        """
        Add a single node to the Markov Chain.

        Parameters:
        -----------
        node: node
            A node can be any hashable Python object.

        Examples
        --------
        >>> from pgmpy.models import MarkovChainMonteCarlo as MCMC
        >>> model = MCMC()
        >>> model.add_node(0)
        """
        super().add_node(node, **kwargs)
        self._weights[node] = 0

    def add_weighted_edges_from(self, ebunch, weight='weight', **attr):
        """
        Add all the edges (weighted) in ebunch.

        If nodes referred in the ebunch are not already present, they
        will be automatically added. Node names should be strings.

        Parameters:
        -----------
        ebunch: container of edges
            Each edge given in the container will be added to the graph.
            The edges must be given as 3-tuples (u, v, weight).

        Examples:
        ---------
        >>> from pgmpy.models import MarkovChainMonteCarlo as MCMC
        >>> model = MCMC()
        >>> model.add_weighted_edges_from([(0, 1, 0.25), (0, -1, 0.5)])
        """
        self.reset_weights()
        super().add_weighted_edges_from(ebunch, weight, **attr)
        self._weights = {node: 0 for node in self.nodes()}

    def set_start_state(self, node):
        """
        Set a start state for sampling from the Markov Chain.

        Parameters:
        -----------
        node: node
            Must be an existing node in the Markov Chain.

        Examples:
        ---------
        >>> from pgmpy.models import MarkovChainMonteCarlo as MCMC
        >>> models = MCMC()
        >>> models.add_node(0)
        >>> models.set_start_state(0)
        """
        if node not in self.nodes():
            raise ValueError('Start state should be an existing node in the Markov Chain')
        self.reset_weights()
        self._weights[node] = 1
        self.start_state = node

    def reset_weights(self):
        """
        Reset the initial probability distribution on the states of the model.

        Examples:
        ---------
        >>> from pgmpy.models import MarkovChainMonteCarlo as MCMC
        >>> model = MCMC()
        >>> model.reset_weights()
        """
        for node in self.weights:
            self._weights[node] = 0
        self.start_state = None

    def check_markov_chain(self):
        """
        Checks if the weights on the edges of the model define a transition probability table. Returns True if all
        the weights of outgoing edges are in the range [0, 1] and sum to 1 at each node.

        Examples:
        ---------
        >>> from pgmpy.models import MarkovChainMonteCarlo as MCMC
        >>> model = MCMC([(0, 1, 0.25), (0, -1, 0.25), (0, 0, 0.5), (1, 1, 1), (-1, -1, 1)])
        >>> model.check_markov_chain()
        True
        """
        for node, ebunch in self.edge.items():
            edge_weights = map(lambda x: x['weight'], ebunch.values())
            sum = 0
            for weight in edge_weights:
                sum += weight
                if weight < 0 or weight > 1.0:
                    return False
            if not np.allclose(sum, 1.0):
                return False
        return True

    @property
    def weights(self):
        """
        Returns the probability distribution on the states of the Markov Chain.

        >>> from pgmpy.models import MarkovChainMonteCarlo as MCMC
        >>> model = MCMC([(0, 1, 0.25), (0, -1, 0.25), (0, 0, 0.5), (1, 1, 1), (-1, -1, 1)])
        >>> model.set_start_state(0)
        >>> model.weights
        {-1: 0, 0: 1, 1: 0}
        """
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        """
        Assigns a probability distribution on the states of the Markov Chain.

        Parameters:
        -----------
        new_weights: dict
            key set of the dict must same as the set of nodes of the Markov Chain.
            values must lie in the range [0, 1] and must sum to 1.

        Examples:
        ---------
        >>> from pgmpy.models import MarkovChainMonteCarlo as MCMC
        >>> model = MCMC([(0, 1, 0.25), (0, -1, 0.25), (0, 0, 0.5), (1, 1, 1), (-1, -1, 1)])
        >>> model.weights = {0: 0.1, -1: 0.5, 1: 0.4}
        """
        if not isinstance(new_weights, dict):
            raise ValueError('new weights must be a dict')
        if set(new_weights.keys()) != set(self.nodes()):
            raise ValueError('Must assign weights to all and only the states of the Markov Chain')
        for weight in new_weights.values():
            if weight < 0 or weight > 1:
                raise ValueError('All weights must lie in the interval [0, 1]')
        self._weights = new_weights

    def sample(self, size=1):
        """
        Generate a sample from the Markov Chain by simulating a random walk.
        'Probabilistic Graphical Model Principles and Techniques', Koller and
        Friedman, Algorithm 12.5 pp 509.

        Parameters:
        -----------
        size: int: the number of samples to be generated.

        Examples:
        ---------
        >>> from pgmpy.models import MarkovChainMonteCarlo as MCMC
        >>> model = MCMC([(0, 1, 0.25), (0, -1, 0.25), (0, 0, 0.5), (1, 1, 1), (-1, -1, 1)])
        >>> model.set_start_state(0)
        >>> model.sample(5)
        array([ 0.,  0.,  0.,  0.,  1.])
        """
        sampled = np.empty(size)
        sampled[0] = sample_discrete(list(self._weights.keys()), list(self._weights.values()))[0]
        for i in range(size - 1):
            values = np.array(list(self.edge[sampled[i]].keys()))
            weights = list(map(lambda x: x['weight'], self.edge[sampled[i]].values()))
            sampled[i + 1] = sample_discrete(values, weights)[0]
        return sampled
