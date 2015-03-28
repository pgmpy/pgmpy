class EliminationOrdering:

    """
    This class implements tools to help finding good elimination orderings.
    Besically, a greedy algorithm scores each variables currently in the moralized graph, pick one, removing it, and score the graph again.
    The scoring is done by a cost function. From ('Probabilistic Graphical Model
        Principles and Techniques' - Koller and Friedman), "empirical results show that these heuristic algorithms [min-neighbors, min-fill, min-weight, weighted-min-fill] perform surprisingly well in practice. Generally, Min-Fill and Weighted-Min-Fill tend to work better on more problems".

    Parameters
    ----------
    bayesian_model: input bayesian model
        A bayesian model to be moralized (an undirected graph where the parents of a v-structures are connected).
    """

    def __init__(self, bayesian_model):
        self.bayesian_model = bayesian_model
        self.moralized_graph = bayesian_model.moralize()

    def find_elimination_ordering(self, nodes, cost_func):
        """
        A greedy algorithm that eliminates a less score variable per time.

        Parameters
        ----------
        nodes: list with nodes
            nodes to be eliminated
        cost_func: a function
            A function which can compute the score of a given
        """
        ordering = []
        while nodes:
            scorings = [{"node": v, "score": cost_func(v)} for v in nodes]
            sortedScorings = sorted(scorings, key=lambda k: k['score'])
            ordering.append(sortedScorings[0]["node"])
            nodes.remove(sortedScorings[0]["node"])
        return ordering

    def weighted_min_fill(self, node):
        """
        The score of a node is the sum of weights of the edges that need to be added to the graph due to its elimination, where a weight of an edge is the product of the weights — domain cardinality — of its constituent vertices.

        Parameters
        ----------
            node: a string
                the variable to be scored
        """
        edges = self.moralized_graph.fill_in_edges(node)
        sumWeight = 0
        for edge in edges:
            sumWeight = sumWeight + self.bayesian_model.get_cardinality(edge[0]) * self.bayesian_model.get_cardinality(edge[1])
        return sumWeight

    def min_neighbors(self, node):
        """
        The cost of a node is the number of neighbors it has in the current graph.

        Parameters
        ----------
            node: a string
                the variable to be scored
        """
        return len(self.moralized_graph.neighbors(node))

    def min_weight(self, node):
        """
        The cost of a vertex is the product of weights — domain cardinality — of its neighbors.

        Parameters
        ----------
            node: a string
                the variable to be scored
        """
        product = 1
        for neighbor in self.moralized_graph.neighbors(node):
            product = product * self.bayesian_model.get_cardinality(neighbor)
        return product

    def min_fill(self, node):
        """
        The cost of a node is the number of edges that need to be added (fill in edges) to the graph due to its elimination

        Parameters
        ----------
            node: a string
                the variable to be scored
        """
        return len(self.moralized_graph.fill_in_edges(node))
