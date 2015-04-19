from itertools import combinations

import numpy as np

from pgmpy.models import BayesianModel


class BaseEliminationOrder:
    def __init__(self, model):
        if not isinstance(model, BayesianModel):
            raise ValueError("Model should be a BayesianModel instance")
        self.bayesian_model = model
        self.moralized_model = self.bayesian_model.moralize()

    def cost(self, node):
        return 0

    def get_elimination_order(self, nodes):
        ordering = []
        while nodes:
            scores = {node: self.cost(node) for node in nodes}
            min_score_node = min(scores, key=scores.get)
            ordering.append(min_score_node)
            nodes.remove(min_score_node)
        return ordering

    def fill_in_edges(self, node):
        """
        Return edges needed to be added to the graph if a node is removed.

        Parameters
        ----------
        node: one node.
            Node to be removed from the graph.
        """
        return combinations(self.bayesian_model.neighbours(node), 2)


class WeightedMinFill(BaseEliminationOrder):
    def cost(self, node):
        edges = self.moralized_model.fill_in_edges(node)
        return sum([self.bayesian_model.get_cardinality(edge[0]) *
                    self.bayesian_model.get_cardinality(edge[1]) for edge in edges])


class MinNeighbours(BaseEliminationOrder):
    def cost(self, node):
        return len(self.moralized_model.neighbors(node))


class MinWeight(BaseEliminationOrder):
    def cost(self, node):
        return np.prod([self.bayesian_model.get_cardinality(neig_node) for neig_node in
                        self.moralized_model.neighbors(node)])


class MinFill(BaseEliminationOrder):
    def cost(self, node):
        return len(self.moralized_model.fill_in_edges(node))

# class EliminationOrdering:
#
#     """
#     This class implements tools to help finding good elimination orderings.
#     Besically, a greedy algorithm scores each variables currently in the
#     moralized graph, pick one, removing it, and score the graph again.
#     The scoring is done by a cost function. From ('Probabilistic Graphical
#     Model Principles and Techniques' - Koller and Friedman), "empirical results
#     show that these heuristic algorithms [min-neighbors, min-fill, min-weight,
#     weighted-min-fill] perform surprisingly well in practice. Generally,
#     Min-Fill and Weighted-Min-Fill tend to work better on more problems".
#
#     Parameters
#     ----------
#     model: input bayesian model
#         A bayesian model to be moralized (an undirected graph where the parents
#         of a v-structures are connected).
#     """
#
#     def __init__(self, model):
#         self.bayesian_model = model
#         if not isinstance(model, BayesianModel):
#             raise ValueError("EliminationOrdering should"
#                              " receive a Bayesian model.")
#         self.moralized_graph = self.bayesian_model.moralize()
#
#     def find_elimination_ordering(self, nodes, cost_func):
#         """
#         A greedy algorithm that eliminates a less score variable per time.
#
#         Parameters
#         ----------
#         nodes: list with nodes
#             nodes to be eliminated
#         cost_func: a function
#             A function which can compute the score of a given
#         """
#         ordering = []
#         while nodes:
#             scorings = [{"node": v, "score": cost_func(v)} for v in nodes]
#             scorings_in_order = sorted(scorings, key=lambda k: k['score'])
#             ordering.append(scorings_in_order[0]["node"])
#             nodes.remove(scorings_in_order[0]["node"])
#         return ordering
#
#     def weighted_min_fill(self, node):
#         """
#         The score of a node is the sum of weights of the edges that need to
#         be added to the graph due to its elimination, where a weight of an
#         edge is the product of the weights, domain cardinality, of its
#         constituent vertices.
#
#         Parameters
#         ----------
#             node: a string
#                 the variable to be scored
#         """
#         edges = self.moralized_graph.fill_in_edges(node)
#         sum_weight = 0
#         for edge in edges:
#             sum_weight = sum_weight + \
#                 self.bayesian_model.get_cardinality(
#                     edge[0]) * self.bayesian_model.get_cardinality(edge[1])
#         return sum_weight
#
#     def min_neighbors(self, node):
#         """
#         The cost of a node is the number of neighbors it has in
#         the current graph.
#
#         Parameters
#         ----------
#             node: a string
#                 the variable to be scored
#         """
#         return len(self.moralized_graph.neighbors(node))
#
#     def min_weight(self, node):
#         """
#         The cost of a vertex is the product of weights, domain cardinality,
#         of its neighbors.
#
#         Parameters
#         ----------
#             node: a string
#                 the variable to be scored
#         """
#         product = 1
#         for neighbor in self.moralized_graph.neighbors(node):
#             product = product * self.bayesian_model.get_cardinality(neighbor)
#         return product
#
#     def min_fill(self, node):
#         """
#         The cost of a node is the number of edges that need to be added
#         (fill in edges) to the graph due to its elimination
#
#         Parameters
#         ----------
#             node: a string
#                 the variable to be scored
#         """
#         return len(self.moralized_graph.fill_in_edges(node))
