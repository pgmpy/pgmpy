import numpy as np
import pandas as pd

from scipy.special import rel_entr
from typing import Any
from itertools import combinations
from math import factorial

from pgmpy.base import DAG, MAG
from pgmpy.models import LinearGaussianBayesianNetwork, DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import (
    VariableElimination,
    BeliefPropagation,
    CausalInference,
    ExactInference,
)
from pgmpy.ci_tests import get_ci_test


class RCA:
    def __init__(self, model: DAG, seed=None) -> None:
        self.model = model.copy()
        self.seed = seed

    def attribute_shift(
        self,
        baseline: pd.DataFrame,
        shifted: pd.DataFrame,
        target: str,
        n_draws: int = 1000,
    ):
        baseline_model = DiscreteBayesianNetwork(self.model.edges)
        baseline_model.fit(baseline)

        shifted_model = DiscreteBayesianNetwork(self.model.edges)
        shifted_model.fit(shifted)

        candidate_nodes = set(self.model.get_ancestors(target)) | {target}

        def build_hybrid_model(shifted_nodes):
            """Use shifted CPDs for shifted_nodes and baseline CPDs elsewhere."""
            hybrid_model = DiscreteBayesianNetwork(self.model.edges)

            cpds = [
                (
                    shifted_model.get_cpds(node)
                    if node in shifted_nodes
                    else baseline_model.get_cpds(node)
                )
                for node in self.model.nodes
            ]

            hybrid_model.add_cpds(*cpds)
            hybrid_model.check_model()

            return hybrid_model

        def target_distribution(shifted_nodes):
            hybrid_model = build_hybrid_model(shifted_nodes)

            inference = VariableElimination(hybrid_model)
            return inference.query(
                variables=[target],
                show_progress=False,
            )

        baseline_distribution = target_distribution(set())
        baseline_probabilities = baseline_distribution.values

        def value_function(shifted_nodes):

            shifted_nodes = frozenset(shifted_nodes)

            shifted_probabilities = target_distribution(shifted_nodes).values

            kl_div = np.sum(
                rel_entr(
                    shifted_probabilities,
                    baseline_probabilities,
                )
            )
            return kl_div

        return self.shapley_values(
            players=candidate_nodes,
            value_function=value_function,
        )

    @staticmethod
    def shapley_values(players, value_function):
        players = list(players)
        n_players = len(players)

        shapley_values = {}

        for player in players:
            other_players = [p for p in players if p != player]
            contribution = 0.0

            for subset_size in range(n_players):
                for subset in combinations(other_players, subset_size):
                    subset = set(subset)

                    weight = (
                        factorial(subset_size)
                        * factorial(n_players - subset_size - 1)
                        / factorial(n_players)
                    )

                    marginal_contribution = value_function(
                        subset | {player}
                    ) - value_function(subset)

                    contribution += weight * marginal_contribution

            shapley_values[player] = contribution

        return shapley_values

    def changed_mechanisms(
        self,
        baseline: pd.DataFrame,
        shifted: pd.DataFrame,
        ci_test=None,
        significance_level: float = 0.05,
    ):

        baseline_data = baseline.copy()
        shifted_data = shifted.copy()

        baseline_data["mechanism"] = 0
        shifted_data["mechanism"] = 1

        data = pd.concat(
            [baseline_data, shifted_data],
            ignore_index=True,
        )

        independence_test = get_ci_test(
            ci_test,
            data=data,
        )

        results = {}

        for node in self.model.nodes:
            parents = set(self.model.predecessors(node))

            test_result = independence_test(
                node,
                "mechanism",
                parents,
                significance_level=significance_level,
            )

            results[node] = test_result

        return results
