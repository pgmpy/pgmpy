import networkx as nx
import numpy as np

from pgmpy.causal_explainability._base import _BaseAttribution
from pgmpy.causal_explainability._shapley import ShapleyEngine


class IntrinsicCausalInfluence(_BaseAttribution):
    """Decomposes a statistical functional of a target into contributions from upstream noise terms.

    For LinearGaussianBayesianNetwork with variance functional: analytical.
    contribution_i = w_i^2 * sigma_i^2 where w_i = total effect of node i's noise on target.

    Parameters
    ----------
    functional : callable(np.ndarray) -> float, optional
        Default: np.var.
    """

    def __init__(self, functional=None):
        self.functional = functional if functional is not None else np.var

    def _attribute(self, model, data, target, **kwargs):
        from pgmpy.models import LinearGaussianBayesianNetwork

        ancestors = nx.ancestors(model, target) | {target}

        if isinstance(model, LinearGaussianBayesianNetwork) and self.functional is np.var:
            return self._analytical_lgbn(model, target, ancestors)
        return self._monte_carlo(model, data, target, ancestors, **kwargs)

    def _analytical_lgbn(self, model, target, ancestors):
        topo_order = [n for n in nx.topological_sort(model) if n in ancestors]

        # Compute total effect of each node's noise on target via back-propagation
        total_effect = dict.fromkeys(topo_order, 0.0)
        total_effect[target] = 1.0

        for node in reversed(topo_order):
            if node == target:
                continue
            effect = 0.0
            for child in model.successors(node):
                if child not in ancestors:
                    continue
                cpd = model.get_cpds(child)
                parent_idx = cpd.evidence.index(node)
                beta = cpd.beta[parent_idx + 1]
                effect += beta * total_effect[child]
            total_effect[node] = effect

        result = {}
        for node in topo_order:
            cpd = model.get_cpds(node)
            sigma = cpd.std
            result[node] = total_effect[node] ** 2 * sigma**2
        return result

    def _monte_carlo(self, model, data, target, ancestors, **kwargs):
        n_samples = kwargs.get("n_samples", 1000)
        seed = kwargs.get("seed", None)
        rng = np.random.default_rng(seed)

        topo_order = [n for n in nx.topological_sort(model) if n in ancestors]
        node_to_idx = {n: i for i, n in enumerate(topo_order)}

        from pgmpy.models import LinearGaussianBayesianNetwork

        def _simulate_target(active_noise):
            node_values = {}
            for node in topo_order:
                cpd = model.get_cpds(node)
                if isinstance(model, LinearGaussianBayesianNetwork):
                    parents = cpd.evidence
                    mean = cpd.beta[0]
                    for j, p in enumerate(parents):
                        if p in node_values:
                            mean += cpd.beta[j + 1] * node_values[p]
                    if node_to_idx[node] in active_noise:
                        noise = rng.normal(0, cpd.std)
                    else:
                        noise = 0.0
                    node_values[node] = mean + noise
            return node_values[target]

        def value_fn(coalition):
            target_samples = np.array([_simulate_target(coalition) for _ in range(n_samples)])
            return self.functional(target_samples)

        engine = ShapleyEngine(n_players=len(topo_order), value_function=value_fn, method="auto")
        indexed_result = engine.compute()
        return {topo_order[i]: v for i, v in indexed_result.items()}
