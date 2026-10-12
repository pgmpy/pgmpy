import networkx as nx
import numpy as np

from pgmpy.causal_explainability._base import _BaseAttribution
from pgmpy.causal_explainability._shapley import ShapleyEngine


class AnomalyAttribution(_BaseAttribution):
    """Attributes an observed anomaly at a target node to upstream mechanisms.

    Parameters
    ----------
    anomaly_scorer : callable(observed, expected_samples) -> float, optional
        Default: squared normalized residual.

    References
    ----------
    [1] Budhathoki, K., Janzing, D., Blöbaum, P., & Ng, H. (2022). Causal
        structure-based root cause analysis of outliers. ICML 2022.
    """

    def __init__(self, anomaly_scorer=None):
        self.anomaly_scorer = anomaly_scorer

    def _attribute(self, model, data, target, **kwargs):
        observation = kwargs["observation"]
        n_samples = kwargs.get("n_samples", 500)

        ancestors = list(nx.ancestors(model, target) | {target})
        topo_order = [n for n in nx.topological_sort(model) if n in ancestors]

        from pgmpy.models import LinearGaussianBayesianNetwork

        if isinstance(model, LinearGaussianBayesianNetwork) and self.anomaly_scorer is None:
            return self._attribute_lgbn(model, target, observation, topo_order, n_samples)
        return self._attribute_general(model, data, target, observation, topo_order, n_samples)

    def _attribute_lgbn(self, model, target, observation, topo_order, n_samples):
        # Compute residuals for each node
        residuals = {}
        for node in topo_order:
            cpd = model.get_cpds(node)
            expected = cpd.beta[0]
            for j, p in enumerate(cpd.evidence):
                expected += cpd.beta[j + 1] * observation[p]
            residuals[node] = observation[node] - expected

        # Compute total effect of each node's noise on target
        total_effect = dict.fromkeys(topo_order, 0.0)
        total_effect[target] = 1.0
        node_set = set(topo_order)
        for node in reversed(topo_order):
            if node == target:
                continue
            effect = 0.0
            for child in model.successors(node):
                if child not in node_set:
                    continue
                cpd = model.get_cpds(child)
                parent_idx = cpd.evidence.index(node)
                beta = cpd.beta[parent_idx + 1]
                effect += beta * total_effect[child]
            total_effect[node] = effect

        target_cpd = model.get_cpds(target)
        rng = np.random.default_rng(None)

        def value_fn(coalition):
            active_nodes = {topo_order[i] for i in coalition}
            samples = np.zeros(n_samples)
            for k in range(n_samples):
                val = 0.0
                for node in topo_order:
                    if node in active_nodes:
                        val += total_effect[node] * residuals[node]
                    else:
                        val += total_effect[node] * rng.normal(0, model.get_cpds(node).std)
                samples[k] = val**2 / target_cpd.std**2
            return np.mean(samples)

        engine = ShapleyEngine(n_players=len(topo_order), value_function=value_fn, method="auto")
        indexed_result = engine.compute()
        return {topo_order[i]: v for i, v in indexed_result.items()}

    def _attribute_general(self, model, data, target, observation, topo_order, n_samples):
        rng = np.random.default_rng(None)

        from pgmpy.models import LinearGaussianBayesianNetwork

        def _get_expected_samples(node, parent_vals, n):
            cpd = model.get_cpds(node)
            if isinstance(model, LinearGaussianBayesianNetwork):
                mean = cpd.beta[0]
                for j, p in enumerate(cpd.evidence):
                    mean += cpd.beta[j + 1] * parent_vals[p]
                return rng.normal(mean, cpd.std, n)
            else:
                probs_col = self._get_cpd_column(model, cpd, parent_vals)
                return rng.choice(len(probs_col), size=n, p=probs_col)

        scorer = self.anomaly_scorer
        if scorer is None:

            def scorer(observed, expected_samples):
                mean = np.mean(expected_samples)
                var = np.var(expected_samples) + 1e-10
                return (observed - mean) ** 2 / var

        def value_fn(coalition):
            active_nodes = {topo_order[i] for i in coalition}
            scores = []
            for _ in range(max(1, n_samples // 10)):
                node_values = {}
                for node in topo_order:
                    parent_vals = {p: node_values[p] for p in model.get_parents(node) if p in node_values}
                    if node in active_nodes:
                        node_values[node] = observation[node]
                    else:
                        samples = _get_expected_samples(node, parent_vals, 1)
                        node_values[node] = samples[0]

                parent_vals_target = {p: node_values[p] for p in model.get_parents(target) if p in node_values}
                expected = _get_expected_samples(target, parent_vals_target, 50)
                scores.append(scorer(node_values[target], expected))
            return np.mean(scores)

        engine = ShapleyEngine(n_players=len(topo_order), value_function=value_fn, method="auto")
        indexed_result = engine.compute()
        return {topo_order[i]: v for i, v in indexed_result.items()}

    def _get_cpd_column(self, model, cpd, parent_vals):
        state_names = model.states
        cards = [len(state_names[p]) for p in cpd.variables[1:]]
        parent_indices = []
        for p in cpd.variables[1:]:
            p_states = state_names[p]
            p_val = parent_vals.get(p, list(p_states)[0])
            p_idx = list(p_states).index(p_val)
            parent_indices.append(p_idx)
        prob_table = cpd.get_values()
        idx = 0
        for j, p_idx in enumerate(parent_indices):
            stride = 1
            for k in range(j + 1, len(cards)):
                stride *= cards[k]
            idx += p_idx * stride
        return prob_table[:, idx]
