import numpy as np

from pgmpy.causal_explainability._base import _BaseAttribution
from pgmpy.causal_explainability._shapley import ShapleyEngine


class CausalFeatureRelevance(_BaseAttribution):
    """Quantifies how relevant each direct parent is for a node's mechanism.

    Uses Shapley-based decomposition to attribute a statistical functional
    of the target to its direct parents.

    Parameters
    ----------
    functional : callable(np.ndarray) -> float, optional
        Statistical functional to decompose. Default: np.var.
    level : str
        "node" for single target, "graph" for all nodes.

    References
    ----------
    [1] Janzing, D., Minorics, L., & Blöbaum, P. (2020). Feature relevance
        quantification in explainable AI: A causal problem. AISTATS 2020.
    """

    def __init__(self, functional=None, level="node"):
        self.functional = functional if functional is not None else np.var
        self.level = level

    def _attribute(self, model, data, target, **kwargs):
        if self.level == "graph":
            return {node: self._attribute_single(model, data, node, **kwargs) for node in model.nodes()}
        return self._attribute_single(model, data, target, **kwargs)

    def _attribute_single(self, model, data, target, **kwargs):
        parents = list(model.get_parents(target))
        if not parents:
            return {}

        from pgmpy.models import LinearGaussianBayesianNetwork

        if isinstance(model, LinearGaussianBayesianNetwork) and self.functional is np.var:
            return self._analytical_lgbn(model, data, target, parents)
        return self._monte_carlo(model, data, target, parents, **kwargs)

    def _analytical_lgbn(self, model, data, target, parents):
        cpd = model.get_cpds(target)
        result = {}
        for i, parent in enumerate(parents):
            beta_i = cpd.beta[i + 1]
            var_parent = np.var(data[parent].values)
            result[parent] = beta_i**2 * var_parent
        return result

    def _monte_carlo(self, model, data, target, parents, **kwargs):
        n_samples = kwargs.get("n_samples", 1000)
        seed = kwargs.get("seed", None)
        rng = np.random.default_rng(seed)

        player_to_parent = {i: p for i, p in enumerate(parents)}
        parent_data = {p: data[p].values for p in parents}
        n_data = len(data)
        cpd = model.get_cpds(target)

        from pgmpy.models import LinearGaussianBayesianNetwork

        def value_fn(coalition):
            samples = np.zeros(n_samples)
            for s_idx in range(n_samples):
                parent_vals = {}
                data_idx = rng.integers(0, n_data)
                for i, parent in enumerate(parents):
                    if i in coalition:
                        parent_vals[parent] = parent_data[parent][data_idx]
                    else:
                        marginal_idx = rng.integers(0, n_data)
                        parent_vals[parent] = parent_data[parent][marginal_idx]

                if isinstance(model, LinearGaussianBayesianNetwork):
                    val = cpd.beta[0]
                    for j, p in enumerate(cpd.evidence):
                        val += cpd.beta[j + 1] * parent_vals[p]
                    samples[s_idx] = val
                else:
                    state_names = model.states
                    parent_indices = []
                    cards = [len(state_names[p]) for p in cpd.variables[1:]]
                    for p in cpd.variables[1:]:
                        p_states = state_names[p]
                        p_val = parent_vals[p]
                        p_idx = list(p_states).index(p_val)
                        parent_indices.append(p_idx)
                    prob_table = cpd.get_values()
                    idx = 0
                    for j, p_idx in enumerate(parent_indices):
                        stride = 1
                        for k in range(j + 1, len(cards)):
                            stride *= cards[k]
                        idx += p_idx * stride
                    probs = prob_table[:, idx]
                    samples[s_idx] = sum(i * p for i, p in enumerate(probs))
            return self.functional(samples)

        engine = ShapleyEngine(n_players=len(parents), value_function=value_fn, method="auto")
        indexed_result = engine.compute()
        return {player_to_parent[i]: v for i, v in indexed_result.items()}
