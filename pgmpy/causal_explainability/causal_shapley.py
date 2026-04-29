import networkx as nx
import numpy as np

from pgmpy.causal_explainability._base import _BaseAttribution
from pgmpy.causal_explainability._shapley import ShapleyEngine


class CausalShapleyValues(_BaseAttribution):
    """Shapley values for individual predictions respecting causal structure.

    Parameters
    ----------
    method : str
        "observational" — condition on coalition (standard SHAP).
        "interventional" — use do-operator for coalitions (Janzing 2020).
        "causal" — interventional for ancestors, observational for descendants (Heskes 2020).

    References
    ----------
    [1] Lundberg, S. M. & Lee, S. I. (2017). A Unified Approach to
        Interpreting Model Predictions. NeurIPS 2017. (observational)
    [2] Janzing, D., Minorics, L., & Blöbaum, P. (2020). Feature relevance
        quantification in explainable AI: A causal problem. AISTATS 2020.
        (interventional)
    [3] Heskes, T., Bucur, E., Goethals, B., & Sijben, E. (2020). Causal
        Shapley Values: Exploiting Causal Knowledge to Explain Individual
        Predictions of Complex Models. NeurIPS 2020. (causal)
    """

    def __init__(self, method="interventional"):
        if method not in ("observational", "interventional", "causal"):
            raise ValueError(f"method must be 'observational', 'interventional', or 'causal', got '{method}'")
        self.method = method

    def _attribute(self, model, data, target, **kwargs):
        observation = kwargs["observation"]
        features = [n for n in model.nodes() if n != target]
        if not features:
            return {}

        player_to_feature = {i: f for i, f in enumerate(features)}
        feature_to_player = {f: i for i, f in enumerate(features)}

        from pgmpy.models import LinearGaussianBayesianNetwork

        if isinstance(model, LinearGaussianBayesianNetwork):
            value_fn = self._make_value_fn_lgbn(model, data, target, observation, features, player_to_feature)
        else:
            value_fn = self._make_value_fn_general(model, data, target, observation, features, player_to_feature)

        causal_ordering = None
        if self.method == "causal":
            topo = list(nx.topological_sort(model))
            feature_topo = [f for f in topo if f in features]
            causal_ordering = [{feature_to_player[f]} for f in feature_topo]

        engine = ShapleyEngine(
            n_players=len(features),
            value_function=value_fn,
            causal_ordering=causal_ordering,
            method="auto",
        )
        n_perms = kwargs.get("n_permutations", 1000)
        seed = kwargs.get("seed", None)
        indexed_result = engine.compute(n_permutations=n_perms, seed=seed)
        return {player_to_feature[i]: v for i, v in indexed_result.items()}

    def _make_value_fn_lgbn(self, model, data, target, observation, features, player_to_feature):
        topo_order = list(nx.topological_sort(model))
        data_means = {col: data[col].mean() for col in data.columns}
        ancestors_of_target = nx.ancestors(model, target)

        def value_fn(coalition):
            active_features = {player_to_feature[i] for i in coalition}
            node_values = {}

            for node in topo_order:
                if node == target:
                    cpd = model.get_cpds(node)
                    val = cpd.beta[0]
                    for j, p in enumerate(cpd.evidence):
                        val += cpd.beta[j + 1] * node_values.get(p, data_means.get(p, 0.0))
                    return val

                if node in active_features:
                    node_values[node] = observation[node]
                else:
                    if self.method == "interventional":
                        node_values[node] = data_means[node]
                    elif self.method == "observational":
                        cpd = model.get_cpds(node)
                        val = cpd.beta[0]
                        for j, p in enumerate(cpd.evidence):
                            val += cpd.beta[j + 1] * node_values.get(p, data_means.get(p, 0.0))
                        node_values[node] = val
                    else:  # causal
                        if node in ancestors_of_target:
                            node_values[node] = data_means[node]
                        else:
                            cpd = model.get_cpds(node)
                            val = cpd.beta[0]
                            for j, p in enumerate(cpd.evidence):
                                val += cpd.beta[j + 1] * node_values.get(p, data_means.get(p, 0.0))
                            node_values[node] = val
            return 0.0

        return value_fn

    def _make_value_fn_general(self, model, data, target, observation, features, player_to_feature):
        n_mc = 200
        rng = np.random.default_rng(None)
        ancestors_of_target = nx.ancestors(model, target)

        def value_fn(coalition):
            active_features = {player_to_feature[i] for i in coalition}
            do_vars = {}
            evidence_vars = {}

            for f in active_features:
                if self.method == "interventional":
                    do_vars[f] = observation[f]
                elif self.method == "observational":
                    evidence_vars[f] = observation[f]
                else:  # causal
                    if f in ancestors_of_target:
                        do_vars[f] = observation[f]
                    else:
                        evidence_vars[f] = observation[f]

            try:
                samples = model.simulate(
                    n_samples=n_mc,
                    do=do_vars if do_vars else None,
                    evidence=evidence_vars if evidence_vars else None,
                    seed=int(rng.integers(1e9)),
                    show_progress=False,
                )
                return samples[target].mean()
            except Exception:
                return 0.0

        return value_fn
