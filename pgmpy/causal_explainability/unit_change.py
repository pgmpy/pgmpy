from pgmpy.causal_explainability._base import _BaseAttribution
from pgmpy.causal_explainability._shapley import ShapleyEngine


class UnitChangeAttribution(_BaseAttribution):
    """Attributes the change in a target variable between two observations to each direct parent.

    For LinearGaussianBayesianNetwork: analytical decomposition beta_i * (Pa_i_new - Pa_i_old).
    For DiscreteBayesianNetwork: Shapley values over parent set.
    """

    def _validate(self, model, data, target):
        if target is not None and target not in model.nodes():
            raise ValueError(f"Target '{target}' is not a node in the model. Model nodes: {list(model.nodes())}")

    def _attribute(self, model, data, target, **kwargs):
        observation_old = kwargs["observation_old"]
        observation_new = kwargs["observation_new"]

        from pgmpy.models import LinearGaussianBayesianNetwork

        if isinstance(model, LinearGaussianBayesianNetwork):
            return self._attribute_lgbn(model, target, observation_old, observation_new)
        return self._attribute_general(model, target, observation_old, observation_new)

    def _attribute_lgbn(self, model, target, observation_old, observation_new):
        cpd = model.get_cpds(target)
        parents = cpd.evidence
        result = {}
        for i, parent in enumerate(parents):
            beta_i = cpd.beta[i + 1]
            delta = observation_new[parent] - observation_old[parent]
            result[parent] = beta_i * delta
        return result

    def _attribute_general(self, model, target, observation_old, observation_new):
        parents = list(model.get_parents(target))
        if not parents:
            return {}

        player_to_parent = {i: p for i, p in enumerate(parents)}

        def _evaluate_target(parent_values):
            from pgmpy.models import DiscreteBayesianNetwork

            if isinstance(model, DiscreteBayesianNetwork):
                cpd = model.get_cpds(target)
                state_names = model.states
                target_states = state_names[target]

                parent_indices = []
                cards = [len(state_names[p]) for p in cpd.variables[1:]]
                for p in cpd.variables[1:]:
                    p_states = state_names[p]
                    p_val = parent_values[p]
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
                return sum(i * p for i, (s, p) in enumerate(zip(target_states, probs)))
            else:
                raise NotImplementedError(f"UnitChangeAttribution not yet supported for {type(model).__name__}.")

        def value_fn(coalition):
            parent_vals = {}
            for i, parent in enumerate(parents):
                if i in coalition:
                    parent_vals[parent] = observation_new[parent]
                else:
                    parent_vals[parent] = observation_old[parent]
            return _evaluate_target(parent_vals)

        engine = ShapleyEngine(n_players=len(parents), value_function=value_fn, method="auto")
        indexed_result = engine.compute()
        return {player_to_parent[i]: v for i, v in indexed_result.items()}
