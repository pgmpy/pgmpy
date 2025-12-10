from networkx.algorithms.dag import descendants

from pgmpy.base import DAG
from pgmpy.identification import BaseIdentification
from pgmpy.inference.CausalInference import CausalInference


class InstrumentalVariables(BaseIdentification):

    def __init__(self, variant=None, scaling_indicators=None) -> None:
        self.supported_graph_types = DAG
        self.variant = variant
        self.scaling_indicators = scaling_indicators

    def _get_scaling_indicators(self, causal_graph):
        exposure = causal_graph.get_role("exposures")
        outcome = causal_graph.get_role("outcomes")
        latent_variables = causal_graph.get_role("latents")
        all_nodes = causal_graph.nodes()
        observed_nodes = all_nodes - latent_variables
        scaling_indicators = {}

        if self.scaling_indicators is not None and len(self.scaling_indicators) == len(
            latent_variables
        ):
            return self.scaling_indicators

        # checks for missing scaling indicators and assigns for the missing nodes (if any)
        if self.scaling_indicators is not None and len(self.scaling_indicators) < len(
            latent_variables
        ):
            missing_scaling_indicators = set(latent_variables) - set(
                self.scaling_indicators.keys()
            )
            for node in missing_scaling_indicators:
                for neighbour in causal_graph.neighbors(node):
                    if neighbour in observed_nodes:
                        if not (node in exposure and neighbour != outcome):
                            self.scaling_indicators[node] = neighbour
                            break
            return self.scaling_indicators

        for node in latent_variables:
            for neighbour in causal_graph.neighbors(node):
                if neighbour in observed_nodes:
                    if not (node in exposure and neighbour != outcome):
                        scaling_indicators[node] = neighbour
                        scaling_indicators
                        break
        return scaling_indicators

    def _iv_transformations(self, X, Y, causal_graph, scaling_indicators=None):
        full_graph_ = causal_graph.copy()

        exposures = full_graph_.get_role("exposures")
        latent_variables = full_graph_.get_role("latents")
        observed = full_graph_.get_role("observed")
        scaling_indicators = self._get_scaling_indicators(causal_graph)

        if not full_graph_.has_edge(X, Y):
            raise ValueError(f"The edge from {X} -> {Y} does not exist in the graph")

        if (X in exposures) and (Y in observed):
            if full_graph_.has_edge(X, Y):
                full_graph_.remove_edge(X, Y)
            dependent_var = Y

        elif Y in latent_variables:
            full_graph_.add_edge(Y, scaling_indicators[Y])
            dependent_var = scaling_indicators[Y]

        else:
            dependent_var = Y

        variable_parents = [
            var for var in causal_graph.predecessors(Y) if not var.startswith(".")
        ]

        for parent_y in variable_parents:
            if full_graph_.has_edge(X, Y):
                full_graph_.remove_edge(X, Y)
            if parent_y in latent_variables:
                full_graph_.add_edge(scaling_indicators[parent_y], dependent_var)

        return full_graph_, dependent_var

    def _identify(self, causal_graph):
        exposure = causal_graph.get_role("exposures")[0]
        outcome = causal_graph.get_role("outcomes")[0]
        if len(exposure) != 1:
            raise ValueError(
                f"The current implementation suppports only one exposure. Got: {len(exposure)}"
            )
        if len(outcome) != 1:
            raise ValueError(
                f"The current implementation suppports only one outcome. Got: {len(outcome)}"
            )
        all_nodes = causal_graph.nodes()
        observed = all_nodes - causal_graph.get_role("latents")

        latent_variables = causal_graph.get_role("latents")
        scaling_indicators = self._get_scaling_indicators(causal_graph)

        if (exposure in scaling_indicators.keys()) and (
            scaling_indicators[exposure] == outcome
        ):
            raise ValueError(
                f"{outcome} is the scaling indicator of {exposure}. Please specify the correct `scaling_indicators`"
            )

        transformed_graph, dependent_var = self._iv_transformations(
            exposure, outcome, causal_graph, scaling_indicators
        )

        if self.variant == "conditional":
            if (exposure, outcome) in transformed_graph.edges:
                G_c = transformed_graph.remove_edge(exposure, outcome)
            else:
                G_c = transformed_graph

            instruments = []
            for Z in set(observed) - {exposure, outcome}:
                W = CausalInference(G_c)._nearest_separator(G_c, outcome, Z)

                if (
                    not W
                    or (W.intersection(descendants(G_c, outcome)))
                    or (exposure in W)
                ):
                    continue

                # Condition to check if X d-connected to I after conditioning on W.
                elif exposure in (causal_graph.active_trail_nodes([Z], observed=W))[Z]:
                    instruments.append(Z)
                    instruments.extend(list(W))
                else:
                    continue
            if bool(instruments):
                for i in instruments:
                    causal_graph.with_role("instrument", i, inplace=True)
                return causal_graph, True
            else:
                return causal_graph, False

        else:
            if exposure in latent_variables:
                explanatory_var = scaling_indicators[exposure]
            else:
                explanatory_var = exposure

            d_connected_x = transformed_graph.active_trail_nodes([explanatory_var])[
                explanatory_var
            ]

            # Compute the d-connected nodes to Y except any variable connected through X.
            transformed_graph_copy = transformed_graph.copy()
            transformed_graph_copy.remove_edges_from(
                list(transformed_graph_copy.in_edges(explanatory_var))
            )
            d_connected_y = transformed_graph_copy.active_trail_nodes([dependent_var])[
                dependent_var
            ]

            # Remove {X, Y} because they can't be IV for X -> Y
            identified_instruments = (
                d_connected_x - d_connected_y - {dependent_var, explanatory_var}
            )

            if bool(identified_instruments) is False:
                return causal_graph, False
            else:
                return (
                    causal_graph.with_role(
                        "instrument", identified_instruments, inplace=False
                    ),
                    True,
                )
