from pgmpy.base import DAG
from pgmpy.identification import BaseIdentification


class InstrumentVariables(BaseIdentification):

    def __init__(self, variant=None):
        self.supported_graph_types = DAG
        self.variant = variant

    def _get_scaling_indicators(self, causal_graph):
        latent_nodes = causal_graph.get_role("latents")
        observed_nodes = set(causal_graph.get_role("observed"))

        scaling_indicators = {}
        for node in latent_nodes:
            for neighbour in causal_graph.neighbors(node):
                if neighbour in observed_nodes:
                    scaling_indicators[node] = neighbour
                    break

        return scaling_indicators

    def _iv_transformations(self, X, Y, causal_graph, scaling_indicators={}):
        full_graph = causal_graph.copy()
        print("full graph:", full_graph.edges())

        exposures = full_graph.get_role("exposures")
        observed = full_graph.get_role("observed")
        latent_variables = set(full_graph.get_role("latents"))

        if not full_graph.has_edge(X, Y):
            raise ValueError(f"The edge from {X} -> {Y} does not exist in the graph")

        if (X in exposures) and (Y in observed):
            full_graph.remove_edge(X, Y)

        elif Y in latent_variables:
            full_graph.add_edge("." + Y, scaling_indicators[Y])
            dependent_var = scaling_indicators[Y]

        else:
            dependent_var = Y

        variable_parents = [
            var for var in causal_graph.predecessors(Y) if not var.startswith(".")
        ]

        for parent_y in variable_parents:
            full_graph.remove_edge(parent_y, Y)
            if parent_y in latent_variables:
                full_graph.add_edge("." + scaling_indicators[parent_y], dependent_var)

        return full_graph, dependent_var

    def _identify(self, causal_graph):
        exposure = causal_graph.get_role("exposures")[0]
        outcome = causal_graph.get_role("outcomes")[0]
        # observed = causal_graph.get_role("observed")
        latent_variables = set(causal_graph.get_role("latents"))
        scaling_indicators = self._get_scaling_indicators(causal_graph)

        # if (X in scaling_indicators.keys()) and (scaling_indicators[X] == Y):
        #     logger.warning(
        #         f"{Y} is the scaling indicator of {X}. Please specify `scaling_indicators`"
        #     )

        transformed_graph, dependent_var = self._iv_transformations(
            exposure, outcome, causal_graph, scaling_indicators=scaling_indicators
        )

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

        if identified_instruments is None:
            return causal_graph, False
        else:
            return (
                causal_graph.with_role(
                    "instrument", identified_instruments, inplace=False
                ),
                True,
            )
