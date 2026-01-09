from networkx.algorithms.dag import descendants

from pgmpy.base import DAG
from pgmpy.identification import BaseIdentification
from pgmpy.inference.CausalInference import CausalInference


class InstrumentalVariables(BaseIdentification):
    """
    Given a causal graph, finds the variable satisfying instrumental variable criteria.

    Given a causal graph with roles 'exposure', 'outcome' and 'latents' specified,
    this class provides methods to find the variable satisfying the instrumental variable criteria.
    It also provides a method to validate whether a given instrumental variable is valid.

    Parammeters
    ----------
    variant: str
        The variant of instrumental variable identification to use. Default is None (non-conditional).

        - 'conditional': Returns a causal graph with identified conditional instrument variable and its
                         coresponding conditional variables.

    scaling_indicators: dict, optional
        A dictionary specifying the scaling indicators for latent variables in the causal graph.
        The keys of the dictionary should be the latent variable names, and the values should be their
        corresponding scaling indicators.
        If scaling indicators are not provided, the method will find scaling indicators automatically.
        If all scaling indicators are not provided, the method will automatically find the missing ones.

    Examples
    --------
    TO : DO

    References
    ----------
    .. [1] Ankan, A., Wortel, I., Bollen, K. A., & Textor, J. (2023).
           Combining Graphical and Algebraic Approaches for Parameter
           Identification in Latent Variable Structural Equation Models.
           arXiv:2302.13220 [stat.ME]. https://arxiv.org/abs/2302.13220 :contentReference[oaicite:0]{index=0}
    """

    def __init__(
        self,
        variant=None,
        scaling_indicators=None,
    ) -> None:
        self.supported_graph_types = (DAG,)
        self.variant = variant
        self.scaling_indicators = scaling_indicators

    def _get_scaling_indicators(self, causal_graph):
        exposure = causal_graph.get_role("exposures")
        outcome = causal_graph.get_role("outcomes")
        latent_variables = causal_graph.get_role("latents")
        all_nodes = causal_graph.nodes()
        observed_nodes = all_nodes - latent_variables

        if (
            self.scaling_indicators is not None
            and set(latent_variables) == set(self.scaling_indicators.keys())
            and all(v is not None for v in self.scaling_indicators.values())
        ):
            return self.scaling_indicators

        existing_keys = (
            set(self.scaling_indicators.keys())
            if self.scaling_indicators is not None
            else set()
        )

        if self.scaling_indicators is None:
            self.scaling_indicators = {}

        missing_scaling_indicators = set(latent_variables) - existing_keys

        for node in missing_scaling_indicators:
            for neighbour in causal_graph.neighbors(node):
                if neighbour in observed_nodes:
                    if not (node in exposure and neighbour != outcome):
                        self.scaling_indicators[node] = neighbour
                        break
        return self.scaling_indicators

    def _iv_transformations(self, X, Y, causal_graph, scaling_indicators=None):
        full_graph_ = causal_graph.copy()
        latent_variables = full_graph_.get_role("latents")

        scaling_indicators = self._get_scaling_indicators(causal_graph)

        if not full_graph_.has_edge(X, Y):
            raise ValueError(f"The edge from {X} -> {Y} does not exist in the graph")

        if full_graph_.has_edge(X, Y):
            full_graph_.remove_edge(X, Y)
            dependent_var = Y

        if Y in latent_variables:
            full_graph_.add_edge(Y, scaling_indicators[Y])
            dependent_var = scaling_indicators[Y]

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
            conditionals = []
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
                    conditionals.extend(W)

                else:
                    continue

            if len(instruments):
                causal_graph.with_role("instrument", instruments, inplace=True)
                causal_graph.with_role("conditional", conditionals, inplace=True)
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

    def _validate(self, causal_graph):
        """
        Validate the causal graph for instrumental variable identification.

        Given a causal graph with variable roles 'exposure, 'outcome', 'instruments' defined,
        this method checks whether the given instrument set is valid.

        Parameters
        ----------
        causal_graph: DAG
            The causal graph to validate.

        Returns
        -------
        bool: True if the 'instrument' set is valid, False otherwise.
        """

        exposure = causal_graph.get_role("exposures")[0]
        outcome = causal_graph.get_role("outcomes")[0]
        instruments = causal_graph.get_role("instrument")[0]

        if len(exposure) != 1:
            raise ValueError(
                f"The current implementation suppports only one exposure. Got: {len(exposure)}"
            )
        if len(outcome) != 1:
            raise ValueError(
                f"The current implementation suppports only one outcome. Got: {len(outcome)}"
            )
        if len(instruments) != 1:
            raise ValueError(
                f"The current implementation suppports only one instrument. Got: {len(instruments)}"
            )

        conditional_vars = causal_graph.get_role("conditional")

        # Remove all outgoing edges from X.
        # Check I is d-separated from Y conditioned on Z

        copy = causal_graph.copy()

        copy.remove_edge(exposure, outcome)
        return copy.is_dconnected(
            instruments, exposure, observed=conditional_vars
        ) and not copy.is_dconnected(instruments, outcome, observed=conditional_vars)
