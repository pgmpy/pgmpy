from networkx.algorithms.dag import descendants

from pgmpy.base import DAG
from pgmpy.identification import BaseIdentification
from pgmpy.inference.CausalInference import CausalInference


class InstrumentalVariables(BaseIdentification):
    """
    Given a causal graph, finds the (conditional) instrumental variable(s) as described in [1]_.

    Given a causal graph with roles 'exposure', 'outcome' and 'latents' specified, this class provides methods to find
    conditional or non-conditional instrumental variables that can be used to identify the causal effect of the exposure
    on the outcome. It also provides a method to validate whether a given instrumental variable is valid.

    Parammeters
    ----------
    variant: str, optional
        The variant of instrumental variable identification to use. Supported variants are:
        - 'non-conditional': Returns a causal graph with identified non-conditional instrument variable(s).
        - 'conditional': Returns a causal graph with identified conditional instrument variable and its
                         corresponding conditional variables.

    scaling_indicators: dict, optional
        A dictionary specifying the scaling indicators for exposure and/or outcome variables if they are latent of the
        form {latent_variable: scaling_indicator}. Scaling indicators are observed variables that have an incoming edge
        from the corresponding latent variable. This uses the observed variable as a proxy measurement for the latent
        exposure/outcome and allows identification of certain causal effects that would otherwise be unidentifiable.

        If scaling indicators are not provided, if required, the method will automatically assign scaling indicators by
        selecting an child observed node of the latent variable that is not the outcome (for latent exposures).

    Examples
    --------
    >>> from pgmpy.base import DAG
    >>> from pgmpy.identification import InstrumentalVariables
    >>> edges = [
    ...     ("Z", "X"),
    ...     ("X", "Y"),
    ...     ("L1", "X"),
    ...     ("L1", "Y"),
    ...     ("L1", "W1"),
    ...     ("W1", "Y"),
    ... ]
    >>> causal_graph = DAG(edges, exposures="X", outcomes="Y", latents=["L1"])
    >>> iv_identifier = InstrumentalVariables(variant="non-conditional")
    >>> identified_graph, is_identified = iv_identifier.identify(causal_graph)
    >>> is_identified
    True
    >>> identified_graph.get_role("instrument")
    {'Z'}
    >>> iv_identifier.validate(identified_graph)
    True

    References
    ----------
    .. [1] Ankan, A., Wortel, I., Bollen, K. A., & Textor, J. (2023). Combining Graphical and Algebraic Approaches for
           Parameter Identification in Latent Variable Structural Equation Models. arXiv:2302.13220 [stat.ME].
    """

    def __init__(
        self,
        variant="non-conditional",
        scaling_indicators=None,
    ) -> None:
        self.supported_graph_types = (DAG,)
        self.variant = variant.lower()
        self.scaling_indicators = (
            dict() if scaling_indicators is None else scaling_indicators
        )

    def _get_scaling_indicators(self, causal_graph):
        exposure = causal_graph.get_role("exposures")[0]
        outcome = causal_graph.get_role("outcomes")[0]
        latent_variables = causal_graph.get_role("latents")
        all_nodes = causal_graph.nodes()
        observed_nodes = all_nodes - latent_variables

        if exposure in latent_variables:
            if exposure in self.scaling_indicators.keys():
                if self.scaling_indicators[exposure] == outcome:
                    raise ValueError(
                        f"{outcome} is the outcome variable and cannot be the scaling indicator for the latent exposure"
                        f"{exposure}."
                    )
            else:
                self.scaling_indicators[exposure] = [
                    neighbour
                    for neighbour in causal_graph.neighbors(exposure)
                    if neighbour != outcome and neighbour in observed_nodes
                ][0]

        if outcome in latent_variables:
            if outcome not in self.scaling_indicators.keys():
                self.scaling_indicators[outcome] = [
                    neighbour
                    for neighbour in causal_graph.neighbors(outcome)
                    if neighbour in observed_nodes
                ][0]

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

        elif self.variant == "non-conditional":
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
        Validates the causal graph whether the specified (conditional) instrumental variable(s) is/are valid.

        Given a causal graph with variable roles 'exposure, 'outcome', 'instruments', and optionally 'conditional'
        defined, this method checks whether the given instrument set is valid.

        Parameters
        ----------
        causal_graph: DAG
            The causal graph to validate. This graph must have roles 'exposure', 'outcome', 'instrument', and optionally
            'conditional' defined.

        Returns
        -------
        bool: True if the specified instrumental set is valid, False otherwise.
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
