from typing import Literal

import networkx as nx
from networkx.algorithms.dag import descendants

from pgmpy.base import DAG
from pgmpy.identification import BaseGraphicalIdentification


class InstrumentalVariables(BaseGraphicalIdentification):
    """
    Given a causal graph, identifies instrumental variables for estimating causal effects.

    Given a causal graph with `exposures` and `outcomes` roles specified, this class provides
    methods to identify and validate instrumental variables (IVs) satisfying the instrumental
    variable conditions.

    Parameters
    ----------
    variant: {'unconditional', 'conditional', 'total-conditional'}, default='unconditional'
        The variant of instrumental variable identification to use:
        - 'unconditional': Searches for standard (unconditional) instrumental variables.
        - 'conditional': Searches for instruments that are valid conditional on a set of covariates.
        - 'total-conditional': Searches for conditional instruments where the conditioning set
          does not block any causal path between the exposure and outcome.

    scaling_indicators: dict[str, str], optional
        A mapping from latent variable names to their observed scaling indicators.
        If not provided, scaling indicators are selected automatically from each latent
        variable's observed neighbors.

    Examples
    --------
    >>> from pgmpy.base import DAG
    >>> from pgmpy.identification import InstrumentalVariables
    >>> dag = DAG(
    ...     ebunch=[("Z", "X"), ("X", "Y"), ("U", "X"), ("U", "Y")],
    ...     roles={"exposures": "X", "outcomes": "Y"},
    ...     latents={"U"},
    ... )
    >>> identified_dag, is_identified = InstrumentalVariables().identify(dag)
    >>> is_identified
    True
    >>> identified_dag.get_role("instruments")
    ['Z']
    >>> InstrumentalVariables().validate(identified_dag)
    True
    """

    def __init__(
        self,
        variant: Literal["unconditional", "conditional", "total-conditional"] = "unconditional",
        scaling_indicators: dict[str, str] | None = None,
    ) -> None:
        if variant not in ("unconditional", "conditional", "total-conditional"):
            raise ValueError(
                f"Invalid variant: {variant}. Expected one of 'unconditional', 'conditional', 'total-conditional'."
            )
        self.supported_graph_types = (DAG,)
        self.variant = variant
        self.scaling_indicators = scaling_indicators

    def _get_scaling_indicators(self, causal_graph: DAG, exposure: str, outcome: str) -> dict[str, str]:
        """
        Returns or completes a mapping of scaling indicators for all latent variables in the model.
        """
        latents = (
            set(causal_graph.latents) if hasattr(causal_graph, "latents") else set(causal_graph.get_role("latents"))
        )
        observed = (
            set(causal_graph.observed) if hasattr(causal_graph, "observed") else (set(causal_graph.nodes()) - latents)
        )

        scaling: dict[str, str] = {}
        if self.scaling_indicators is not None:
            scaling = dict(self.scaling_indicators)
            if exposure in scaling and scaling[exposure] == outcome:
                raise ValueError(
                    f"{outcome} is the scaling indicator for {exposure}. "
                    f"Specify a different scaling indicator for {exposure}."
                )

        for node in latents:
            if node in scaling:
                continue
            for neighbour in causal_graph.neighbors(node):
                if neighbour in observed:
                    if node == exposure and neighbour == outcome:
                        continue
                    scaling[node] = neighbour
                    break

        return scaling

    def _iv_transformations(
        self, X: str, Y: str, causal_graph: DAG, scaling_indicators: dict[str, str]
    ) -> tuple[DAG, str]:
        """
        Transforms the causal graph for finding instrumental variables.
        """
        full_graph = causal_graph.copy()
        latents = (
            set(causal_graph.latents) if hasattr(causal_graph, "latents") else set(causal_graph.get_role("latents"))
        )
        observed = (
            set(causal_graph.observed) if hasattr(causal_graph, "observed") else (set(causal_graph.nodes()) - latents)
        )

        if not full_graph.has_edge(X, Y):
            raise ValueError(f"The edge from {X} -> {Y} does not exist in the graph")

        if (X in observed) and (Y in observed):
            full_graph.remove_edge(X, Y)
            return full_graph, Y
        elif Y in latents:
            full_graph.add_edge("." + Y, scaling_indicators[Y])
            dependent_var = scaling_indicators[Y]
        else:
            dependent_var = Y

        variable_parents = [var for var in causal_graph.predecessors(Y) if not var.startswith(".")]
        for parent_y in variable_parents:
            if full_graph.has_edge(parent_y, Y):
                full_graph.remove_edge(parent_y, Y)
            if parent_y in latents:
                full_graph.add_edge("." + scaling_indicators[parent_y], dependent_var)

        return full_graph, dependent_var

    @staticmethod
    def _nearest_separator(G: DAG, Y: str, Z: str, observed: set[str]) -> set[str] | None:
        """
        Finds the nearest separator for Y and Z in graph G among observed variables.
        """
        W = set()
        ancestors_y_z = nx.ancestors(G, Y).union(nx.ancestors(G, Z)).union({Y, Z})
        ancestral_G = G.subgraph(ancestors_y_z).copy()

        M = ancestral_G.moralize()
        visited = {Y}
        to_visit = list(M.neighbors(Y))

        while to_visit:
            node = to_visit.pop()
            if node == Z:
                return None
            visited.add(node)
            if node in observed:
                W.add(node)
            else:
                to_visit.extend([n for n in M.neighbors(node) if n not in visited])

        if Y not in G.active_trail_nodes([Z], observed=W)[Z]:
            return W
        return None

    def _find_unconditional_ivs(
        self, causal_graph: DAG, exposure: str, outcome: str, observed: set[str], latents: set[str]
    ) -> set[str]:
        scaling_indicators = self._get_scaling_indicators(causal_graph, exposure, outcome)
        transformed_graph, dependent_var = self._iv_transformations(exposure, outcome, causal_graph, scaling_indicators)

        explanatory_var = scaling_indicators[exposure] if exposure in latents else exposure

        d_connected_x = transformed_graph.active_trail_nodes([explanatory_var])[explanatory_var]

        transformed_graph_copy = transformed_graph.copy()
        transformed_graph_copy.remove_edges_from(list(transformed_graph_copy.in_edges(explanatory_var)))
        d_connected_y = transformed_graph_copy.active_trail_nodes([dependent_var])[dependent_var]

        candidates = d_connected_x - d_connected_y - {explanatory_var, dependent_var}
        return candidates & observed

    def _find_conditional_ivs(
        self,
        causal_graph: DAG,
        exposure: str,
        outcome: str,
        observed: set[str],
        total_effect: bool = False,
    ) -> list[tuple[str, set[str]]]:
        scaling_indicators = self._get_scaling_indicators(causal_graph, exposure, outcome)
        transformed_graph, dependent_var = self._iv_transformations(exposure, outcome, causal_graph, scaling_indicators)
        if (exposure, outcome) in transformed_graph.edges():
            transformed_graph.remove_edge(exposure, outcome)
        G_c = transformed_graph

        nodes_on_paths: set[str] = set()
        if total_effect:
            all_paths = list(nx.all_simple_paths(causal_graph, exposure, outcome))
            nodes_on_paths = {node for path in all_paths for node in path} - {exposure, outcome}

        instruments: list[tuple[str, set[str]]] = []
        for Z in observed - {exposure, outcome}:
            W = self._nearest_separator(G_c, dependent_var, Z, observed)
            if not W:
                continue

            if total_effect and W.intersection(nodes_on_paths):
                continue

            if W.intersection(descendants(G_c, dependent_var)) or (exposure in W):
                continue

            if exposure in causal_graph.active_trail_nodes([Z], observed=W)[Z]:
                instruments.append((Z, W))

        return instruments

    def _identify(self, causal_graph: DAG):
        exposures = causal_graph.get_role("exposures")
        outcomes = causal_graph.get_role("outcomes")

        if len(exposures) != 1:
            raise ValueError(f"The current implementation supports only one exposure. Got: {len(exposures)}")
        if len(outcomes) != 1:
            raise ValueError(f"The current implementation supports only one outcome. Got: {len(outcomes)}")

        exposure = exposures[0]
        outcome = outcomes[0]

        latents = (
            set(causal_graph.latents) if hasattr(causal_graph, "latents") else set(causal_graph.get_role("latents"))
        )
        observed = (
            set(causal_graph.observed) if hasattr(causal_graph, "observed") else (set(causal_graph.nodes()) - latents)
        )

        if self.variant == "unconditional":
            ivs = self._find_unconditional_ivs(causal_graph, exposure, outcome, observed, latents)
            if ivs:
                identified_dag = causal_graph.with_role("instruments", sorted(ivs), inplace=False)
                return identified_dag, True
            return causal_graph, False

        is_total = self.variant == "total-conditional"
        cond_ivs = self._find_conditional_ivs(causal_graph, exposure, outcome, observed, total_effect=is_total)

        if not cond_ivs:
            return causal_graph, False

        annotated_graphs = [
            causal_graph.with_role("instruments", [z], inplace=False).with_role(
                "conditioning", sorted(w), inplace=False
            )
            for z, w in cond_ivs
        ]
        if len(annotated_graphs) == 1:
            return annotated_graphs[0], True
        return annotated_graphs, True

    def _validate(self, causal_graph: DAG) -> bool:
        exposures = causal_graph.get_role("exposures")
        outcomes = causal_graph.get_role("outcomes")
        instruments = causal_graph.get_role("instruments")

        if len(exposures) != 1 or len(outcomes) != 1 or not instruments:
            return False

        exposure = exposures[0]
        outcome = outcomes[0]
        latents = (
            set(causal_graph.latents) if hasattr(causal_graph, "latents") else set(causal_graph.get_role("latents"))
        )
        observed = (
            set(causal_graph.observed) if hasattr(causal_graph, "observed") else (set(causal_graph.nodes()) - latents)
        )

        try:
            if self.variant == "unconditional":
                valid_ivs = self._find_unconditional_ivs(causal_graph, exposure, outcome, observed, latents)
                return set(instruments).issubset(valid_ivs)

            conditioning = set(causal_graph.get_role("conditioning"))
            is_total = self.variant == "total-conditional"
            valid_cond_ivs = self._find_conditional_ivs(
                causal_graph, exposure, outcome, observed, total_effect=is_total
            )
            for z in instruments:
                if (z, conditioning) in valid_cond_ivs:
                    return True
            return False
        except Exception:
            return False
