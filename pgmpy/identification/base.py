class BaseIdentification:
    """Base class for all identification methods.

    All identification methods in pgmpy should inherit `BaseIdentification`.

    Examples
    --------
    >>> from pgmpy.identification import BaseIdentification
    >>> class SimpleId(BaseIdentification):
    ...     "A simple identification method when all variable are observed"
    ...
    ...     def __call__(self):
    ...         outcome_parents = set(
    ...             self.causal_graph.predecessors(self.causal_graph.exposure)
    ...         ) - {self.exposure}
    ...         self.causal_graph.add_roles("adjustment", outcome_parents)
    ...         return outcome_parents, True
    ...
    """

    def __init__(self, causal_graph):
        # TODO: Once CausalGraph class is defined,
        #       add a check here for the type of causal_graph.
        self.causal_graph = causal_graph.copy()
        self.observed_variables = frozenset(self.causal_graph.nodes()).difference(
            self.causal_graph.latents
        )
        self.latent_variables = self.causal_graph.latents
