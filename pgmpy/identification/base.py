class BaseIdentification:
    """Base class for all identification methods.

    All identification methods in pgmpy should inherit `BaseIdentification`.

    Examples
    --------
    >>> from pgmpy.identification import BaseIdentification
    >>> class SimpleId(BaseIdentification):
    ...     "A simple identification method when all variable are observed"
    ...
    ...     def _identify(self, causal_graph):
    ...         outcome_parents = set(
    ...             causal_graph.predecessors(causal_graph.exposure)
    ...         ) - {causal_graph.exposure}
    ...         causal_graph.add_roles("adjustment", outcome_parents)
    ...         return outcome_parents, True
    ...
    """

    def identify(self, causal_graph):
        """Method to run the identification method.

        The method accepts a causal graph and returns a causal
        graph with the same structure and with defined variable
        roles based on the identification method.

        Returns
        -------
        causal_graph: Instance of CausalGraph
            The causal graph with variable roles assigned.

        success: bool
            Whether the causal graph with given exposure and outcome is
            identified.
        """
        causal_graph = causal_graph.copy()
        return self._identify(causal_graph)

    def __call__(self, causal_graph):
        """Alias for the `identify` method"""
        return self.identify(causal_graph)
