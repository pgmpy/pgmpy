class BaseIdentification:
    """Base class for all identification methods.

    All identification methods in pgmpy must inherit `BaseIdentification`.
    Inheriting methods need to define the `_identify` method, which implements
    the specific identification algorithm. The `_identify` method should take a
    causal graph as input and return a modified version of the graph with
    variable roles assigned, along with a boolean indicating whether the
    identification was successful.

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
    ...         causal_graph = causal_graph.add_roles("adjustment", outcome_parents)
    ...         return casual_graph, True
    ...
    """

    def identify(self, causal_graph):
        """
        Run the identification algorithm on a causal graph.

        This method applies the identification procedure to the input causal
        graph, annotating it with variable roles (e.g., adjustment, IVs) while
        keeping the original graphical structure.

        Parameters
        ----------
        causal_graph : DAG, PDAG, ADMG, MAG, or PAG object
            The input causal graph on which to perform identification.

        Returns
        -------
        identified_graph : DAG, PDAG, ADMG, MAG, or PAG object
            A new causal graph instance with variable roles assigned.

        success : bool
            True if the exposure and outcome are successfully identified; False otherwise.
        """
        if causal_graph.is_valid_causal_structure():
            return self._identify(causal_graph)

    def __call__(self, causal_graph):
        """Alias for the `identify` method"""
        return self.identify(causal_graph)
