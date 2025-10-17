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
    ...         outcome_parents = causal_graph.predecessors(
    ...             causal_graph.get_role("exposure")
    ...         )
    ...         identified_cg = causal_graph.with_role("adjustment", outcome_parents)
    ...         return identified_cg, True
    ...
    """

    def _validate_causal_graph(self, causal_graph):
        # Check if the passed causal_graph is supported by the method.
        if not isinstance(causal_graph, self.supported_graph_types):
            raise ValueError(
                f"The `causal_graph` must be an instance of {self.supported_graph_types} for this method."
            )

        # Check if causal_graph has `exposure` and `outcome` roles assigned.
        causal_graph.is_valid_causal_structure()

    def identify(self, causal_graph):
        """
        Run the identification algorithm on a causal graph.

        This method applies the identification procedure to the input causal
        graph, annotating it with variable roles (e.g., adjustment, IVs) while
        keeping the original graphical structure.

        Parameters
        ----------
        causal_graph : DAG, PDAG, ADMG, MAG, or PAG object
            The input causal graph on which to perform identification. The
            causal graph must have variables with exposure and outcome roles
            defined.

        Returns
        -------
        identified_graph : DAG, PDAG, ADMG, MAG, or PAG object
            A new causal graph instance with variable roles assigned according
            to the identification method.

        success : bool
            True if the exposure and outcome are successfully identified; False
            otherwise.
        """
        self._validate_causal_graph(causal_graph)
        return self._identify(causal_graph)

    def validate(self, causal_graph):
        """
        Validate the input causal graph for identification.

        This method checks if the variable roles assigned in the `causal_graph`
        are appropriate for identification. For example, given a causal graph
        with exposure, outcome, and adjustment roles, it verifies that the
        adjustment set is valid for the given exposure and outcome.

        Parameters
        ----------
        causal_graph : DAG, PDAG, ADMG, MAG, or PAG object
            The input causal graph to validate.

        Returns
        -------
        bool:
            True if the graph is valid for identification; False otherwise.
        """
        self._validate_causal_graph(causal_graph)
        return self._validate(causal_graph)

    def __call__(self, causal_graph):
        """Alias for the `identify` method"""
        return self.identify(causal_graph)
