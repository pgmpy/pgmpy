from typing import Literal

from pgmpy.base import ADMG, DAG
from pgmpy.identification.probability_expression import ProbabilityExpressionTree


class BaseGraphicalIdentification:
    """Base class for identification methods that return annotated graphs.

    Graph-returning identification methods inherit `BaseGraphicalIdentification`.
    Inheriting methods need to define the `_identify` method, which implements
    the specific identification algorithm. The `_identify` method should take a
    causal graph as input and return a modified version of the graph with
    variable roles assigned, along with a boolean indicating whether the
    identification was successful.

    Examples
    --------
    >>> from pgmpy.identification import BaseGraphicalIdentification
    >>> class SimpleId(BaseGraphicalIdentification):
    ...     "A simple identification method when all variable are observed"
    ...
    ...     def _identify(self, causal_graph):
    ...         outcome_parents = causal_graph.predecessors(
    ...             causal_graph.get_role("exposures")
    ...         )
    ...         identified_cg = causal_graph.with_role("adjustment", outcome_parents)
    ...         return identified_cg, True
    ...
    """

    def _validate_causal_graph(self, causal_graph):
        # Check if the passed causal_graph is supported by the method.
        if not isinstance(causal_graph, self.supported_graph_types):
            raise ValueError(f"The `causal_graph` must be an instance of {self.supported_graph_types} for this method.")

        # Check if causal_graph has `exposures` and `outcomes` roles assigned.
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
            causal graph must have variables with exposures and outcomes roles
            defined.

        Returns
        -------
        identified_graph : DAG, PDAG, ADMG, MAG, or PAG object
            A new causal graph instance with variable roles assigned according
            to the identification method.

        success : bool
            True if the exposures and outcomes are successfully identified; False
            otherwise.
        """
        self._validate_causal_graph(causal_graph)
        return self._identify(causal_graph)

    def validate(self, causal_graph):
        """
        Validate the input causal graph for identification.

        This method checks if the variable roles assigned in the `causal_graph`
        are appropriate for identification. For example, given a causal graph
        with exposures, outcomes, and adjustment roles, it verifies that the
        adjustment set is valid for the given exposures and outcomes.

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


class BaseFormulaIdentification:
    """Base class for identification methods that return a symbolic expression.

    Subclasses should define ``supported_graph_types`` and implement
    ``_identify``. The ``_identify`` method must return a
    ``ProbabilityExpressionTree`` when the effect is identifiable, or
    ``False`` otherwise. If identification fails, subclasses should set
    ``self.hedge_`` to the pair of graphs forming the hedge.

    By default, only the ``"outcomes"`` role is required. The ``"exposures"``
    and ``"conditioning"`` roles may be empty, as in ID and IDC queries with
    no interventions or no conditioning variables. Subclasses may require
    additional roles through ``required_roles``.

    Attributes
    ----------
    hedge_ : tuple of graph objects or None
        The pair ``(F, F_prime)`` forming a hedge for a non-identifiable
        effect. Reset to ``None`` before each call to ``_identify``.

    Examples
    --------
    >>> from pgmpy.base import DAG
    >>> from pgmpy.identification import BaseFormulaIdentification
    >>> from pgmpy.identification.probability_expression import (
    ...     ProbabilityExpressionTree, ProbabilityNode
    ... )
    >>> class SimpleFormulaId(BaseFormulaIdentification):
    ...     supported_graph_types = (DAG,)
    ...     def _identify(self, causal_graph):
    ...         y = causal_graph.get_role("outcomes")
    ...         x = causal_graph.get_role("exposures")
    ...         return ProbabilityExpressionTree(
    ...             root=ProbabilityNode(frozenset(y), cond=frozenset(x))
    ...         )
    """

    supported_graph_types = ()
    required_roles = ("outcomes",)
    hedge_: tuple[ADMG | DAG, ADMG | DAG] | None

    def _validate_causal_graph(self, causal_graph: ADMG | DAG) -> None:
        """Validate the causal graph before running identification.

        Checks that:

        1. ``causal_graph`` is an instance of one of ``supported_graph_types``.
        2. Every role listed in ``required_roles`` is nonempty.

        Parameters
        ----------
        causal_graph : ADMG or DAG
            The causal graph with the roles in ``required_roles`` assigned.
            By default, only ``"outcomes"`` is required; ``"exposures"`` and
            ``"conditioning"`` may be empty.

        Raises
        ------
        ValueError
            If the graph type is not supported, or if a required role is
            missing.
        """
        if not isinstance(causal_graph, self.supported_graph_types):
            raise ValueError(
                f"causal_graph must be an instance of "
                f"{self.supported_graph_types} for this method. "
                f"Got {type(causal_graph).__name__}."
            )

        for role in self.required_roles:
            if not causal_graph.get_role(role):
                raise ValueError(f"causal_graph must have '{role}' role assigned for {type(self).__name__}.")

    def identify(self, causal_graph: ADMG | DAG) -> ProbabilityExpressionTree | Literal[False]:
        """
        Run the identification algorithm on a causal graph.

        Validates the graph via ``_validate_causal_graph``, resets
        ``self.hedge_`` to ``None``, then delegates to ``_identify``.

        Parameters
        ----------
        causal_graph : ADMG or DAG
            The causal graph with the roles in ``required_roles`` assigned.
            By default, only ``"outcomes"`` is required; ``"exposures"`` and
            ``"conditioning"`` may be empty.

        Returns
        -------
        ProbabilityExpressionTree
            The symbolic formula for the identified causal effect. Access the
            expression tree via ``result.root``.

        False
            If the causal effect is not identifiable. The pair of graphs
            forming the hedge is stored in ``self.hedge_``.
        """
        self._validate_causal_graph(causal_graph)
        self.hedge_ = None
        return self._identify(causal_graph)

    def _identify(self, causal_graph: ADMG | DAG) -> ProbabilityExpressionTree | Literal[False]:
        """Override in subclasses to implement the identification algorithm.

        Parameters
        ----------
        causal_graph : ADMG or DAG
            The causal graph with the roles in ``required_roles`` assigned.
            By default, only ``"outcomes"`` is required; ``"exposures"`` and
            ``"conditioning"`` may be empty.

        Returns
        -------
        ProbabilityExpressionTree or False
        """
        raise NotImplementedError

    def __call__(self, causal_graph: ADMG | DAG) -> ProbabilityExpressionTree | Literal[False]:
        """Alias for the ``identify`` method."""
        return self.identify(causal_graph)
