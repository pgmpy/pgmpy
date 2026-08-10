"""Deprecated compatibility shim for :class:`pgmpy.causal_discovery.ExpertKnowledge`."""

import warnings

from pgmpy.causal_discovery import ExpertKnowledge as _ExpertKnowledge


class ExpertKnowledge(_ExpertKnowledge):
    """
    Deprecated: use :class:`pgmpy.causal_discovery.ExpertKnowledge` instead.

    Class to specify expert knowledge for causal discovery / structure
    learning algorithms. This class delegates to the canonical implementation
    in `pgmpy.causal_discovery`; refer to its documentation for details.

    Parameters map onto the canonical class with two legacy conventions:
    `screening_method` is the former name of `ci_test`, and a
    `temporal_order` of `[[]]` is the former sentinel for "no temporal
    information" (canonical uses `None`).
    """

    def __init__(
        self,
        forbidden_edges=None,
        required_edges=None,
        temporal_order=None,
        search_space=None,
        screening_method=None,
        significance_level=0.05,
        **kwargs,
    ):
        warnings.warn(
            "ExpertKnowledge is deprecated and will be removed in v2.0. "
            "Please use pgmpy.causal_discovery.ExpertKnowledge instead.",
            FutureWarning,
            stacklevel=2,
        )
        # Stored for sklearn get_params/clone round-trips of this shim.
        self.screening_method = screening_method
        if temporal_order == [[]]:
            temporal_order = None
        super().__init__(
            forbidden_edges=forbidden_edges,
            required_edges=required_edges,
            temporal_order=temporal_order,
            search_space=search_space,
            ci_test=screening_method,
            significance_level=significance_level,
            **kwargs,
        )

    def apply_expert_knowledge(self, graph):
        """
        Legacy alias of :meth:`pgmpy.causal_discovery.ExpertKnowledge.apply_to`.
        """
        if not hasattr(self, "forbidden_edges_"):
            self.fit()
        return self.apply_to(graph)

    def limit_search_space(self, data):
        """
        Legacy method: resolve `search_space` into forbidden edges for `data`.
        Delegates to :meth:`pgmpy.causal_discovery.ExpertKnowledge.fit`.
        """
        self.fit(data)
        self.forbidden_edges = set(self.forbidden_edges_)
        return self
