from __future__ import annotations

from collections.abc import Hashable, Iterable

import numpy as np

from pgmpy.base._base import _Coregraph


class _Timeseriescoregraph(_Coregraph):
    def __init__(
        self,
        tau_max: int | None = None,
    ):
        self._set_tau_max(tau_max)
        super().__init__()

    def _set_tau_max(self, tau_max: int | None) -> None:
        if tau_max is not None:
            if isinstance(tau_max, bool) or not isinstance(tau_max, (int, np.integer)):
                raise TypeError(f"tau_max must be an integer, got {type(tau_max)}")
            if tau_max < 0:
                raise ValueError(f"tau_max must be a non-negative integer, got {tau_max}")

        self._tau_max = tau_max if tau_max is not None else 0
        self._tau_max_fixed = tau_max is not None

    @property
    def tau_max(self) -> int:
        """
        Returns the maximum lag (tau_max) for the time series graph.

        """
        return self._tau_max

    def _check_lag(self, lag: int) -> None:
        if self._tau_max_fixed:
            if lag > self._tau_max:
                raise ValueError(
                    f"Lag {lag} exceeds the maximum lag (tau_max) of {self._tau_max}. "
                    "Please set tau_max to a higher value if you want to allow this lag."
                )
        else:
            self._tau_max = max(self._tau_max, lag)

    @staticmethod
    def _normalize_node(node) -> tuple:
        if hasattr(node, "to_tuple"):
            node = node.to_tuple()
        if isinstance(node, (tuple, list)):
            if len(node) != 2:
                raise ValueError(f"Node must be (variable, lag) pair. Got : {tuple(node)}")
            var, offset = node
            if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)):
                raise TypeError(f"The time of node {(var, offset)} must be an integer, got {type(offset)}")
            if var is None or not isinstance(var, Hashable):
                raise TypeError(f"Variable names must be hashable and not None. Got {var} of type {type(var)}")
            return var, offset
        if node is None:
            raise TypeError("Variabe names must be hashable and not None. Got None")
        return (node, 0)

    def _normalize_edge(self, u, v, edge_type: str | None) -> tuple:

        u, v = self._normalize_node(u), self._normalize_node(v)
        lag = v[1] - u[1]
        if lag < 0:
            if edge_type is not None:
                markers = self._to_markers((u, v, edge_type))
                edge_type = self._to_edge_type(v, u, markers)
            u, v = v, u
            lag = -lag
        return ((u[0], -lag), (v[0], 0), edge_type)

    def add_node(self, node: Hashable, **attr) -> None:
        node = self._normalize_node(node)
        if not super().has_node(node):
            node = (node[0], 0)
        super().add_node(node, **attr)

    def add_nodes_from(self, nodes: Iterable[Hashable], **attr) -> None:
        for node in nodes:
            self.add_node(node, **attr)

    def add_edge(self, u: Hashable, v: Hashable, edge_type: str) -> None:
        u, v, edge_type = self._normalize_edge(u, v, edge_type)
        self._check_lag(-u[1])
        super().add_edge(u, v, edge_type=edge_type)

    def remove_edge(self, u: Hashable, v: Hashable, edge_type: str) -> None:
        u, v, edge_type = self._normalize_edge(u, v, edge_type)
        super().remove_edge(u, v, edge_type=edge_type)

    def has_edge(self, u: Hashable, v: Hashable, edge_type: str | None = None) -> bool:
        u, v, edge_type = self._normalize_edge(u, v, edge_type or "--")
        return super().has_edge(u, v, edge_type if edge_type != "--" else None)
