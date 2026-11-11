from __future__ import annotations

from collections.abc import Hashable, Iterable

import numpy as np

from pgmpy.base._base import _CoreGraph


class _TimeSeriesCoreGraph(_CoreGraph):
    """
    Base class for all time-series graph types in pgmpy.

    This class generalizes over a *node* vocabulary: every endpoint is a ``(variable, time)`` pair, and the
    invariants are stated in terms of lags rather than endpoint markers. Subclasses (e.g. ``TimeSeriesDAG``,
    ``TimeSeriesPDAG``) are expected to declare their own ``SUPPORTED_EDGE_TYPES`` exactly as the static
    classes do.

    Parameters
    ----------
    tau_max : int, optional (default: inferred)
        The maximum lag the graph accepts. If None, it starts at 0 and grows to fit whatever is added.

    Docs in is progress.
    """

    def __init__(
        self,
        tau_max: int | None = None,
    ):
        # Must precede the super().__init__(): the inherited consturctor adds the edges by calling self.add_edge(),
        # which is overridden below and reads this state.
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

    def variables(self) -> list:
        """
        Returns the list of distinct variable names, with times dropped and duplicates removed.

        Returns
        -------
        list
            List of variables in the time series graph. Sorting by ``str`` rather than natural order keeps
            mixed types comparable.
        """
        return sorted({var for var, _ in super().nodes()}, key=str)

    def get_parents(self, node: Hashable) -> set:
        return super().get_parents(self._normalize_node(node))

    def get_children(self, node: Hashable) -> set:
        return super().get_children(self._normalize_node(node))

    def get_edges(
        self,
        data: bool = True,
        edge_types: Iterable[str] | None = None,
        lag: int | Iterable[int] | None = None,
        lag_relative: bool = False,
    ) -> list[tuple]:
        if lag is None:
            lags = None
        elif isinstance(lag, (int, np.integer)) and not isinstance(lag, bool):
            lags = {int(lag)}
        else:
            lags = {int(value) for value in lag}
        edges = []
        for u, v, edge_type in super().get_edges(data=True, edge_types=edge_types):
            if u[1] > v[1]:
                markers = self._to_markers((u, v, edge_type))
                u, v, edge_type = v, u, self._to_edge_type(v, u, markers)
            edge_lag = -u[1]
            if lags is not None and edge_lag not in lags:
                continue
            if lag_relative:
                edges.append((u[0], v[0], edge_lag, edge_type) if data else (u[0], v[0], edge_lag))
            else:
                edges.append((u, v, edge_type) if data else (u, v))
        return edges

    def get_lag(self, u: Hashable, v: Hashable) -> int:
        u_stored, v_stored, _ = self._normalize_edge(u, v, None)
        if not super().has_edge(u_stored, v_stored):
            raise ValueError(f"No such edge: {u} -- {v}")
        return -u_stored[1]

    def get_parents_by_lag(self, variable: Hashable) -> dict[int, list]:
        variable = self._normalize_node(variable)[0]
        if not super().has_node((variable, 0)):
            raise ValueError(f"Variable not in the graph: {variable}.")
        parents: dict[int, list] = {}
        for parent in _CoreGraph.get_parents(self, (variable, 0)):
            parents.setdefault(-parent[1], []).append(parent[0])
        return {lag: sorted(variables, key=str) for lag, variables in sorted(parents.items())}

    def from_adjacency(cls, adj, encoding: str = "tigramite", variables: Iterable[Hashable] | None = None):
        """
        In progress.
        """
