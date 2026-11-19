from __future__ import annotations

import itertools
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
        edge_list: Iterable[tuple] | None = None,
        tau_max: int | None = None,
    ):
        # Must precede the super().__init__(): the inherited consturctor adds the edges by calling self.add_edge(),
        # which is overridden below and reads this state.
        self._set_tau_max(tau_max)
        super().__init__(
            edge_list=edge_list,
        )

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

    def _to_nodes(self, nodes) -> list:
        def is_node(target) -> bool:
            return hasattr(target, "to_tuple") or (
                isinstance(target, tuple)
                and len(target) == 2
                and isinstance(target[1], (int, np.integer))
                and not isinstance(target[1], bool)
            )

        if isinstance(nodes, str) or is_node(nodes) or not hasattr(nodes, "__iter__"):
            nodes = [nodes]
        resolved = []
        for target in nodes:
            node = self._normalize_node(target)
            if is_node(target):
                resolved.append(node)
                continue
            stored = [other for other in self.nodes() if other[0] == node[0]]
            if not stored:
                raise ValueError(f"Variable {node[0]} not found in the graph.")
            resolved.extend(stored)
        return resolved

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
        if u[1] != 0:
            markers = self._to_markers((u, v, edge_type))
            if markers[u] == ">" and markers[v] != ">":
                raise ValueError(
                    f"The edge {u} {edge_type} {v} points backwards in time. A lagged link must not put an "
                    f"arrowhead on the earlier endpoint alone."
                )
        super().add_edge(u, v, edge_type=edge_type)

    def remove_edge(self, u: Hashable, v: Hashable, edge_type: str) -> None:
        u, v, edge_type = self._normalize_edge(u, v, edge_type)
        super().remove_edge(u, v, edge_type=edge_type)

    def has_edge(self, u: Hashable, v: Hashable, edge_type: str | None = None) -> bool:
        u, v, edge_type = self._normalize_edge(u, v, edge_type)
        return super().has_edge(u, v, edge_type)

    def with_role(self, role: str, variables, inplace: bool = False):
        return super().with_role(role=role, variables=self._to_nodes(variables), inplace=inplace)

    def without_role(self, role: str, variables=None, inplace: bool = False):
        if variables is None:
            return super().without_role(role=role, variables=None, inplace=inplace)
        return super().without_role(role=role, variables=self._to_nodes(variables), inplace=inplace)

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
        if encoding != "tigramite":
            raise ValueError(
                f"{cls.__name__} only supports encoding='tigramite'; got encoding={encoding!r}. The "
                f"two-dimensional encodings have no axis for the lag."
            )
        mark_to_edge_type = {
            "-->": "->",
            "<--": "<-",
            "o-o": "--",
            "<->": "<>",
            "o->": "o>",
            "<-o": "<o",
        }
        adj = np.asarray(adj)
        if adj.ndim != 3 or adj.shape[0] != adj.shape[1]:
            raise ValueError(
                f"Expected a three-dimensional array of shape (N, N, tau_max + 1), the third axis being "
                f"the lag. Got shape {adj.shape}."
            )
        n_vars, _, n_lags = adj.shape
        if variables is None:
            variables = list(range(n_vars))
        else:
            variables = list(variables)
            if len(variables) != n_vars:
                raise ValueError(f"Got {len(variables)} names for {n_vars} variables.")
            if len(set(variables)) != n_vars:
                raise ValueError(f"Variable names must be unique. Got: {variables}")

        is_string = adj.dtype.kind in "US"
        if adj.dtype.kind == "O":
            kinds = {isinstance(cell, str) for cell in adj.flat}
            if kinds == {True, False}:
                raise ValueError("adj mixes string marks and numbers. Use tigramite marks or 0/1, not both.")
            is_string = kinds == {True}
        if is_string:
            mirror = str.maketrans("<>", "><")
            for i, j in itertools.combinations(range(n_vars), 2):
                upper, lower = str(adj[i, j, 0]), str(adj[j, i, 0])
                if upper[::-1].translate(mirror) != lower:
                    raise ValueError(
                        f"adj[{i}, {j}, 0] = '{upper}' and adj[{j}, {i}, 0] = '{lower}' do not describe the same "
                        f"link. A lag-0 link is written in both cells, the second the mirror of the first."
                    )
        edge_list = []
        for i, j, tau in itertools.product(range(n_vars), range(n_vars), range(n_lags)):
            cell = adj[i, j, tau]
            if is_string:
                mark = str(cell)
                if mark == "":
                    continue
                if tau == 0 and i > j:
                    continue
                edge_type = mark_to_edge_type.get(mark)
                if edge_type is None or edge_type not in cls.SUPPORTED_EDGE_TYPES:
                    raise ValueError(
                        f"adj[{i}, {j}, {tau}] = '{mark}' is not a link {cls.__name__} can hold. Supported "
                        f"edge types are {sorted(cls.SUPPORTED_EDGE_TYPES)}."
                    )
                edge_list.append(((variables[i], -tau), (variables[j], 0), edge_type))
            else:
                if cell == 0:
                    continue
                if cell != 1:
                    raise ValueError(f"adj[{i}, {j}, {tau}] = {cell} is not a 0/1 link marker.")
                symmetric = tau == 0 and adj[j, i, 0] == 1
                edge_type = "--" if symmetric else "->"
                if edge_type not in cls.SUPPORTED_EDGE_TYPES:
                    raise ValueError(
                        f"adj[{i}, {j}, 0] and adj[{j}, {i}, 0] are both 1, which marks an unoriented link. "
                        f"{cls.__name__} supports {sorted(cls.SUPPORTED_EDGE_TYPES)}."
                    )
                if symmetric and i > j:
                    continue
                edge_list.append(((variables[i], -tau), (variables[j], 0), edge_type))

        model = cls(edge_list=edge_list, tau_max=n_lags - 1)
        present = set(model.variables)
        model.add_nodes_from([variable for variable in variables if variable not in present])
        return model

    def to_adjacency(
        self,
        encoding: str = "tigramite",
        variables: Iterable[Hashable] | None = None,
    ):
        if encoding != "tigramite":
            raise ValueError(
                f"{type(self).__name__} only supports encoding='tigramite'; got encoding={encoding!r}. The "
                f"two-dimensional encodings have no axis for the lag."
            )

        to_mark = {
            "->": "-->",
            "<-": "<--",
            "--": "o-o",
            "<>": "<->",
            "o>": "o->",
            "<o": "<-o",
        }

        variables = list(self.variables()) if variables is None else list(variables)
        if missing := (set(self.variables()) - set(variables)):
            raise ValueError(f"Variables missing from the requested order: {sorted(missing, key=str)}")
        index = {var: i for i, var in enumerate(variables)}
        out = np.full((len(variables), len(variables), self.tau_max + 1), "", dtype="<U3")
        for u, v, tau, edge_type in self.get_edges(lag_relative=True):
            if edge_type not in to_mark:
                raise ValueError(f"No tigramite mark for the edge type '{edge_type}'.")
            out[index[u], index[v], tau] = to_mark[edge_type]
            if tau == 0:
                markers = self._to_markers(((u, 0), (v, 0), edge_type))
                out[index[v], index[u], 0] = to_mark[self._to_edge_type((v, 0), (u, 0), markers)]
        return out

    def get_subgraph(self, nodes: Iterable[Hashable] | None = None, edge_types: Iterable[str] | None = None):
        nodes = set(super().nodes()) if nodes is None else {self._normalize_node(node) for node in nodes}
        if missing := (nodes - set(super().nodes())):
            raise ValueError(f"Nodes {sorted(missing)} not in graph.")

        subgraph = self.__class__()
        subgraph._set_tau_max(self.tau_max if self._tau_max_fixed else None)
        for node in super().nodes():
            if node in nodes:
                _CoreGraph.add_node(subgraph, node)
        for u, v, edge_type in super().get_edges(data=True, edge_types=edge_types):
            if u in nodes and v in nodes:
                subgraph.add_edge(u, v, edge_type=edge_type)

        for role, variables in self.get_role_dict().items():
            if retained := [node for node in variables if node in nodes]:
                subgraph.with_role(role=role, variables=retained, inplace=True)
        return subgraph

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.tau_max == other.tau_max

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.tau_max))
